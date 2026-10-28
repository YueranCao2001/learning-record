"""Confirm frozen prompts on new missing-field groups; reuse generation and scoring."""
import argparse
from collections import Counter
from pathlib import Path
from contract import validate_target
from prompt_diagnostic import ROOT, read, metrics


def compose():
    protocol = read(ROOT / "data/prompt-confirmation-v1.json")
    prior = ROOT / "results" / protocol["prior_run"]
    saved_prompts = read(prior / "prompts.json")
    prompts = {c: saved_prompts[c] for c in protocol["conditions"]}
    prior_rows = read(prior / "cases.json")
    lookup = {r["id"]: r for r in prior_rows}
    rows = [dict(lookup[n], cohort="replay") for n in protocol["replay_ids"]]
    old = prior_rows + read(ROOT / "results/2026-10-23-boundary-v2/data.json")["rows"]
    old_groups, ids, texts = {r["group"] for r in old}, {r["id"] for r in old}, {r["text"] for r in old}
    used_groups = set()
    for group in protocol["groups"]:
        if group["group"] in old_groups | used_groups:
            raise ValueError("source group overlaps")
        used_groups.add(group["group"])
        for state in protocol["states"]:
            time_present = state in ("complete", "missing-location")
            location_present = state in ("complete", "missing-time")
            text = group["template"].format(event=group["event"],
                time_clause=group["time_present" if time_present else "time_missing"],
                location_clause=group["location_present" if location_present else "location_missing"])
            row = {"id": group["id"] + "-" + state, "group": group["group"], "family": group["template_family"],
                   "state": state, "cohort": "new-confirmation", "split": "dev", "text": text,
                   "expected": {"event": group["event"], "time": group["time"] if time_present else None,
                                "location": group["location"] if location_present else None}}
            validate_target(row["expected"])
            if row["id"] in ids or text in texts:
                raise ValueError("duplicate or overlapping id/text")
            if any(v is not None and v not in text for v in row["expected"].values()):
                raise ValueError("nonliteral label")
            ids.add(row["id"]); texts.add(text); rows.append(row)
    if len(rows) != protocol["expected_rows"] or len(rows) - len(protocol["replay_ids"]) != protocol["new_count"]:
        raise ValueError("case budget mismatch")
    if len(rows) * len(prompts) * len(protocol["models"]) != protocol["expected_generation_calls"]:
        raise ValueError("generation budget mismatch")
    return protocol, prompts, rows


def aggregate(rows, outputs):
    _, records = metrics(rows, outputs)
    result = {"count": len(rows), "exact": sum(r["exact"] for r in records),
              "json_valid": sum(o["score"]["json_valid"] for o in outputs),
              "schema_valid": sum(o["score"]["schema_valid"] for o in outputs),
              "fields_correct": {k: sum(r["fields"][k] for r in records) for k in ("event", "time", "location")}}
    errors = []
    for row, output, record in zip(rows, outputs, records):
        if not output["score"]["schema_valid"]:
            errors.append({"id": row["id"], "kind": "format-or-schema", "field": None})
            continue
        for key, expected in row["expected"].items():
            actual = record["parsed"][key]
            if actual != expected:
                kind = "missing-to-value" if expected is None else "value-to-null" if actual is None else "wrong-span"
                errors.append({"id": row["id"], "field": key, "kind": kind, "expected": expected, "actual": actual})
    result["error_counts"] = dict(Counter(e["kind"] for e in errors))
    return result, records, errors


def review(folder):
    folder = Path(folder)
    protocol, rows = read(folder / "protocol.json"), read(folder / "cases.json")
    prior = ROOT / "results" / protocol["prior_run"]
    summaries, transitions, replay, errors, details = {}, {}, {}, {}, {}
    calls = 0
    for model in protocol["models"]:
        summaries[model], replay[model], errors[model], details[model] = {}, {}, {}, {}
        original = None
        for condition in protocol["conditions"]:
            outputs = read(folder / f"{model}-{condition}.json")
            metrics(rows, outputs)  # Also rejects missing/reordered rows or forged scores.
            calls += len(outputs)
            new_pairs = [(r, o) for r, o in zip(rows, outputs) if r["cohort"] == "new-confirmation"]
            new_rows, new_outputs = map(list, zip(*new_pairs))
            summary, records, failures = aggregate(new_rows, new_outputs)
            summary["by_state"] = {}
            summary["by_group"] = {}
            for label, field, values in (("by_state", "state", protocol["states"]),
                                         ("by_group", "group", [g["group"] for g in protocol["groups"]])):
                for value in values:
                    subset = [(r, o) for r, o in new_pairs if r[field] == value]
                    summary[label][value] = aggregate(*map(list, zip(*subset)))[0]
            summaries[model][condition], errors[model][condition], details[model][condition] = summary, failures, records
            if condition == "original":
                original = records
            else:
                transitions[model] = {"fixed_ids": [b["id"] for a, b in zip(original, records) if not a["exact"] and b["exact"]],
                                      "regressed_ids": [b["id"] for a, b in zip(original, records) if a["exact"] and not b["exact"]]}
            old_outputs = {r["id"]: r for r in read(prior / f"{model}-{condition}.json")}
            replay_rows = [(r, o) for r, o in zip(rows, outputs) if r["cohort"] == "replay"]
            replay[model][condition] = {"count": len(replay_rows), "exact": sum(o["score"]["exact"] for _, o in replay_rows),
                                       "tokens_equal": {r["id"]: o["token_ids"] == old_outputs[r["id"]]["token_ids"] for r, o in replay_rows}}
    return {"training_steps": 0, "human_review": "pending", "promote_to_main": False,
            "total_generation_calls": calls, "new_source_groups": len(protocol["groups"]), "new_cases": protocol["new_count"],
            "all_replay_tokens_reproduce": all(v for m in replay.values() for c in m.values() for v in c["tokens_equal"].values()),
            "summaries_new_only": summaries, "transitions_new_only": transitions, "replay": replay,
            "errors_new_only": errors, "details_new_only": details}


def main():
    import gc
    from importlib.metadata import version
    import torch
    from peft import PeftModel
    from transformers import AutoTokenizer, set_seed
    from audit_evaluation import state_hash
    from train_smoke import REVISION, SNAPSHOT, digest, generate, load_base, prompt_ids, save_json
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    args = parser.parse_args()
    if Path(args.run).name != args.run or args.run in (".", ".."):
        raise ValueError("single new directory name required")
    protocol, prompts, rows = compose()
    assert protocol["base_revision"] == REVISION and protocol["training_steps"] == 0
    assert protocol["generation"] == {"do_sample": False, "max_new_tokens": 128}
    upload = read(ROOT / "results/2026-10-23-boundary-v2/candidate/upload.json")
    snapshot = Path(upload["downloaded_snapshot"])
    assert upload["commit"] == protocol["adapter_revision"]
    assert all(digest(snapshot / n) == sha for n, sha in upload["files_sha256"].items())
    folder = ROOT / "results" / args.run
    folder.mkdir(parents=True, exist_ok=False)
    for name, value in (("protocol", protocol), ("prompts", prompts), ("cases", rows)):
        save_json(folder / (name + ".json"), value)
    sources = ("confirm_prompt.py", "prompt_diagnostic.py", "train_smoke.py", "contract.py", "audit_evaluation.py",
               "data/prompt-confirmation-v1.json", f"results/{protocol['prior_run']}/prompts.json",
               f"results/{protocol['prior_run']}/cases.json", "results/2026-10-23-boundary-v2/data.json")
    save_json(folder / "manifest.json", {"base_revision": REVISION, "adapter_revision": upload["commit"],
              "adapter_sha256": upload["files_sha256"]["adapter_model.safetensors"],
              "source_sha256": {n: digest(ROOT / n) for n in sources}, "training_steps": 0,
              "prompts_unchanged_from_prior": True, "human_review": "pending", "new_cases": 16, "replay_cases": 4})
    set_seed(protocol["seed"])
    assert torch.cuda.is_available()
    save_json(folder / "environment.json", {"packages": {n: version(n) for n in ("torch", "transformers", "peft", "huggingface_hub")},
              "gpu": torch.cuda.get_device_name(0), "cuda": torch.version.cuda, "new_dependencies_installed": False})
    tokenizer = AutoTokenizer.from_pretrained(SNAPSHOT, local_files_only=True)
    save_json(folder / "prompt-tokens.json", {c: [{"id": r["id"], "input_ids": prompt_ids(tokenizer, {"instruction": p}, r)}
                                                   for r in rows] for c, p in prompts.items()})
    for model_name in protocol["models"]:
        model = load_base()
        if model_name == "candidate":
            model = PeftModel.from_pretrained(model, snapshot, local_files_only=True, is_trainable=False)
        model.eval()
        before, hashes = state_hash(model), {}
        for condition in protocol["conditions"]:
            print(model_name, condition, "16 new cases + 4 replay", flush=True)
            outputs = generate(model, tokenizer, {"instruction": prompts[condition], "rows": rows})
            save_json(folder / f"{model_name}-{condition}.json", outputs)
            hashes[condition] = state_hash(model)
            assert hashes[condition] == before
        save_json(folder / f"{model_name}-weights.json", {"before": before, "after_each_condition": hashes, "unchanged": True})
        del model
        gc.collect(); torch.cuda.empty_cache()
    report = review(folder)
    assert report["total_generation_calls"] == protocol["expected_generation_calls"]
    save_json(folder / "review.json", report)
    print("Complete", report["total_generation_calls"], "generations; replay", report["all_replay_tokens_reproduce"], flush=True)


if __name__ == "__main__":
    main()
