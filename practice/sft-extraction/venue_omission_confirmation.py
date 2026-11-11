"""Confirm venue omissions across new notices with fixed cached decoding."""
import argparse
import gc
from importlib.metadata import version
from pathlib import Path

from confirm_prompt import aggregate
from contract import validate_target
from prompt_diagnostic import ROOT, read
from time_regression import TRAINED, WARM

SPEC = "data/venue-omission-confirmation-v1.json"


def prepare(spec, prior_data):
    rows = []
    for source in spec["sources"]:
        venue = f"{source['venue_cue']}：{source['venue']}"
        for state, time in (("explicit", source["time"]), ("undetermined", spec["missing_status"]), ("removed", None)):
            time_clause = f"{source['time_cue']}：{time}"
            orders = {"venue-only": (venue,)} if time is None else {
                "time-first": (time_clause, venue), "venue-first": (venue, time_clause)
            }
            for order, clauses in orders.items():
                notice = source["template"].format(event=source["event"], fields=source["separator"].join(clauses))
                rows.append({"id": f"{source['id']}-{state}-{order}", "group": source["id"], "split": "dev",
                             "cohort": "new-venue-omission-confirmation", "time_state": state, "order": order,
                             "text": notice, "expected": {"event": source["event"], "time": time if state == "explicit" else None,
                                                          "location": source["venue"]}})
    assert len(rows) == 15 and len({row["group"] for row in rows}) == 3
    lookup = {row["id"]: row for row in prior_data["rows"]}
    rows += [lookup[identity] for identity in spec["replay_ids"]]
    assert len({r["id"] for r in rows}) == len({r["text"] for r in rows}) == 18
    for row in rows:
        validate_target(row["expected"])
        assert row["split"] == "dev" and row["cohort"]
        assert all(v is None or v in row["text"] for v in row["expected"].values())
    return {"instruction": prior_data["instruction"], "rows": rows}


def main():
    import torch
    from peft import PeftModel
    from transformers import AutoTokenizer
    from audit_evaluation import state_hash
    from evaluate_fit import teacher_forced
    from train_smoke import REVISION, SNAPSHOT, digest, fixture, generate, load_base, save_json

    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    args = parser.parse_args()
    if Path(args.run).name != args.run or args.run in (".", ".."):
        raise ValueError("new run directory required")
    spec = read(ROOT / SPEC)
    prior = ROOT / "results" / spec["source_run"]
    data = prepare(spec, read(prior / "cases.json"))
    historical = read(ROOT / f"results/{WARM}/data.json")["rows"] + read(prior / "cases.json")["rows"]
    for key in ("id", "group", "text"):
        assert not {r[key] for r in data["rows"][:15]} & {r[key] for r in historical}
    prior_manifest = read(prior / "manifest.json")
    runs = {"warm-start": (WARM, "extended"), "canonical": (TRAINED, "canonical"), "target": (TRAINED, "target")}
    paths = {c: ROOT / "artifacts" / run / arm for c, (run, arm) in runs.items()}
    hashes = {c: digest(path / "adapter_model.safetensors") for c, path in paths.items()}
    assert hashes == prior_manifest["adapter_files_sha256"]
    sources = [SPEC, "venue_omission_confirmation.py", "time_regression.py", "evaluate_fit.py", "fit_metrics.py",
               "train_smoke.py", "contract.py", "confirm_prompt.py", "prompt_diagnostic.py", "audit_evaluation.py",
               "annotation-rules-v1.md", f"results/{WARM}/data.json"]
    sources += [(prior / n).relative_to(ROOT).as_posix() for n in ("manifest.json", "cases.json", "review.json")]
    sources += [(prior / f"{c}-generation.json").relative_to(ROOT).as_posix() for c in runs]
    manifest = {"source_sha256": {n: digest(ROOT / n) for n in sources}, "base_revision": REVISION,
                "adapter_files_sha256": hashes,
                "published_reference_adapter_sha256": prior_manifest["published_reference_adapter_sha256"],
                "generation": {**prior_manifest["generation"], "use_cache": True}}
    environment = {"gpu": torch.cuda.get_device_name(0), "cuda": torch.version.cuda,
                   "packages": {n: version(n) for n in ("torch", "transformers", "peft", "huggingface_hub")}}
    tokenizer = AutoTokenizer.from_pretrained(SNAPSHOT, local_files_only=True)
    fixtures = {row["id"]: fixture(tokenizer, data, row) for row in data["rows"]}
    output = ROOT / "results" / args.run
    output.mkdir(parents=True, exist_ok=False)
    for name, value in (("protocol", spec), ("cases", data), ("manifest", manifest), ("environment", environment)):
        save_json(output / f"{name}.json", value)

    results = {}
    for condition in spec["conditions"]:
        model = PeftModel.from_pretrained(load_base(), paths[condition], local_files_only=True, is_trainable=False)
        model.eval()
        assert not any(p.requires_grad for p in model.parameters())
        before = state_hash(model)
        generated = generate(model, tokenizer, data)
        aggregate(data["rows"], generated)
        summary, details, errors = aggregate(data["rows"][:15], generated[:15])
        previous = {r["id"]: r["token_ids"] for r in read(prior / f"{condition}-generation.json")}
        replays = {o["id"]: o["token_ids"] == previous[o["id"]] for o in generated[15:]}
        assert set(replays) == set(spec["replay_ids"]) and all(replays.values())
        omission_ids = {e["id"] for e in errors if e["kind"] == "value-to-null"}
        teachers = [teacher_forced(model, tokenizer, fixtures[o["id"]], o)
                    for o in generated[:15] if o["id"] in omission_ids]
        after = state_hash(model)
        assert before == after
        save_json(output / f"{condition}-generation.json", generated)
        save_json(output / f"{condition}-teacher.json", teachers)
        results[condition] = {"summary": summary, "details": details, "errors": errors, "token_replays": replays,
                              "generation_calls": len(generated), "teacher_forced_forwards": len(teachers),
                              "parameters_before": before, "parameters_after": after}
        print(condition, summary, "new omissions checked", len(teachers), flush=True)
        del model
        gc.collect()
        torch.cuda.empty_cache()
    save_json(output / "review.json", {"training_steps": 0, "human_review": "pending", "promote_to_main": False,
              "weights_unchanged": True, "generation_calls": sum(c["generation_calls"] for c in results.values()),
              "teacher_forced_forwards": sum(c["teacher_forced_forwards"] for c in results.values()),
              "known_token_replays": sum(len(c["token_replays"]) for c in results.values()),
              "new_cases": 15, "new_source_groups": 3, "results": results})


if __name__ == "__main__":
    main()
