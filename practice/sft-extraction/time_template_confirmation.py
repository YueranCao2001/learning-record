"""Measure cross-field omissions across new synthetic templates with fixed weights."""
import argparse
import gc
from importlib.metadata import version
from pathlib import Path

from confirm_prompt import aggregate
from contract import validate_target
from prompt_diagnostic import ROOT, read
from time_regression import WARM, TRAINED

SPEC = "data/time-template-confirmation-v1.json"
PRIOR = "2026-11-05-time-regression-v1"


def prepare(spec, historical, prior):
    rows = []
    for source in spec["sources"]:
        for cue_index, cue in enumerate(spec["cues"]):
            for state, missing_time, missing_location in (
                ("complete", False, False), ("missing-time", True, False),
                ("missing-location", False, True), ("missing-both", True, True)
            ):
                time = spec["missing_status"] if missing_time else source["time"]
                location = spec["missing_status"] if missing_location else source["venue"]
                text = source["template"].format(event=source["event"],
                            location_clause=f"{cue}：{location}", time_clause=f"{source['time_cue']}：{time}")
                rows.append({"id": f"{source['id']}-cue{cue_index}-{state}", "group": source["id"], "cue": cue,
                             "state": state, "split": "dev", "cohort": "time-template-confirmation", "text": text,
                             "expected": {"event": source["event"], "time": None if missing_time else time,
                                          "location": None if missing_location else location}})
    for key in ("id", "group", "text"):
        assert not {r[key] for r in rows} & {r[key] for r in historical["rows"] + prior["rows"]}
    replays = [r for r in prior["rows"] if r["id"] in ("confirm-02-missing-location", "time-regression-time-first")]
    assert len(rows) == 24 and len(replays) == 2
    rows += replays
    assert len({r["id"] for r in rows}) == len({r["text"] for r in rows}) == 26
    for row in rows:
        validate_target(row["expected"])
        assert all(v is None or v in row["text"] for v in row["expected"].values())
    return {"instruction": historical["instruction"], "rows": rows}


def main():
    import torch
    from peft import PeftModel
    from transformers import AutoTokenizer, set_seed
    from audit_evaluation import state_hash
    from train_smoke import REVISION, SNAPSHOT, digest, generate, load_base, save_json

    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    args = parser.parse_args()
    if Path(args.run).name != args.run or args.run in (".", ".."):
        raise ValueError("new run directory required")
    spec = read(ROOT / SPEC)
    prior_dir = ROOT / "results" / PRIOR
    data = prepare(spec, read(ROOT / f"results/{WARM}/data.json"), read(prior_dir / "cases.json"))
    output = ROOT / "results" / args.run
    output.mkdir(parents=True, exist_ok=False)
    records = {"warm-start": (WARM, "extended"), "canonical": (TRAINED, "canonical"), "target": (TRAINED, "target")}
    paths, hashes = {}, {}
    sources = [SPEC, "time_template_confirmation.py", "time_regression.py", "train_smoke.py", "audit_evaluation.py",
               "confirm_prompt.py", "prompt_diagnostic.py", "contract.py", "annotation-rules-v1.md",
               f"results/{WARM}/data.json", f"results/{PRIOR}/cases.json", f"results/{PRIOR}/manifest.json"]
    for condition, (run, arm) in records.items():
        paths[condition] = ROOT / "artifacts" / run / arm
        summary_file = f"results/{run}/{arm}/summary.json"
        hashes[condition] = digest(paths[condition] / "adapter_model.safetensors")
        assert hashes[condition] == read(ROOT / summary_file)["adapter_file_sha256"]
        sources += [summary_file, f"results/{PRIOR}/{condition}-generation.json"]
    manifest = {"source_sha256": {n: digest(ROOT / n) for n in sources}, "base_revision": REVISION,
                "adapter_files_sha256": hashes,
                "published_reference_adapter_sha256": read(prior_dir / "manifest.json")["published_reference_adapter_sha256"],
                "generation": {"do_sample": False, "max_new_tokens": 128, "enable_thinking": False}}
    environment = {"gpu": torch.cuda.get_device_name(0), "cuda": torch.version.cuda,
                   "packages": {n: version(n) for n in ("torch", "transformers", "peft", "huggingface_hub")}}
    for name, value in (("protocol", spec), ("cases", data), ("manifest", manifest), ("environment", environment)):
        save_json(output / f"{name}.json", value)

    results = {}
    tokenizer = AutoTokenizer.from_pretrained(SNAPSHOT, local_files_only=True)
    set_seed(42)
    for condition in spec["conditions"]:
        model = PeftModel.from_pretrained(load_base(), paths[condition], local_files_only=True, is_trainable=False)
        model.eval()
        assert not any(p.requires_grad for p in model.parameters())
        before = state_hash(model)
        generated = generate(model, tokenizer, data)
        after = state_hash(model)
        assert before == after
        save_json(output / f"{condition}-generation.json", generated)
        aggregate(data["rows"], generated)
        summary, details, errors = aggregate(data["rows"][:24], generated[:24])
        references = {r["id"]: r["token_ids"] for r in read(prior_dir / f"{condition}-generation.json")}
        replays = {r["id"]: r["token_ids"] == references[r["id"]] for r in generated[24:]}
        assert len(replays) == 2 and all(replays.values())
        by_state = {}
        for state in ("complete", "missing-time", "missing-location", "missing-both"):
            selected = [(r, o) for r, o in zip(data["rows"][:24], generated[:24]) if r["state"] == state]
            by_state[state] = aggregate(*map(list, zip(*selected)))[0]
        results[condition] = {"summary": summary, "by_state": by_state, "details": details, "errors": errors,
                              "token_replays": replays, "parameters_before": before, "parameters_after": after}
        print(condition, summary, flush=True)
        del model
        gc.collect()
        torch.cuda.empty_cache()
    save_json(output / "review.json", {"training_steps": 0, "human_review": "pending", "promote_to_main": False,
              "weights_unchanged": True, "generation_calls": 78, "new_cases": 24, "new_source_groups": 3,
              "known_token_replays": 6, "results": results})


if __name__ == "__main__":
    main()
