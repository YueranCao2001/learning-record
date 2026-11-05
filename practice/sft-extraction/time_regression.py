"""Replay a known time regression and vary its neighboring location clause."""
import argparse
import gc
from importlib.metadata import version
from pathlib import Path

from confirm_prompt import aggregate
from contract import validate_target
from prompt_diagnostic import ROOT, read

SPEC = "data/time-regression-v1.json"
WARM = "2026-11-02-optimization-budget-v1"
TRAINED = "2026-11-04-venue-cue-comparison-v1"


def prepare(spec, source):
    by_id = {r["id"]: r for r in source["rows"] if r["group"] == spec["source_group"]}
    rows = [by_id[i] for i in spec["frozen_ids"]]
    assert len(rows) == 4 and len({(r["expected"]["time"] is None, r["expected"]["location"] is None) for r in rows}) == 4
    anchor = by_id[spec["anchor_id"]]
    heading, location_line, time_line, closing = anchor["text"].splitlines()
    cue, status = spec["original_cue"], spec["original_status"]
    assert location_line == f"一、{cue}：{status}"
    assert time_line == f"二、日程：{anchor['expected']['time']}" and anchor["expected"]["location"] is None
    alternatives = {
        "cue-only": (f"一、{spec['replacement_cue']}：{status}", time_line),
        "status-only": (f"一、{cue}：{spec['replacement_status']}", time_line),
        "cue-and-status": (f"一、{spec['replacement_cue']}：{spec['replacement_status']}", time_line),
        "time-first": ("一、" + time_line.removeprefix("二、"), "二、" + location_line.removeprefix("一、")),
    }
    rows += [{**anchor, "id": f"time-regression-{name}", "cohort": "time-regression",
              "text": "\n".join((heading, *lines, closing))} for name, lines in alternatives.items()]
    assert len({r["id"] for r in rows}) == len({r["text"] for r in rows}) == 8
    for row in rows:
        validate_target(row["expected"])
        assert row["split"] == "dev"
        assert all(v is None or v in row["text"] for v in row["expected"].values())
    return {"instruction": source["instruction"], "rows": rows}


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
    records = {"warm-start": (WARM, "extended"), "canonical": (TRAINED, "canonical"), "target": (TRAINED, "target")}
    data_file = f"results/{WARM}/data.json"
    data = prepare(spec, read(ROOT / data_file))
    output = ROOT / "results" / args.run
    output.mkdir(parents=True, exist_ok=False)
    paths, hashes, references = {}, {}, {}
    sources = [SPEC, data_file, "time_regression.py", "train_smoke.py", "contract.py", "confirm_prompt.py",
               "prompt_diagnostic.py", "audit_evaluation.py", "annotation-rules-v1.md", f"results/{TRAINED}/manifest.json"]
    for condition, (run, arm) in records.items():
        record = f"results/{run}/{arm}"
        paths[condition] = ROOT / "artifacts" / run / arm
        hashes[condition] = digest(paths[condition] / "adapter_model.safetensors")
        assert hashes[condition] == read(ROOT / record / "summary.json")["adapter_file_sha256"]
        references[condition] = {r["id"]: r["token_ids"] for r in read(ROOT / record / "after.json")}
        sources += [record + "/summary.json", record + "/after.json"]
    published = read(ROOT / f"results/{TRAINED}/manifest.json")["published_reference_adapter_sha256"]
    manifest = {"source_sha256": {n: digest(ROOT / n) for n in sources}, "base_revision": REVISION,
                "adapter_files_sha256": hashes, "published_reference_adapter_sha256": published,
                "decode": {"do_sample": False, "max_new_tokens": 128, "enable_thinking": False}}
    environment = {"gpu": torch.cuda.get_device_name(0), "cuda": torch.version.cuda,
                   "packages": {n: version(n) for n in ("torch", "transformers", "peft", "huggingface_hub")}}
    for name, value in (("protocol", spec), ("cases", data), ("manifest", manifest), ("environment", environment)):
        save_json(output / f"{name}.json", value)

    set_seed(42)
    results = {}
    tokenizer = AutoTokenizer.from_pretrained(SNAPSHOT, local_files_only=True)
    for condition in spec["conditions"]:
        model = PeftModel.from_pretrained(load_base(), paths[condition], local_files_only=True, is_trainable=False)
        model.eval()
        assert not any(p.requires_grad for p in model.parameters())
        before = state_hash(model)
        generated = generate(model, tokenizer, data)
        after = state_hash(model)
        assert before == after
        scores, details, errors = aggregate(data["rows"], generated)
        replays = {r["id"]: r["token_ids"] == references[condition][r["id"]]
                   for r in generated if r["id"] in spec["frozen_ids"]}
        assert set(replays) == set(spec["frozen_ids"]) and all(replays.values())
        save_json(output / f"{condition}-generation.json", generated)
        results[condition] = {"scores": scores, "details": details, "errors": errors, "token_replays": replays,
                              "parameter_sha256_before": before, "parameter_sha256_after": after}
        print(condition, scores, flush=True)
        del model
        gc.collect()
        torch.cuda.empty_cache()
    save_json(output / "review.json", {"training_steps": 0, "human_review": "pending", "promote_to_main": False,
              "weights_unchanged": True, "known_token_replays": sum(len(r["token_replays"]) for r in results.values()),
              "generation_calls": sum(r["scores"]["count"] for r in results.values()), "source_groups": 1,
              "results": results})


if __name__ == "__main__":
    main()
