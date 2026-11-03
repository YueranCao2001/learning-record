"""Prepare paired venue labels and measure fixed adapters on the development pairs."""
import argparse
import gc
from importlib.metadata import version
from itertools import product
from pathlib import Path

from confirm_prompt import aggregate
from contract import validate_pilot_data
from prompt_diagnostic import ROOT, read

PROTOCOL = "data/venue-pairs-v1.json"
HISTORY = "results/2026-11-02-optimization-budget-v1/data.json"
DIAGNOSTIC = "results/2026-11-02-absence-diagnostic-v1/cases.json"
ADAPTER_RECORDS = {
    "baseline": "results/2026-10-23-boundary-v2/candidate/upload.json",
    "short": "results/2026-10-30-repair-comparison-v1/repair/summary.json",
    "extended": "results/2026-11-02-optimization-budget-v1/extended/summary.json",
}


def prepare(protocol, history, diagnostic):
    sources = protocol["sources"]
    for key in ("id", "family", "template", "event", "venue"):
        if len({source[key] for source in sources}) != len(sources):
            raise ValueError(f"duplicate source {key}")
    cells = set(product(protocol["cues"], protocol["statuses"]))
    for split, cohort in (("train", "candidate"), ("dev", "matched")):
        observed = [(s["cue"], s["status"]) for s in sources if (s["split"], s["cohort"]) == (split, cohort)]
        if len(observed) != len(cells) or set(observed) != cells:
            raise ValueError(f"incomplete or repeated coverage: {cohort}")

    rows = []
    for source in sources:
        cue_seen = source["cue"] in protocol["cues"]
        status_seen = source["status"] in protocol["statuses"]
        allowed = {("train", "candidate"): (True, True), ("dev", "matched"): (True, True),
                   ("dev", "new-cue"): (False, True), ("dev", "new-status"): (True, False)}
        if allowed.get((source["split"], source["cohort"])) != (cue_seen, status_seen):
            raise ValueError(f"wrong coverage category: {source['id']}")
        if source["template"].count("{place}") != 1:
            raise ValueError(f"expected one replaceable location clause: {source['id']}")
        for variant, location in (("missing", None), ("explicit", source["venue"])):
            clause = source["cue"] + ("为" + location if location is not None else source["status"])
            text = source["template"].format(event=source["event"], time=source["time"], place=clause)
            if location is None and source["venue"] in text:
                raise ValueError(f"missing variant contains the venue: {source['id']}")
            rows.append({"id": f"{source['id']}-{variant}", "group": source["id"], "split": source["split"],
                         "family": source["family"], "cohort": source["cohort"], "cue": source["cue"],
                         "status": source["status"], "variant": variant, "text": text,
                         "expected": {"event": source["event"], "time": source["time"], "location": location}})
    data = validate_pilot_data({"instruction": history["instruction"], "rows": rows})
    previous = history["rows"] + diagnostic
    for key in ("id", "group", "text", "family"):
        if {r[key] for r in rows} & {r.get(key) for r in previous}:
            raise ValueError(f"historical overlap: {key}")
    return data


def summarize(rows, outputs):
    summary, records, errors = aggregate(rows, outputs)
    pairs = {}
    for row, record in zip(rows, records):
        pair = pairs.setdefault(row["group"], {"cohort": row["cohort"]})
        pair[row["variant"]] = {"id": row["id"], "location_correct": record["fields"]["location"],
                                "exact": record["exact"], "parsed": record["parsed"]}
    summary["location_by_variant"] = {
        variant: sum(pair[variant]["location_correct"] for pair in pairs.values())
        for variant in ("missing", "explicit")
    }
    summary["both_location_correct"] = sum(
        pair["missing"]["location_correct"] and pair["explicit"]["location_correct"] for pair in pairs.values()
    )
    summary["both_exact"] = sum(pair["missing"]["exact"] and pair["explicit"]["exact"] for pair in pairs.values())
    return {"summary": summary, "pairs": pairs, "errors": errors}


def main():
    import torch
    from peft import PeftModel
    from transformers import AutoTokenizer, set_seed
    from audit_evaluation import state_hash
    from train_smoke import REVISION, SNAPSHOT, digest, fixture, generate, load_base, save_json

    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    args = parser.parse_args()
    if Path(args.run).name != args.run or args.run in (".", ".."):
        raise ValueError("new directory required")
    protocol = read(ROOT / PROTOCOL)
    data = prepare(protocol, read(ROOT / HISTORY), read(ROOT / DIAGNOSTIC))
    train = [r for r in data["rows"] if r["split"] == "train"]
    dev = [r for r in data["rows"] if r["split"] == "dev"]
    tokenizer = AutoTokenizer.from_pretrained(SNAPSHOT, local_files_only=True)
    fixtures = [fixture(tokenizer, data, r) for r in train]
    tokens = [{"id": f["id"], "prefix_tokens": f["prefix_tokens"], "target_tokens": f["target_tokens"],
               "sequence_tokens": len(f["input_ids"])} for f in fixtures]

    baseline = read(ROOT / ADAPTER_RECORDS["baseline"])
    paths = {"baseline": Path(baseline["downloaded_snapshot"]),
             "short": ROOT / "artifacts/2026-10-30-repair-comparison-v1/repair",
             "extended": ROOT / "artifacts/2026-11-02-optimization-budget-v1/extended"}
    hashes = {c: digest(path / "adapter_model.safetensors") for c, path in paths.items()}
    expected = {"baseline": baseline["files_sha256"]["adapter_model.safetensors"]}
    expected.update({c: read(ROOT / ADAPTER_RECORDS[c])["adapter_file_sha256"] for c in ("short", "extended")})
    if hashes != expected:
        raise ValueError("adapter identity mismatch")
    folder = ROOT / "results" / args.run
    folder.mkdir(parents=True, exist_ok=False)
    inputs = (PROTOCOL, HISTORY, DIAGNOSTIC, *ADAPTER_RECORDS.values(), "venue_pairs.py", "contract.py",
              "confirm_prompt.py", "prompt_diagnostic.py", "train_smoke.py", "audit_evaluation.py",
              "annotation-rules-v1.md")
    manifest = {"source_sha256": {name: digest(ROOT / name) for name in inputs}, "base_revision": REVISION,
                "adapter_files_sha256": hashes, "published_reference_adapter_sha256": hashes["baseline"],
                "decode": {"do_sample": False, "max_new_tokens": 128, "enable_thinking": False}}
    environment = {"gpu": torch.cuda.get_device_name(0), "cuda": torch.version.cuda,
                   "packages": {n: version(n) for n in ("torch", "transformers", "peft", "huggingface_hub")}}
    for name, value in (("protocol", protocol), ("cases", data), ("token-budget", tokens),
                        ("manifest", manifest), ("environment", environment)):
        save_json(folder / f"{name}.json", value)

    set_seed(42)
    results, weight_checks = {}, {}
    for condition in protocol["conditions"]:
        model = PeftModel.from_pretrained(load_base(), paths[condition], local_files_only=True, is_trainable=False)
        model.eval()
        assert not any(p.requires_grad for p in model.parameters())
        before = state_hash(model)
        outputs = generate(model, tokenizer, {"instruction": data["instruction"], "rows": dev})
        after = state_hash(model)
        assert before == after
        weight_checks[condition] = {"before": before, "after": after}
        save_json(folder / f"{condition}-generation.json", outputs)
        results[condition] = summarize(dev, outputs)
        print(condition, results[condition]["summary"], flush=True)
        del model
        gc.collect()
        torch.cuda.empty_cache()
    save_json(folder / "review.json",
              {"human_review": "pending", "training_steps": 0, "promote_to_main": False,
               "weights_unchanged": True, "weight_checks": weight_checks, "provenance": protocol["provenance"],
               "candidate_rows": len(train), "dev_rows": len(dev),
               "candidate_groups": len({r["group"] for r in train}), "dev_groups": len({r["group"] for r in dev}),
               "total_generation_calls": sum(r["summary"]["count"] for r in results.values()),
               "conditions": results})


if __name__ == "__main__":
    main()
