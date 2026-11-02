"""Surface-boundary contrasts for the two remaining absent-location errors."""
from pathlib import Path
from contract import validate_target
from confirm_prompt import aggregate
from prompt_diagnostic import ROOT, read


def compose():
    protocol = read(ROOT / "data/absence-diagnostic-v1.json")
    source = read(ROOT / "results" / protocol["source_run"] / "data.json")
    lookup = {r["id"]: r for r in source["rows"]}
    rows, queue = [], []
    for spec in protocol["anchors"]:
        anchor, positive = lookup[spec["source_id"]], lookup[spec["positive_source_id"]]
        old_clause = spec["cue"] + spec["status"]
        if anchor["text"].count(old_clause) != 1 or anchor["text"].count(spec["remove_span"]) != 1:
            raise ValueError("anchor span mismatch")
        variants = {
            "anchor": anchor["text"],
            "cue-only": anchor["text"].replace(old_clause, protocol["canonical_cue"] + spec["status"]),
            "status-only": anchor["text"].replace(old_clause, spec["cue"] + protocol["canonical_status"]),
            "cue-and-status": anchor["text"].replace(old_clause, protocol["canonical_cue"] + protocol["canonical_status"]),
            "drop-clause": anchor["text"].replace(spec["remove_span"], ""),
            "explicit-value": anchor["text"].replace(old_clause, spec["positive_clause"])}
        if variants["explicit-value"] != positive["text"]:
            raise ValueError("positive control differs from frozen source")
        for variant in protocol["variants"]:
            expected = dict(anchor["expected"])
            if variant == "explicit-value": expected["location"] = positive["expected"]["location"]
            validate_target(expected)
            text = variants[variant]
            if any(v is not None and v not in text for v in expected.values()): raise ValueError("nonliteral target")
            rows.append({"id": spec["id"] + "-" + variant, "group": anchor["group"], "cohort": "absence-diagnostic",
                         "split": "dev", "variant": variant, "source_id": spec["source_id"], "text": text, "expected": expected,
                         "replay_source_id": spec["source_id"] if variant == "anchor" else spec["positive_source_id"] if variant == "explicit-value" else None})
        queue.append({"source_id": anchor["id"], "group": anchor["group"], "text": anchor["text"], "proposed_expected": anchor["expected"],
                      "rule_reference": "annotation-rules-v1.md: undetermined information maps to JSON null",
                      "assistant_rationale": "No actual venue is specified by the absence clause. Cue phrase is not a venue name. The same cue with a stated venue is an explicit-value control.",
                      "positive_control_id": positive["id"], "positive_expected": positive["expected"], "human_review": "pending"})
    if len(rows) != protocol["expected_cases"] or len({r["text"] for r in rows}) != len(rows):
        raise ValueError("case count or duplicate text")
    return protocol, source["instruction"], rows, queue


def prior_outputs(condition):
    name = {"baseline": "2026-10-30-repair-comparison-v1/baseline.json",
            "short": "2026-10-30-repair-comparison-v1/repair/after.json",
            "extended": "2026-11-02-optimization-budget-v1/extended/after.json"}[condition]
    return {o["id"]: o for o in read(ROOT / "results" / name)}


def review(folder):
    folder = Path(folder)
    protocol, rows = read(folder / "protocol.json"), read(folder / "cases.json")
    summaries, matrix, replay, all_records = {}, [], {}, {}
    for condition in protocol["conditions"]:
        outputs = read(folder / f"{condition}-generation.json")
        summary, records, errors = aggregate(rows, outputs)
        all_records[condition] = records
        summary["errors"] = errors
        summary["location_correct"] = sum(r["fields"]["location"] for r in records)
        summaries[condition] = summary
        old = prior_outputs(condition)
        replay[condition] = {r["id"]: o["token_ids"] == old[r["replay_source_id"]]["token_ids"]
                             for r, o in zip(rows, outputs) if r["replay_source_id"] is not None}
    for index, row in enumerate(rows):
        matrix.append({"id": row["id"], "source_id": row["source_id"], "variant": row["variant"],
                       "expected_location": row["expected"]["location"],
                       **{c: {"parsed": all_records[c][index]["parsed"], "exact": all_records[c][index]["exact"],
                              "location_correct": all_records[c][index]["fields"]["location"]} for c in protocol["conditions"]}})
    return {"training_steps": 0, "human_review": "pending", "promote_to_main": False,
            "total_generation_calls": sum(s["count"] for s in summaries.values()), "source_groups": 2,
            "new_independent_sources": 0, "known_replays": replay,
            "all_known_tokens_reproduce": all(v for c in replay.values() for v in c.values()),
            "weights_unchanged": all(read(folder / f"{c}-weights.json")["unchanged"] for c in protocol["conditions"]),
            "summaries": summaries, "matrix": matrix, "scope": protocol["provenance"]}


def main():
    import argparse
    import gc
    from importlib.metadata import version
    import torch
    from peft import PeftModel
    from transformers import AutoTokenizer, set_seed
    from audit_evaluation import state_hash
    from train_smoke import REVISION, SNAPSHOT, digest, generate, load_base, save_json
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    args = parser.parse_args()
    if Path(args.run).name != args.run or args.run in (".", ".."):
        raise ValueError("new directory required")
    protocol, instruction, rows, queue = compose()
    upload = read(ROOT / "results/2026-10-23-boundary-v2/candidate/upload.json")
    paths = {"baseline": Path(upload["downloaded_snapshot"]), "short": ROOT / "artifacts/2026-10-30-repair-comparison-v1/repair",
             "extended": ROOT / "artifacts/2026-11-02-optimization-budget-v1/extended"}
    hashes = {"baseline": upload["files_sha256"]["adapter_model.safetensors"],
              "short": read(ROOT / "results/2026-10-30-repair-comparison-v1/repair/summary.json")["adapter_file_sha256"],
              "extended": read(ROOT / "results/2026-11-02-optimization-budget-v1/extended/summary.json")["adapter_file_sha256"]}
    assert all(digest(paths[c] / "adapter_model.safetensors") == hashes[c] for c in paths)
    folder = ROOT / "results" / args.run
    folder.mkdir(parents=True, exist_ok=False)
    for name, value in (("protocol", protocol), ("cases", rows), ("label-review", queue)):
        save_json(folder / (name + ".json"), value)
    names = ("absence_diagnostic.py", "contract.py", "confirm_prompt.py", "prompt_diagnostic.py", "train_smoke.py",
             "audit_evaluation.py", "data/absence-diagnostic-v1.json", "annotation-rules-v1.md",
             "results/2026-11-02-optimization-budget-v1/data.json", "results/2026-11-02-optimization-budget-v1/extended/summary.json",
             "results/2026-10-30-repair-comparison-v1/repair/summary.json")
    save_json(folder / "manifest.json", {"source_sha256": {n: digest(ROOT / n) for n in names}, "base_revision": REVISION,
              "adapter_files_sha256": hashes, "published_reference_adapter_sha256": hashes["baseline"],
              "human_review": "pending", "training_steps": 0, "instruction": instruction})
    assert torch.cuda.is_available()
    set_seed(42)
    save_json(folder / "environment.json", {"packages": {n: version(n) for n in ("torch", "transformers", "peft", "huggingface_hub")},
              "gpu": torch.cuda.get_device_name(0), "cuda": torch.version.cuda, "new_dependencies_installed": False})
    tokenizer = AutoTokenizer.from_pretrained(SNAPSHOT, local_files_only=True)
    for condition in protocol["conditions"]:
        model = PeftModel.from_pretrained(load_base(), paths[condition], local_files_only=True, is_trainable=False)
        model.eval()
        assert not any(p.requires_grad for p in model.parameters())
        before = state_hash(model)
        print(condition, "12 fixed absence contrasts", flush=True)
        save_json(folder / f"{condition}-generation.json", generate(model, tokenizer, {"instruction": instruction, "rows": rows}))
        after = state_hash(model)
        assert before == after
        save_json(folder / f"{condition}-weights.json", {"before": before, "after": after, "unchanged": True})
        del model
        gc.collect(); torch.cuda.empty_cache()
    result = review(folder)
    assert result["total_generation_calls"] == protocol["expected_generation_calls"] and result["all_known_tokens_reproduce"]
    save_json(folder / "review.json", result)
    print({c: {k: s[k] for k in ("exact", "location_correct", "count")} for c, s in result["summaries"].items()}, flush=True)


if __name__ == "__main__":
    main()
