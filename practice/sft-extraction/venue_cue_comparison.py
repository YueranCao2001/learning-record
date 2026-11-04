"""Compare training with generic versus task-specific venue cues at fixed budget."""
import argparse
from importlib.metadata import version
from pathlib import Path

import torch
from transformers import AutoTokenizer

from boundary_experiment import reload_arm
from confirm_prompt import aggregate
from repair_experiment import train_arm
from review_repair import paired
from train_smoke import REVISION, ROOT, SNAPSHOT, digest, fixture, save_json
from venue_pairs import PROTOCOL, read, summarize

SPEC = "data/venue-cue-comparison-v1.json"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    args = parser.parse_args()
    if Path(args.run).name != args.run or args.run in (".", ".."):
        raise ValueError("new run directory required")
    spec = read(ROOT / SPEC)
    previous = ROOT / "results" / spec["source"]
    warm = ROOT / "results" / spec["warm_start"]
    snapshot = ROOT / "artifacts" / spec["warm_start"] / "extended"
    start = read(warm / "extended/summary.json")
    assert digest(snapshot / "adapter_model.safetensors") == start["adapter_file_sha256"]
    frozen = read(previous / "cases.json")
    train = [r for r in frozen["rows"] if r["split"] == "train"]
    dev = [r for r in frozen["rows"] if r["split"] == "dev"]
    history = [r for r in read(warm / "data.json")["rows"] if r["cohort"] == "historical-regression"]
    assert len(train) == 8 and len(dev) == 12 and len(history) == 45
    assert {r["group"] for r in train} == set(spec["assistant_label_review"])
    assert not {r["group"] for r in train} & {r["group"] for r in dev + history}
    assert len({r["id"] for r in train + dev + history}) == 65
    assert frozen["instruction"] == read(warm / "data.json")["instruction"]

    canonical = []
    for row in train:
        suffix = row["status"] if row["variant"] == "missing" else "为" + row["expected"]["location"]
        clause = row["cue"] + suffix
        assert row["text"].count(clause) == 1
        canonical.append({**row, "cue": spec["canonical_cue"],
                          "text": row["text"].replace(clause, spec["canonical_cue"] + suffix, 1)})
    assert not {r["text"] for r in train + canonical} & {r["text"] for r in dev + history}
    evaluation = {"instruction": frozen["instruction"], "rows": dev + history}
    assert all(r["split"] == "dev" for r in evaluation["rows"])
    tokenizer = AutoTokenizer.from_pretrained(SNAPSHOT, local_files_only=True)
    arm_rows = {"canonical": canonical, "target": train}
    fixtures = {arm: {r["id"]: fixture(tokenizer, frozen, r) for r in rows} for arm, rows in arm_rows.items()}
    order = [r["id"] for r in train] * spec["repetitions"]
    assert len(order) == spec["steps_per_arm"]
    for identity in order:
        a, b = (fixtures[arm][identity] for arm in spec["arms"])
        assert [t for t in a["labels"] if t != -100] == [t for t in b["labels"] if t != -100]
    budget = {arm: {"sequence_tokens": sum(len(fixtures[arm][i]["input_ids"]) for i in order),
                    "target_tokens": sum(fixtures[arm][i]["target_tokens"] for i in order)} for arm in spec["arms"]}
    configuration = read(warm / "training-configuration.json")
    configuration.pop("repair_order")
    configuration.update(planned_steps_per_arm=len(order), canonical_order=order, target_order=order)

    output = ROOT / "results" / args.run
    output.mkdir(parents=True, exist_ok=False)
    artifact = ROOT / "artifacts" / args.run
    published = read(previous / "manifest.json")["published_reference_adapter_sha256"]
    sources = [SPEC, PROTOCOL, "annotation-rules-v1.md", "venue_cue_comparison.py", "venue_pairs.py",
               "repair_experiment.py", "boundary_experiment.py", "train_smoke.py", "confirm_prompt.py", "review_repair.py",
               "contract.py", "fit_metrics.py", "prompt_diagnostic.py"]
    sources += [f"results/{spec['source']}/{n}" for n in ("cases.json", "manifest.json", "extended-generation.json")]
    sources += [f"results/{spec['warm_start']}/{n}" for n in
                ("data.json", "training-configuration.json", "extended/after.json", "extended/summary.json")]
    manifest = {"base_revision": REVISION, "source_sha256": {n: digest(ROOT / n) for n in sources},
                "initial_adapter_file_sha256": start["adapter_file_sha256"],
                "published_reference_adapter_sha256": published}
    environment = {"gpu": torch.cuda.get_device_name(0), "cuda": torch.version.cuda,
                   "packages": {n: version(n) for n in ("torch", "transformers", "peft", "huggingface_hub")}}
    for name, value in (("protocol", spec), ("configuration", configuration), ("training-rows", arm_rows),
                        ("token-budget", budget), ("manifest", manifest), ("environment", environment)):
        save_json(output / f"{name}.json", value)

    reference_history = {o["id"]: o for o in read(warm / "extended/after.json")}
    reference = read(previous / "extended-generation.json") + [reference_history[r["id"]] for r in history]
    outputs = {"warm_start": reference}
    checks = {}
    print("Frozen worker's generation banner says 57; each phase uses 12 paired dev + 45 historical rows.", flush=True)
    for arm in spec["arms"]:
        saved, summary = train_arm(snapshot, tokenizer, evaluation, fixtures[arm], configuration, arm,
                                   output / arm, artifact / arm)
        assert summary["initial_adapter_sha256"] == start["trained_adapter_sha256"]
        assert summary["frozen_base_sha256_before"] == start["frozen_base_sha256_after"]
        assert summary["trained_target_tokens"] == budget[arm]["target_tokens"]
        reload_arm(tokenizer, evaluation, output / arm, artifact / arm, saved, summary)
        checks[arm] = summary
        outputs[arm] = read(output / arm / "after.json")

    results = {}
    changes = {}
    for name, generated in outputs.items():
        pair = summarize(dev, generated[:len(dev)])
        summary, _, errors = aggregate(history, generated[len(dev):])
        results[name] = {"paired_dev": pair, "historical": {"summary": summary, "errors": errors}}
        if name != "warm_start":
            changes[name] = {"paired_dev": paired(dev, reference[:len(dev)], generated[:len(dev)]),
                             "historical": paired(history, reference[len(dev):], generated[len(dev):])}
    changes["target_vs_canonical"] = {
        "paired_dev": paired(dev, outputs["canonical"][:len(dev)], outputs["target"][:len(dev)]),
        "historical": paired(history, outputs["canonical"][len(dev):], outputs["target"][len(dev):])}
    review = {"training_steps": sum(s["steps"] for s in checks.values()), "human_review": "pending",
              "promote_to_main": False, "trained_weights_uploaded": False, "reference_replayed": len(reference),
              "new_generation_calls": sum(s["dev_count"] * 2 for s in checks.values()),
              "identical_supervised_sequences": True, "checks": checks, "results": results, "changes": changes}
    save_json(output / "review.json", review)


if __name__ == "__main__":
    main()
