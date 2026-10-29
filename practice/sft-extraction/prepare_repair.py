"""Data preparation, actual tokenizer budgets and frozen candidate baseline; no training."""
import argparse
from collections import Counter
from importlib.metadata import version
from pathlib import Path
import torch
from peft import PeftModel
from transformers import AutoTokenizer, set_seed
from audit_evaluation import state_hash
from prompt_diagnostic import ROOT, read
from repair_data import compose, evaluation_rows, rejection_checks, validate
from train_smoke import REVISION, SNAPSHOT, digest, fixture, generate, load_base, save_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    args = parser.parse_args()
    if Path(args.run).name != args.run or args.run in (".", ".."):
        raise ValueError("new directory name required")
    data, protocol = compose()
    audit = validate(data, protocol)
    negatives = rejection_checks(data, protocol)
    assert protocol["base_revision"] == REVISION and protocol["executed_optimizer_steps_today"] == 0
    assert protocol["generation"] == {"do_sample": False, "max_new_tokens": 128}
    upload = read(ROOT / "results/2026-10-23-boundary-v2/candidate/upload.json")
    snapshot = Path(upload["downloaded_snapshot"])
    assert upload["commit"] == protocol["initial_adapter_revision"]
    assert all(digest(snapshot / n) == sha for n, sha in upload["files_sha256"].items())
    folder = ROOT / "results" / args.run
    folder.mkdir(parents=True, exist_ok=False)
    for name, value in (("data", data), ("protocol", protocol), ("split-audit", audit), ("rejections", negatives)):
        save_json(folder / (name + ".json"), value)
    sources = ("repair_data.py", "prepare_repair.py", "contract.py", "train_smoke.py", "audit_evaluation.py",
               "data/repair-data-v1.json", "data/repair-protocol-v1.json", "annotation-rules-v1.md",
               "results/2026-10-23-boundary-v2/data.json", "results/2026-10-27-prompt-diagnostic-v1/cases.json",
               "results/2026-10-28-prompt-confirmation-v1/cases.json")
    save_json(folder / "manifest.json", {"source_sha256": {n: digest(ROOT / n) for n in sources},
              "data_sha256": digest(folder / "data.json"), "protocol_sha256": digest(folder / "protocol.json"),
              "base_revision": REVISION, "adapter_revision": upload["commit"],
              "adapter_sha256": upload["files_sha256"]["adapter_model.safetensors"],
              "training_steps": 0, "human_review": "pending", "status": "prepared-not-trained"})
    set_seed(protocol["seed"])
    assert torch.cuda.is_available()
    tokenizer = AutoTokenizer.from_pretrained(SNAPSHOT, local_files_only=True)
    fixtures = {r["id"]: fixture(tokenizer, data, r) for r in data["rows"] if r["split"] == "train"}
    save_json(folder / "train-token-fixtures.json", list(fixtures.values()))
    budgets = {}
    for arm in ("control", "repair"):
        order = protocol[arm + "_order"]
        budgets[arm] = {"planned_steps": len(order), "planned_input_tokens": sum(len(fixtures[n]["input_ids"]) for n in order),
                        "planned_target_tokens": sum(fixtures[n]["target_tokens"] for n in order),
                        "exposure_counts": dict(Counter(order)),
                        "schedule": [{"step": i + 1, "id": n, "target_tokens": fixtures[n]["target_tokens"]} for i, n in enumerate(order)]}
    token_report = {"arms": budgets, "steps_matched": budgets["control"]["planned_steps"] == budgets["repair"]["planned_steps"],
                    "target_tokens_matched": budgets["control"]["planned_target_tokens"] == budgets["repair"]["planned_target_tokens"],
                    "training_fixture_count": len(fixtures), "max_train_length": max(len(f["input_ids"]) for f in fixtures.values()),
                    "all_assistant_masks_and_eos_verified": True, "executed_optimizer_steps": 0}
    save_json(folder / "token-budget.json", token_report)
    environment = {"packages": {n: version(n) for n in ("torch", "transformers", "peft", "huggingface_hub")},
                   "gpu": torch.cuda.get_device_name(0), "cuda": torch.version.cuda, "new_dependencies_installed": False}
    save_json(folder / "environment.json", environment)
    dev = evaluation_rows(data)
    model = PeftModel.from_pretrained(load_base(), snapshot, local_files_only=True, is_trainable=False)
    model.eval()
    before = state_hash(model)
    print("Freezing candidate baseline: 12 new dev + 45 historical regression; no optimizer", flush=True)
    outputs = generate(model, tokenizer, {"instruction": data["instruction"], "rows": dev})
    save_json(folder / "baseline.json", outputs)
    after = state_hash(model)
    assert before == after
    save_json(folder / "weights.json", {"before": before, "after": after, "unchanged": True})
    cohorts = {}
    for cohort in ("new-dev", "historical-regression"):
        selected = [(r, o) for r, o in zip(dev, outputs) if r["cohort"] == cohort]
        cohorts[cohort] = {"count": len(selected), "exact": sum(o["score"]["exact"] for _, o in selected),
                           "schema_valid": sum(o["score"]["schema_valid"] for _, o in selected),
                           "failed_ids": [r["id"] for r, o in selected if not o["score"]["exact"]]}
    prior = {}
    for name in ("results/2026-10-23-boundary-v2/candidate/after.json", "results/2026-10-27-prompt-diagnostic-v1/candidate-original.json",
                 "results/2026-10-28-prompt-confirmation-v1/candidate-original.json"):
        for output in read(ROOT / name):
            if output["id"] in prior: assert prior[output["id"]]["token_ids"] == output["token_ids"]
            prior[output["id"]] = output
    reproduced = {o["id"]: o["token_ids"] == prior[o["id"]]["token_ids"] for r, o in zip(dev, outputs) if r["cohort"] == "historical-regression"}
    report = {"training_steps": 0, "total_generation_calls": len(outputs), "human_review": "pending", "promote_to_main": False,
              "status": "prepared-not-trained", "cohorts": cohorts, "rejected_faults": len(negatives),
              "all_faults_rejected": all(r["rejected"] for r in negatives), "historical_tokens_reproduce": reproduced,
              "all_historical_tokens_reproduce": all(reproduced.values()), "weights_unchanged": True,
              "planned_steps_per_arm": protocol["planned_steps_per_arm"], "source_groups_disjoint": audit["source_groups_disjoint"],
              "template_names_disjoint_only": True, "semantic_independence_proven": False,
              "baseline_does_not_measure_repair_gain": True}
    save_json(folder / "review.json", report)
    print(report, flush=True)


if __name__ == "__main__":
    main()
