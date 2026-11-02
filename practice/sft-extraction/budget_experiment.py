"""Reuse frozen warm-start worker to test 12 versus 36 mixed-sequence updates."""
import argparse
import copy
import gc
from importlib.metadata import version
from pathlib import Path
import torch
from peft import PeftModel
from transformers import AutoTokenizer, set_seed
from audit_evaluation import state_hash
from boundary_experiment import reload_arm
from evaluate_fit import teacher_forced
from prompt_diagnostic import ROOT, read
from repair_data import compose, validate
from repair_experiment import train_arm
from review_budget import review, rows_for_evaluation
from train_smoke import SNAPSHOT, REVISION, digest, fixture, generate, load_base, save_json


def extended_configuration(source, protocol):
    keys = ("seed", "lora", "optimizer", "clip_grad_norm", "batch_size", "gradient_accumulation_steps",
            "scheduler", "dtype", "packing", "max_length", "enable_thinking", "generation")
    result = {k: copy.deepcopy(source[k]) for k in keys}
    result["planned_steps_per_arm"] = protocol["extended_steps"]
    result["repair_order"] = source["repair_order"] * protocol["sequence_repetitions"]
    if len(result["repair_order"]) != result["planned_steps_per_arm"]:
        raise ValueError("budget and repeated schedule disagree")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    args = parser.parse_args()
    if Path(args.run).name != args.run or args.run in (".", ".."):
        raise ValueError("new run directory required")
    protocol = read(ROOT / "data/optimization-budget-v1.json")
    source = ROOT / "results" / protocol["source_run"]
    data, configuration = compose()
    assert data == read(source / "data.json") and configuration == read(source / "source-protocol.json")
    validate(data, configuration)
    source_manifest = read(source / "manifest.json")
    assert all(digest(ROOT / n) == sha for n, sha in source_manifest["source_sha256"].items())
    long_config = extended_configuration(configuration, protocol)
    old_summary = read(source / "repair/summary.json")
    short = ROOT / "artifacts" / protocol["source_run"] / "repair"
    assert digest(short / "adapter_model.safetensors") == old_summary["adapter_file_sha256"]
    upload = read(ROOT / "results/2026-10-23-boundary-v2/candidate/upload.json")
    initial = Path(upload["downloaded_snapshot"])
    assert configuration["initial_adapter_revision"] == upload["commit"]
    assert all(digest(initial / n) == sha for n, sha in upload["files_sha256"].items())
    folder = ROOT / "results" / args.run
    folder.mkdir(parents=True, exist_ok=False)
    for name, value in (("protocol", protocol), ("data", data), ("source-protocol", configuration), ("training-configuration", long_config)):
        save_json(folder / (name + ".json"), value)
    names = ("budget_experiment.py", "review_budget.py", "repair_experiment.py", "boundary_experiment.py", "evaluate_fit.py",
             "fit_metrics.py", "train_smoke.py", "contract.py", "audit_evaluation.py", "repair_data.py", "review_repair.py",
             "confirm_prompt.py", "prompt_diagnostic.py", "data/optimization-budget-v1.json", "data/repair-data-v1.json",
             "data/repair-protocol-v1.json", f"results/{protocol['source_run']}/repair/summary.json")
    save_json(folder / "manifest.json", {"source_sha256": {n: digest(ROOT / n) for n in names},
              "data_sha256": digest(folder / "data.json"), "source_run": protocol["source_run"],
              "base_revision": REVISION, "initial_adapter_revision": upload["commit"],
              "short_adapter_file_sha256": old_summary["adapter_file_sha256"],
              "published_reference_adapter_sha256": upload["files_sha256"]["adapter_model.safetensors"],
              "planned_training_updates_today": 36, "planned_generation_calls": 189, "human_review": "pending"})
    assert torch.cuda.is_available()
    set_seed(configuration["seed"])
    save_json(folder / "environment.json", {"packages": {n: version(n) for n in ("torch", "transformers", "peft", "huggingface_hub")},
              "gpu": torch.cuda.get_device_name(0), "cuda": torch.version.cuda, "new_dependencies_installed": False})
    tokenizer = AutoTokenizer.from_pretrained(SNAPSHOT, local_files_only=True)
    fixtures = {r["id"]: fixture(tokenizer, data, r) for r in data["rows"] if r["split"] == "train"}
    save_json(folder / "token-fixtures.json", list(fixtures.values()))
    rows = rows_for_evaluation(data)
    evaluation = {"instruction": data["instruction"], "rows": [dict(r, split="dev") for r in rows]}
    fit_rows = [r for r in rows if r["cohort"] == "new-train"]
    model = PeftModel.from_pretrained(load_base(), short, local_files_only=True, is_trainable=False)
    model.eval()
    before = state_hash(model)
    print("Replaying 63 outputs for the existing 12-step reference", flush=True)
    outputs = generate(model, tokenizer, evaluation)
    save_json(folder / "short/after.json", outputs)
    expected = {o["id"]: o for o in read(source / "repair/after.json") + read(ROOT / "results/2026-11-01-training-fit-v1/repair-generation.json")}
    equal = all(o["token_ids"] == expected[o["id"]]["token_ids"] for o in outputs)
    assert equal and len(outputs) == 63
    teachers = [teacher_forced(model, tokenizer, fixtures[r["id"]], o) for r, o in zip(fit_rows, outputs[:6])]
    save_json(folder / "short/teacher.json", teachers)
    after = state_hash(model)
    assert before == after
    save_json(folder / "short/replay.json", {"count": 63, "all_tokens_equal": equal, "weights_before": before, "weights_after": after})
    del model
    gc.collect(); torch.cuda.empty_cache()
    artifact = ROOT / "artifacts" / args.run / "extended"
    print("One continuous 36-update path; frozen worker's status text may still say 57 although evaluation has 63 rows", flush=True)
    saved, summary = train_arm(initial, tokenizer, evaluation, fixtures, long_config, "repair", folder / "extended", artifact)
    comparison = {"initial_adapter_equal": summary["initial_adapter_sha256"] == old_summary["initial_adapter_sha256"],
                  "first_twelve_records_equal": read(folder / "extended/training.json")[:12] == read(source / "repair/training.json"),
                  "midrun_parameters_directly_compared": False}
    assert comparison["initial_adapter_equal"] and comparison["first_twelve_records_equal"]
    save_json(folder / "prefix-comparison.json", comparison)
    reload_arm(tokenizer, evaluation, folder / "extended", artifact, saved, summary)
    del saved
    gc.collect(); torch.cuda.empty_cache()
    model = PeftModel.from_pretrained(load_base(), artifact, local_files_only=True, is_trainable=False)
    model.eval()
    before = state_hash(model)
    outputs = read(folder / "extended/after.json")
    teachers = [teacher_forced(model, tokenizer, fixtures[r["id"]], o) for r, o in zip(fit_rows, outputs[:6])]
    save_json(folder / "extended/teacher.json", teachers)
    after = state_hash(model)
    assert before == after
    save_json(folder / "extended/teacher-weights.json", {"before": before, "after": after, "unchanged": True})
    result = review(folder)
    assert result["training_steps"] == protocol["executed_updates_today_expected"]
    assert result["total_generation_calls"] == protocol["generation_budget"]["total"]
    assert result["teacher_forced_forwards"] == protocol["teacher_forced_forwards_expected"]
    assert result["trained_target_tokens"] == old_summary["trained_target_tokens"] * protocol["sequence_repetitions"]
    save_json(folder / "review.json", result)
    print({c: {g: s["exact"] for g, s in groups.items()} for c, groups in result["summaries"].items()}, flush=True)


if __name__ == "__main__":
    main()
