"""Read-only fitting check on the frozen six repair training examples."""
import argparse
import gc
import inspect
from importlib.metadata import version
from pathlib import Path
import torch
from peft import PeftModel
from transformers import AutoTokenizer, set_seed
from transformers.loss.loss_utils import ForCausalLMLoss
from audit_evaluation import state_hash
from fit_metrics import first_difference, review, target_statistics
from prompt_diagnostic import ROOT, read
from train_smoke import REVISION, SNAPSHOT, digest, fixture, generate, load_base, save_json


def teacher_forced(model, tokenizer, f, output):
    ids = torch.tensor([f["input_ids"]], device="cuda")
    labels = torch.tensor([f["labels"]], device="cuda")
    with torch.inference_mode():
        result = model(input_ids=ids, attention_mask=torch.ones_like(ids), labels=labels, use_cache=False)
        stats = target_statistics(result.logits[0], labels[0])
        tokens = [{"input_position": position, "target_id": target, "target_text": tokenizer.decode([target]),
                   "target_logp": logp, "argmax_id": top, "argmax_text": tokenizer.decode([top]), "argmax_logp": top_logp}
                  for position, target, logp, top, top_logp in zip(stats["positions"], stats["target_ids"],
                    stats["target_logps"], stats["argmax_ids"], stats["argmax_logps"])]
        mean_nll = -sum(stats["target_logps"]) / len(tokens)
        assert abs(mean_nll - result.loss.item()) <= 0.00001
        mismatch = first_difference(stats["target_ids"], output["token_ids"])
        free_logp = None
        if mismatch is not None and mismatch < len(tokens) and mismatch < len(output["token_ids"]):
            logits = result.logits[0, tokens[mismatch]["input_position"] - 1].float()
            free_logp = torch.log_softmax(logits, dim=-1)[output["token_ids"][mismatch]].item()
    return {"id": f["id"], "model_loss": result.loss.item(), "first_difference_index": mismatch,
            "free_token_logp_at_first_difference_under_gold_prefix": free_logp, "tokens": tokens}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    args = parser.parse_args()
    if Path(args.run).name != args.run or args.run in (".", ".."):
        raise ValueError("new run directory required")
    protocol = read(ROOT / "data/training-fit-protocol-v1.json")
    source = ROOT / "results" / protocol["source_run"]
    prior_data = read(source / "data.json")
    rows = [r for r in prior_data["rows"] if r["id"] in protocol["case_ids"]]
    assert [r["id"] for r in rows] == protocol["case_ids"] and all(r["split"] == "train" for r in rows)
    prior_manifest = read(source / "manifest.json")
    assert all(digest(ROOT / n) == sha for n, sha in prior_manifest["source_sha256"].items())
    upload = read(ROOT / "results/2026-10-23-boundary-v2/candidate/upload.json")
    baseline = Path(upload["downloaded_snapshot"])
    assert all(digest(baseline / n) == sha for n, sha in upload["files_sha256"].items())
    locations = {"baseline": baseline}
    hashes = {"baseline": upload["files_sha256"]["adapter_model.safetensors"]}
    for arm in ("control", "repair"):
        locations[arm] = ROOT / "artifacts" / protocol["source_run"] / arm
        hashes[arm] = read(source / arm / "summary.json")["adapter_file_sha256"]
        assert digest(locations[arm] / "adapter_model.safetensors") == hashes[arm]
    folder = ROOT / "results" / args.run
    folder.mkdir(parents=True, exist_ok=False)
    save_json(folder / "protocol.json", protocol)
    save_json(folder / "cases.json", rows)
    source_names = ("evaluate_fit.py", "fit_metrics.py", "train_smoke.py", "contract.py", "audit_evaluation.py",
                    "review_repair.py", "confirm_prompt.py", "prompt_diagnostic.py", "data/training-fit-protocol-v1.json",
                    f"results/{protocol['source_run']}/data.json", f"results/{protocol['source_run']}/control/summary.json",
                    f"results/{protocol['source_run']}/repair/summary.json")
    save_json(folder / "manifest.json", {"source_sha256": {n: digest(ROOT / n) for n in source_names},
              "base_revision": REVISION, "baseline_adapter_revision": upload["commit"], "adapter_files_sha256": hashes,
              "published_reference_adapter_sha256": hashes["baseline"], "training_steps": 0,
              "diagnostic_training_view_only": True, "human_review": "pending"})
    (folder / "causal-loss-source.txt").write_text(inspect.getsource(ForCausalLMLoss), encoding="utf-8")
    assert torch.cuda.is_available()
    set_seed(42)
    save_json(folder / "environment.json", {"packages": {n: version(n) for n in ("torch", "transformers", "peft", "huggingface_hub")},
              "gpu": torch.cuda.get_device_name(0), "cuda": torch.version.cuda, "new_dependencies_installed": False})
    tokenizer = AutoTokenizer.from_pretrained(SNAPSHOT, local_files_only=True)
    data = {"instruction": prior_data["instruction"], "rows": rows}
    fixtures = [fixture(tokenizer, data, r) for r in rows]
    prior_fixtures = {r["id"]: r for r in read(ROOT / "results/2026-10-29-repair-preparation-v1/train-token-fixtures.json")}
    assert all(f == prior_fixtures[f["id"]] for f in fixtures)
    assert sum(f["target_tokens"] for f in fixtures) == protocol["expected_target_tokens_per_condition"]
    save_json(folder / "token-fixtures.json", fixtures)
    for condition in protocol["conditions"]:
        model = PeftModel.from_pretrained(load_base(), locations[condition], local_files_only=True, is_trainable=False)
        model.eval()
        assert not any(p.requires_grad for p in model.parameters())
        before = state_hash(model)
        print(condition, "six free generations and six teacher-forced forwards", flush=True)
        # The frozen generate helper selects split=dev; this temporary inference view does not mutate data roles.
        outputs = generate(model, tokenizer, {"instruction": data["instruction"], "rows": [dict(r, split="dev") for r in rows]})
        save_json(folder / f"{condition}-generation.json", outputs)
        teachers = [teacher_forced(model, tokenizer, f, o) for f, o in zip(fixtures, outputs)]
        save_json(folder / f"{condition}-teacher.json", teachers)
        after = state_hash(model)
        assert before == after
        save_json(folder / f"{condition}-weights.json", {"before": before, "after": after, "unchanged": True, "training_steps": 0})
        del model
        gc.collect(); torch.cuda.empty_cache()
    result = review(folder)
    assert result["total_generation_calls"] == protocol["expected_generation_calls"]
    assert result["teacher_forced_forwards"] == protocol["expected_teacher_forced_forwards"]
    save_json(folder / "review.json", result)
    print({c: {k: s[k] for k in ("exact", "token_weighted_mean_nll", "teacher_argmax_correct_tokens", "supervised_tokens")}
           for c, s in result["summaries"].items()}, flush=True)


if __name__ == "__main__":
    main()
