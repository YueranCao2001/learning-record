"""Run the fixed warm-start repair comparison; preserve all preparation sources."""
import argparse
import gc
from importlib.metadata import version
from pathlib import Path
import torch
from peft import PeftModel
from transformers import AutoTokenizer, set_seed
from boundary_experiment import adapter_hash, reload_arm
from prompt_diagnostic import ROOT, read
from repair_data import compose, evaluation_rows, validate
from review_repair import review
from train_smoke import BASE, REVISION, SNAPSHOT, digest, fixture, generate, load_base, base_hash, save_json


def train_arm(snapshot, tokenizer, evaluation, fixtures, protocol, arm, output, artifact):
    output.mkdir(parents=True, exist_ok=False)
    artifact.mkdir(parents=True, exist_ok=False)
    set_seed(protocol["seed"])
    model = PeftModel.from_pretrained(load_base(), snapshot, local_files_only=True, is_trainable=True)
    config = model.peft_config["default"]
    for name in ("r", "lora_alpha", "lora_dropout", "bias"):
        assert getattr(config, name) == protocol["lora"][name]
    assert set(config.target_modules) == set(protocol["lora"]["target_modules"])
    params = {n: p for n, p in model.named_parameters() if p.requires_grad}
    assert params and all("lora_" in n for n in params)
    initial = {n: p.detach().cpu().clone() for n, p in params.items()}
    initial_hash, frozen_before = adapter_hash(initial), base_hash(model)
    settings = protocol["optimizer"]
    optimizer = torch.optim.AdamW(params.values(), lr=settings["lr"], weight_decay=settings["weight_decay"],
                                  foreach=settings["foreach"], fused=settings["fused"])
    initial_optimizer_entries = len(optimizer.state)
    assert initial_optimizer_entries == 0
    history = []
    model.train()
    model.config.use_cache = False
    torch.cuda.reset_peak_memory_stats()
    for step, identity in enumerate(protocol[arm + "_order"], 1):
        row = fixtures[identity]
        ids = torch.tensor([row["input_ids"]], device="cuda")
        labels = torch.tensor([row["labels"]], device="cuda")
        optimizer.zero_grad(set_to_none=True)
        loss = model(input_ids=ids, attention_mask=torch.ones_like(ids), labels=labels).loss
        assert torch.isfinite(loss)
        loss.backward()
        gradients = [p.grad for p in params.values() if p.grad is not None]
        assert gradients and all(torch.isfinite(g).all() for g in gradients)
        assert any(g.abs().max() > 0 for g in gradients)
        assert all(p.grad is None for n, p in model.named_parameters() if "lora_" not in n)
        norm = torch.nn.utils.clip_grad_norm_(params.values(), protocol["clip_grad_norm"])
        assert torch.isfinite(norm)
        optimizer.step()
        history.append({"step": step, "id": identity, "loss": loss.item(), "grad_norm_before_clip": norm.item(),
                        "lr": optimizer.param_groups[0]["lr"], "target_tokens": row["target_tokens"]})
        save_json(output / "training.json", history)
        print(arm, history[-1], flush=True)
    optimizer.zero_grad(set_to_none=True)
    saved = {n: p.detach().cpu().clone() for n, p in params.items()}
    changed = sum(not torch.equal(initial[n], saved[n]) for n in saved)
    frozen_after = base_hash(model)
    assert frozen_before == frozen_after and changed > 0
    final_steps = sorted({int(v["step"].item()) for v in optimizer.state.values()})
    assert final_steps == [protocol["planned_steps_per_arm"]]
    peak = torch.cuda.max_memory_allocated()
    model.config.use_cache = True
    print(arm, "generating 57 post-training outputs", flush=True)
    actual = generate(model, tokenizer, evaluation)
    save_json(output / "after.json", actual)
    model.peft_config["default"].base_model_name_or_path = BASE
    model.peft_config["default"].revision = REVISION
    model.save_pretrained(artifact, safe_serialization=True, save_embedding_layers=False)
    tokenizer.save_pretrained(artifact)
    summary = {"steps": len(history), "optimizer_state_entries_before": initial_optimizer_entries,
               "optimizer_final_steps": final_steps, "initial_adapter_sha256": initial_hash,
               "trained_adapter_sha256": adapter_hash(saved), "changed_adapter_tensors": changed,
               "trainable_parameters": sum(p.numel() for p in params.values()),
               "frozen_base_sha256_before": frozen_before, "frozen_base_sha256_after": frozen_after, "base_unchanged": True,
               "trained_target_tokens": sum(r["target_tokens"] for r in history), "peak_training_allocated_bytes": peak,
               "dev_count": len(actual), "after_exact": sum(o["score"]["exact"] for o in actual),
               "adapter_reload_exact": False, "reload_generation_tokens_equal": False,
               "exact_training_resume_tested": False, "adapter_file_sha256": digest(artifact / "adapter_model.safetensors")}
    save_json(output / "summary.json", summary)
    del model, params, optimizer, loss, gradients
    gc.collect()
    torch.cuda.empty_cache()
    return saved, summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    args = parser.parse_args()
    if Path(args.run).name != args.run or args.run in (".", ".."):
        raise ValueError("new run directory required")
    execution = read(ROOT / "data/repair-execution-v1.json")
    prepared = ROOT / "results" / execution["source_preparation"]
    data, protocol = compose()
    assert data == read(prepared / "data.json") and protocol == read(prepared / "protocol.json")
    audit = validate(data, protocol)
    prior_manifest = read(prepared / "manifest.json")
    assert all(digest(ROOT / n) == sha for n, sha in prior_manifest["source_sha256"].items())
    assert protocol["planned_steps_per_arm"] == execution["steps_per_arm"] == 12
    assert protocol["generation"] == {"do_sample": False, "max_new_tokens": 128}
    assert protocol["scheduler"] == "constant" and protocol["batch_size"] == protocol["gradient_accumulation_steps"] == 1
    assert not protocol["packing"] and protocol["base_revision"] == REVISION
    upload = read(ROOT / "results/2026-10-23-boundary-v2/candidate/upload.json")
    snapshot = Path(upload["downloaded_snapshot"])
    assert upload["commit"] == protocol["initial_adapter_revision"]
    assert all(digest(snapshot / n) == sha for n, sha in upload["files_sha256"].items())
    output = ROOT / "results" / args.run
    output.mkdir(parents=True, exist_ok=False)
    for name, value in (("data", data), ("source-protocol", protocol), ("execution-protocol", execution), ("split-audit", audit)):
        save_json(output / (name + ".json"), value)
    sources = ("repair_experiment.py", "review_repair.py", "boundary_experiment.py", "train_smoke.py", "contract.py",
               "repair_data.py", "confirm_prompt.py", "prompt_diagnostic.py", "data/repair-execution-v1.json",
               "data/repair-protocol-v1.json", "data/repair-data-v1.json")
    save_json(output / "manifest.json", {"source_sha256": {n: digest(ROOT / n) for n in sources},
              "data_sha256": digest(output / "data.json"), "source_preparation": execution["source_preparation"],
              "preparation_manifest_sha256": digest(prepared / "manifest.json"), "base_revision": REVISION,
              "initial_adapter_revision": upload["commit"], "published_reference_adapter_sha256": upload["files_sha256"]["adapter_model.safetensors"],
              "human_review": "pending", "planned_updates": 24, "planned_generation_calls": 285})
    assert torch.cuda.is_available()
    save_json(output / "environment.json", {"packages": {n: version(n) for n in ("torch", "transformers", "peft", "huggingface_hub")},
              "gpu": torch.cuda.get_device_name(0), "cuda": torch.version.cuda, "new_dependencies_installed": False})
    tokenizer = AutoTokenizer.from_pretrained(SNAPSHOT, local_files_only=True)
    fixtures = {r["id"]: fixture(tokenizer, data, r) for r in data["rows"] if r["split"] == "train"}
    assert list(fixtures.values()) == read(prepared / "train-token-fixtures.json")
    evaluation = {"instruction": data["instruction"], "rows": evaluation_rows(data)}
    set_seed(protocol["seed"])
    baseline_model = PeftModel.from_pretrained(load_base(), snapshot, local_files_only=True, is_trainable=False)
    print("Reproducing 57 frozen baseline outputs", flush=True)
    baseline = generate(baseline_model, tokenizer, evaluation)
    save_json(output / "baseline.json", baseline)
    assert [o["token_ids"] for o in baseline] == [o["token_ids"] for o in read(prepared / "baseline.json")]
    save_json(output / "baseline-replay.json", {"count": len(baseline), "all_tokens_equal_preparation": True})
    del baseline_model
    gc.collect()
    torch.cuda.empty_cache()
    for arm in execution["arms"]:
        artifact = ROOT / "artifacts" / args.run / arm
        saved, summary = train_arm(snapshot, tokenizer, evaluation, fixtures, protocol, arm, output / arm, artifact)
        print(arm, "reloading adapter and generating 57 outputs", flush=True)
        reload_arm(tokenizer, evaluation, output / arm, artifact, saved, summary)
        del saved
        gc.collect()
        torch.cuda.empty_cache()
    result = review(output)
    assert result["training_steps"] == execution["executed_updates_expected"]
    assert result["total_generation_calls"] == execution["generation_budget"]["total"]
    assert result["same_initial_adapter"] and result["same_initial_base"] and result["both_reload_exact"]
    save_json(output / "review.json", result)
    print({"steps": result["training_steps"], "generation_calls": result["total_generation_calls"],
           "first_six_equal": result["first_six_training_records_equal"],
           "scores": {n: {c: s["exact"] for c, s in v.items()} for n, v in result["summaries"].items()}}, flush=True)


if __name__ == "__main__":
    main()
