"""Fixed matched-budget data intervention; reuse the frozen v1 model/token helpers."""
import argparse
import gc
import hashlib
import json
from importlib.metadata import version
from pathlib import Path

import torch
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoTokenizer, set_seed

from boundary_data import compose, validate_protocol
from train_smoke import ROOT, BASE, REVISION, SNAPSHOT, digest, save_json, fixture, generate, load_base, base_hash


def seed_and_probe(seed):
    set_seed(seed)
    assert torch.cuda.is_available()
    x = torch.randn(32, 32, device="cuda", requires_grad=True)
    x.square().mean().backward()
    assert torch.isfinite(x.grad).all()


def adapter_hash(tensors):
    h = hashlib.sha256()
    for name, value in sorted(tensors.items()):
        h.update(name.encode())
        h.update(value.contiguous().view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def run_arm(base, tokenizer, data, fixtures, protocol, arm, output, artifact):
    output.mkdir(parents=True, exist_ok=False)
    artifact.mkdir(parents=True, exist_ok=False)
    model = get_peft_model(base, LoraConfig(task_type="CAUSAL_LM", **protocol["lora"]))
    params = {n: p for n, p in model.named_parameters() if p.requires_grad}
    assert params and all("lora_" in n for n in params)
    initial = {n: p.detach().cpu().clone() for n, p in params.items()}
    before_hash = base_hash(model)
    optimizer = torch.optim.AdamW(params.values(), lr=protocol["lr"], weight_decay=0)
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
        assert all(torch.isfinite(p.grad).all() for p in params.values() if p.grad is not None)
        assert any(p.grad is not None and p.grad.abs().max() > 0 for p in params.values())
        assert all(p.grad is None for n, p in model.named_parameters() if "lora_" not in n)
        norm = torch.nn.utils.clip_grad_norm_(params.values(), 1.0)
        assert torch.isfinite(norm)
        optimizer.step()
        history.append({"step": step, "id": identity, "loss": loss.item(),
                        "grad_norm_before_clip": norm.item(), "target_tokens": row["target_tokens"]})
        save_json(output / "training.json", history)
        print(arm, history[-1], flush=True)
    after_hash = base_hash(model)
    assert before_hash == after_hash
    saved = {n: p.detach().cpu().clone() for n, p in params.items()}
    changed = sum(not torch.equal(initial[n], saved[n]) for n in saved)
    assert changed > 0
    peak = torch.cuda.max_memory_allocated()
    model.config.use_cache = True
    outputs = generate(model, tokenizer, data)
    save_json(output / "after.json", outputs)
    model.peft_config["default"].base_model_name_or_path = BASE
    model.peft_config["default"].revision = REVISION
    model.save_pretrained(artifact, safe_serialization=True, save_embedding_layers=False)
    tokenizer.save_pretrained(artifact)
    summary = {"steps": len(history), "base_unchanged": True,
               "frozen_base_sha256_before": before_hash, "frozen_base_sha256_after": after_hash,
               "changed_adapter_tensors": changed, "initial_adapter_sha256": adapter_hash(initial),
               "trained_target_tokens": sum(r["target_tokens"] for r in history),
               "trainable_parameters": sum(p.numel() for p in params.values()),
               "peak_training_allocated_bytes": peak,
               "after_exact": sum(r["score"]["exact"] for r in outputs), "dev_count": len(outputs),
               "adapter_reload_exact": False, "reload_generation_tokens_equal": False,
               "exact_training_resume_tested": False}
    save_json(output / "summary.json", summary)
    del params, optimizer, model, loss
    return saved, summary


def reload_arm(tokenizer, data, output, artifact, saved, summary):
    gc.collect()
    torch.cuda.empty_cache()
    model = PeftModel.from_pretrained(load_base(), artifact, is_trainable=False)
    current = dict(model.named_parameters())
    assert all(torch.equal(v, current[n].detach().cpu()) for n, v in saved.items())
    actual = generate(model, tokenizer, data)
    expected = json.loads((output / "after.json").read_text(encoding="utf-8"))
    assert [r["token_ids"] for r in actual] == [r["token_ids"] for r in expected]
    save_json(output / "reloaded.json", actual)
    summary.update(adapter_reload_exact=True, reload_generation_tokens_equal=True)
    save_json(output / "summary.json", summary)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    args = parser.parse_args()
    assert Path(args.run).name == args.run
    output = ROOT / "results" / args.run
    output.mkdir(parents=True, exist_ok=False)
    protocol_path = ROOT / "data/boundary-protocol-v2.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    data = compose()
    validate_protocol(data, protocol)
    assert protocol["base"] == BASE and protocol["base_revision"] == REVISION
    assert protocol["batch_size"] == 1 and protocol["generation"] == {"do_sample": False, "max_new_tokens": 128}
    save_json(output / "data.json", data)
    save_json(output / "protocol.json", protocol)
    manifest = {"base": BASE, "base_revision": REVISION, "steps": protocol["steps_per_arm"], "protocol": protocol,
                "data_file": (output / "data.json").relative_to(ROOT).as_posix(),
                "data_sha256": digest(output / "data.json"), "protocol_sha256": digest(protocol_path),
                "code_sha256": {name: digest(ROOT / name) for name in (
                    "boundary_experiment.py", "boundary_data.py", "train_smoke.py", "contract.py")},
                "packing": False, "max_length": 512, "enable_thinking": False,
                "dtype": "bfloat16", "optimizer": "AdamW", "weight_decay": 0, "clip_grad_norm": 1,
                "scope": protocol["interpretation"]}
    save_json(output / "manifest.json", manifest)
    seed_and_probe(protocol["seed"])
    environment = {"packages": {n: version(n) for n in (
        "torch", "transformers", "peft", "accelerate", "huggingface_hub")},
        "gpu": torch.cuda.get_device_name(0), "cuda": torch.version.cuda, "gpu_backward_finite": True}
    save_json(output / "environment.json", environment)
    tokenizer = AutoTokenizer.from_pretrained(SNAPSHOT, local_files_only=True)
    fixtures = {r["id"]: fixture(tokenizer, data, r) for r in data["rows"]}
    save_json(output / "token-fixtures.json", list(fixtures.values()))
    base = load_base()
    baseline = generate(base, tokenizer, data)
    save_json(output / "baseline.json", baseline)
    summaries = {}
    for arm in ("control", "candidate"):
        if arm == "candidate":
            seed_and_probe(protocol["seed"])
            base = load_base()
        arm_output = output / arm
        artifact = ROOT / "artifacts" / args.run / arm
        saved, summary = run_arm(base, tokenizer, data, fixtures, protocol, arm, arm_output, artifact)
        del base
        gc.collect()
        torch.cuda.empty_cache()
        reload_arm(tokenizer, data, arm_output, artifact, saved, summary)
        summaries[arm] = summary
        save_json(arm_output / "baseline.json", baseline)
        for name in ("manifest.json", "environment.json"):
            save_json(arm_output / name, json.loads((output / name).read_text(encoding="utf-8")))
        save_json(arm_output / "protocol.json", protocol)
    assert summaries["control"]["initial_adapter_sha256"] == summaries["candidate"]["initial_adapter_sha256"]
    assert summaries["control"]["frozen_base_sha256_before"] == summaries["candidate"]["frozen_base_sha256_before"]
    save_json(output / "comparison.json", {"baseline_exact": sum(r["score"]["exact"] for r in baseline),
              "dev_count": len(baseline), "summaries": summaries, "same_initial_adapter": True,
              "no_promotion_to_main": True})
    print(json.dumps(json.loads((output / "comparison.json").read_text(encoding="utf-8")), ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
