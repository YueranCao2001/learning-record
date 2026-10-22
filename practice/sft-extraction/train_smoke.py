"""Six-step GPU LoRA smoke test; no automatic upload or quality claim."""
import argparse
import gc
import hashlib
import json
from importlib.metadata import version
from pathlib import Path

import torch
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

from contract import load_pilot, score

ROOT = Path(__file__).resolve().parent
BASE = "Qwen/Qwen3-0.6B"
REVISION = "c1899de289a04d12100db370d81485cdf75e47ca"
CACHE = Path("E:/Evren/Evren_work/.hf-cache")
SNAPSHOT = CACHE / "models--Qwen--Qwen3-0.6B/snapshots" / REVISION


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def messages(data, row):
    return [{"role": "system", "content": data["instruction"]},
            {"role": "user", "content": row["text"]}]


def prompt_ids(tokenizer, data, row):
    return tokenizer.apply_chat_template(messages(data, row), tokenize=True,
                                         add_generation_prompt=True, enable_thinking=False)


def fixture(tokenizer, data, row):
    prefix = prompt_ids(tokenizer, data, row)
    target = json.dumps(row["expected"], ensure_ascii=False, separators=(",", ":"))
    full = tokenizer.apply_chat_template(messages(data, row) + [
        {"role": "assistant", "content": target}], tokenize=True,
        add_generation_prompt=False, enable_thinking=False)
    # A template change must fail rather than silently shifting the supervision.
    assert full[:len(prefix)] == prefix, "generation prefix differs from training prefix"
    # Ignore optional trailing newline after end-of-turn; supervise JSON + EOS only.
    eos = full.index(tokenizer.eos_token_id, len(prefix))
    full = full[:eos + 1]
    assert len(full) <= 512, "no silent truncation"
    labels = [-100] * len(prefix) + full[len(prefix):]
    assert len(full) == len(labels) and len(full) > len(prefix)
    assert tokenizer.decode(full[len(prefix):], skip_special_tokens=True).strip() == target
    return {"id": row["id"], "input_ids": full, "labels": labels,
            "prefix_tokens": len(prefix), "target_tokens": len(full) - len(prefix),
            "tokens": [{"position": i, "id": token,
                        "text": tokenizer.decode([token]), "label": labels[i]}
                       for i, token in enumerate(full)]}


def base_hash(model):
    result = hashlib.sha256()
    for name, parameter in model.named_parameters():
        if "lora_" not in name:
            result.update(name.encode())
            result.update(parameter.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    return result.hexdigest()


def load_base():
    return AutoModelForCausalLM.from_pretrained(
        SNAPSHOT, local_files_only=True, torch_dtype=torch.bfloat16,
        attn_implementation="sdpa").to("cuda")


def generate(model, tokenizer, data):
    model.eval()
    result = []
    for row in data["rows"]:
        if row["split"] != "dev":
            continue
        ids = torch.tensor([prompt_ids(tokenizer, data, row)], device="cuda")
        with torch.inference_mode():
            output = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids),
                                    do_sample=False, max_new_tokens=128,
                                    pad_token_id=tokenizer.pad_token_id,
                                    eos_token_id=tokenizer.eos_token_id,
                                    use_cache=True)
        new = output[0, ids.shape[1]:].cpu().tolist()
        raw = tokenizer.decode(new, skip_special_tokens=True)
        result.append({"id": row["id"], "raw": raw, "token_ids": new,
                       "ended_with_eos": new[-1] == tokenizer.eos_token_id,
                       "score": score(raw, row["expected"])})
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True, help="Unique evidence directory name")
    args = parser.parse_args()
    assert Path(args.run).name == args.run
    output = ROOT / "results" / args.run
    output.mkdir(parents=True, exist_ok=False)
    artifact = ROOT / "artifacts" / args.run
    artifact.mkdir(parents=True, exist_ok=False)
    set_seed(42)
    assert torch.cuda.is_available()
    x = torch.randn(32, 32, device="cuda", requires_grad=True)
    x.square().mean().backward()
    assert torch.isfinite(x.grad).all()
    environment = {"packages": {n: version(n) for n in (
        "torch", "transformers", "peft", "accelerate", "huggingface_hub")},
        "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0),
        "capability": list(torch.cuda.get_device_capability(0)),
        "gpu_backward_finite": True}
    save_json(output / "environment.json", environment)
    data = load_pilot()
    tokenizer = AutoTokenizer.from_pretrained(SNAPSHOT, local_files_only=True)
    fixtures = [fixture(tokenizer, data, row) for row in data["rows"]]
    save_json(output / "token-fixtures.json", fixtures)
    manifest = {"base": BASE, "base_revision": REVISION, "seed": 42,
                "data_sha256": digest(ROOT / "data/pilot-v1.json"),
                "script_sha256": digest(__file__),
                "template_sha256": hashlib.sha256(tokenizer.chat_template.encode()).hexdigest(),
                "lora": {"r": 8, "lora_alpha": 16, "lora_dropout": 0.0,
                         "target_modules": ["q_proj", "v_proj"], "bias": "none"},
                "steps": 6, "batch_size": 1, "lr": 0.0002, "weight_decay": 0.0,
                "optimizer": "AdamW", "dtype": "bfloat16", "packing": False,
                "max_length": 512, "enable_thinking": False,
                "generation": {"do_sample": False, "max_new_tokens": 128},
                "scope": "synthetic diagnostic data; real GPU training, not generalization evaluation"}
    save_json(output / "manifest.json", manifest)
    model = load_base()
    print("Loaded base; generating frozen pre-training dev outputs", flush=True)
    before = generate(model, tokenizer, data)
    save_json(output / "baseline.json", before)  # Persisted before any optimizer update.
    model = get_peft_model(model, LoraConfig(task_type="CAUSAL_LM", **manifest["lora"]))
    trainables = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    assert trainables and all("lora_" in n for n, _ in trainables)
    initial = {n: p.detach().cpu().clone() for n, p in trainables}
    frozen_before = base_hash(model)
    optimizer = torch.optim.AdamW([p for _, p in trainables], lr=manifest["lr"], weight_decay=0)
    rows = [f for f, row in zip(fixtures, data["rows"]) if row["split"] == "train"]
    history = []
    model.train()
    model.config.use_cache = False
    torch.cuda.reset_peak_memory_stats()
    for step in range(manifest["steps"]):
        row = rows[step % len(rows)]
        ids = torch.tensor([row["input_ids"]], device="cuda")
        labels = torch.tensor([row["labels"]], device="cuda")
        optimizer.zero_grad(set_to_none=True)
        loss = model(input_ids=ids, attention_mask=torch.ones_like(ids), labels=labels).loss
        assert torch.isfinite(loss)
        loss.backward()
        assert all(torch.isfinite(p.grad).all() for _, p in trainables if p.grad is not None)
        assert any(p.grad is not None and p.grad.abs().max() > 0 for _, p in trainables)
        assert all(p.grad is None for n, p in model.named_parameters() if "lora_" not in n)
        norm = torch.nn.utils.clip_grad_norm_([p for _, p in trainables], 1.0)
        assert torch.isfinite(norm)
        optimizer.step()
        record = {"step": step + 1, "id": row["id"], "loss": loss.item(),
                  "grad_norm_before_clip": norm.item()}
        history.append(record)
        save_json(output / "training.json", history)
        print(record, flush=True)
    changed = [n for n, p in trainables if not torch.equal(initial[n], p.detach().cpu())]
    frozen_after = base_hash(model)
    assert changed and frozen_before == frozen_after
    training_peak = torch.cuda.max_memory_allocated()
    model.config.use_cache = True
    after = generate(model, tokenizer, data)
    save_json(output / "after.json", after)
    model.peft_config["default"].base_model_name_or_path = BASE
    model.peft_config["default"].revision = REVISION
    model.save_pretrained(artifact, safe_serialization=True)
    tokenizer.save_pretrained(artifact)
    adapter_state = {n: p.detach().cpu().clone() for n, p in trainables}
    del optimizer, trainables, model, loss
    gc.collect()
    torch.cuda.empty_cache()
    reloaded = PeftModel.from_pretrained(load_base(), artifact, is_trainable=False)
    reload_state = dict(reloaded.named_parameters())
    assert all(torch.equal(value, reload_state[name].detach().cpu()) for name, value in adapter_state.items())
    loaded_outputs = generate(reloaded, tokenizer, data)
    save_json(output / "reloaded.json", loaded_outputs)
    assert [r["token_ids"] for r in loaded_outputs] == [r["token_ids"] for r in after]
    summary = {"steps": len(history), "trainable_parameters": sum(v.numel() for v in adapter_state.values()),
               "changed_adapter_tensors": len(changed), "frozen_base_sha256_before": frozen_before,
               "frozen_base_sha256_after": frozen_after, "base_unchanged": True,
               "adapter_reload_exact": True, "reload_generation_tokens_equal": True,
               "peak_training_allocated_bytes": training_peak,
               "baseline_exact": sum(r["score"]["exact"] for r in before),
               "after_exact": sum(r["score"]["exact"] for r in after), "dev_count": len(after),
               "exact_training_resume_tested": False, "uploaded": False}
    save_json(output / "summary.json", summary)
    print(summary, flush=True)


if __name__ == "__main__":
    main()
