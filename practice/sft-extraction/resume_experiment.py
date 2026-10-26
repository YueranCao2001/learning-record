"""Four fresh processes: continuous, prefix, full resume and optimizer-reset control."""
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import argparse
import json
from pathlib import Path
import random
import subprocess
import sys
from importlib.metadata import version
import numpy as np
import torch
from torch.nn.attention import sdpa_kernel, SDPBackend
from peft import LoraConfig, get_peft_model, get_peft_model_state_dict, set_peft_model_state_dict
from transformers import AutoTokenizer, set_seed
from contract import load_pilot
from resume_state import cpu_tree, tree_digest, capture_rng, restore_rng
from train_smoke import ROOT, REVISION, SNAPSHOT, digest, save_json, fixture, load_base, base_hash


def phase(run, name):
    output = ROOT / "results" / run
    artifact = ROOT / "artifacts" / run
    folder = output / name
    folder.mkdir(exist_ok=False)
    protocol = json.loads((output / "protocol.json").read_text(encoding="utf-8"))
    assert protocol["base_revision"] == REVISION and protocol["total_steps"] == 4 and protocol["checkpoint_step"] == 2
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"
    set_seed(protocol["seed"])
    assert torch.cuda.is_available()
    data = load_pilot()
    tokenizer = AutoTokenizer.from_pretrained(SNAPSHOT, local_files_only=True)
    rows = [fixture(tokenizer, data, r) for r in data["rows"] if r["split"] == "train"]
    model = get_peft_model(load_base(), LoraConfig(task_type="CAUSAL_LM", **protocol["lora"]))
    params = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    assert params and all("lora_" in n for n, _ in params)
    optimizer = torch.optim.AdamW([p for _, p in params], lr=protocol["lr"], weight_decay=0, foreach=False, fused=False)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.5)
    history, start = [], 0
    initial_adapter_hash = tree_digest(cpu_tree(get_peft_model_state_dict(model)))
    if name in ("resume", "optimizer-reset"):
        checkpoint = artifact / "checkpoint-step-2.pt"
        prefix_report = json.loads((output / "prefix/state.json").read_text(encoding="utf-8"))
        assert digest(checkpoint) == prefix_report["checkpoint_sha256"]
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        assert state["parameter_names"] == [n for n, _ in params]
        assert state["protocol_sha256"] == digest(output / "protocol.json")
        set_peft_model_state_dict(model, state["adapter"])
        assert tree_digest(cpu_tree(get_peft_model_state_dict(model))) == tree_digest(state["adapter"])
        if name == "resume":
            optimizer.load_state_dict(state["optimizer"])
        else:
            for group, original in zip(optimizer.param_groups, state["optimizer"]["param_groups"]):
                group["lr"] = original["lr"]
                group["initial_lr"] = original["initial_lr"]
            assert not optimizer.state
        scheduler.load_state_dict(state["scheduler"])
        history, start = state["history"], state["completed_steps"]
        assert start == protocol["checkpoint_step"]
        # Restore after model/optimizer construction, which may consume random values.
        restore_rng(state["rng"])
        assert tree_digest(capture_rng()) == tree_digest(state["rng"])
    frozen_before = base_hash(model)
    model.train()
    model.config.use_cache = False
    stop = protocol["checkpoint_step"] if name == "prefix" else protocol["total_steps"]
    for index in range(start, stop):
        row = rows[random.randrange(len(rows))]
        numpy_probe = float(np.random.random())
        cpu_probe = torch.rand(1).item()
        ids = torch.tensor([row["input_ids"]], device="cuda")
        labels = torch.tensor([row["labels"]], device="cuda")
        optimizer.zero_grad(set_to_none=True)
        lr = optimizer.param_groups[0]["lr"]
        with sdpa_kernel(SDPBackend.MATH):
            loss = model(input_ids=ids, attention_mask=torch.ones_like(ids), labels=labels).loss
            assert torch.isfinite(loss)
            loss.backward()
        assert all(torch.isfinite(p.grad).all() for _, p in params if p.grad is not None)
        assert any(p.grad is not None and p.grad.abs().max() > 0 for _, p in params)
        assert all(p.grad is None for n, p in model.named_parameters() if "lora_" not in n)
        norm = torch.nn.utils.clip_grad_norm_([p for _, p in params], 1.0)
        assert torch.isfinite(norm)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        history.append({"step": index + 1, "id": row["id"], "loss": loss.item(),
                        "grad_norm": norm.item(), "lr_before": lr, "lr_after": optimizer.param_groups[0]["lr"],
                        "numpy_probe": numpy_probe, "cpu_probe": cpu_probe})
        save_json(folder / "history.json", history)
        print(name, history[-1], flush=True)
    frozen_after = base_hash(model)
    assert frozen_after == frozen_before
    final = {"adapter": cpu_tree(get_peft_model_state_dict(model)), "optimizer": cpu_tree(optimizer.state_dict()),
             "scheduler": scheduler.state_dict(), "rng": capture_rng(), "completed_steps": stop,
             "history": history, "parameter_names": [n for n, _ in params],
             "protocol_sha256": digest(output / "protocol.json")}
    target = artifact / ("checkpoint-step-2.pt" if name == "prefix" else name + "-final.pt")
    torch.save(final, target)
    report = {"pid": os.getpid(), "executed_steps": stop - start, "completed_steps": stop,
              "initial_adapter_sha256": initial_adapter_hash, "base_unchanged": True,
              "base_sha256": frozen_after, "checkpoint_sha256": digest(target),
              "state_sha256": {key: tree_digest(final[key]) for key in ("adapter", "optimizer", "scheduler", "rng", "completed_steps")}}
    save_json(folder / "state.json", report)


def compare(run):
    output, artifact = ROOT / "results" / run, ROOT / "artifacts" / run
    read = lambda name: json.loads((output / name).read_text(encoding="utf-8"))
    continuous = torch.load(artifact / "continuous-final.pt", map_location="cpu", weights_only=True)
    full = torch.load(artifact / "resume-final.pt", map_location="cpu", weights_only=True)
    negative = torch.load(artifact / "optimizer-reset-final.pt", map_location="cpu", weights_only=True)
    checks = {key: tree_digest(continuous[key]) == tree_digest(full[key])
              for key in ("adapter", "optimizer", "scheduler", "rng", "completed_steps", "history", "parameter_names")}
    exact = all(torch.equal(v, full["adapter"][k]) for k, v in continuous["adapter"].items())
    negative_diff = max((v - negative["adapter"][k]).abs().max().item() for k, v in continuous["adapter"].items())
    pids = [read(name + "/state.json")["pid"] for name in ("continuous", "prefix", "resume", "optimizer-reset")]
    prior = json.loads((ROOT / "results/2026-10-23-boundary-v2/candidate/upload.json").read_text(encoding="utf-8"))
    report = {"full_resume_exact": exact and all(checks.values()), "checks": checks,
              "full_resume_adapter_max_abs_diff": max((v - full["adapter"][k]).abs().max().item() for k, v in continuous["adapter"].items()),
              "optimizer_reset_diverges": negative_diff > 0, "optimizer_reset_adapter_max_abs_diff": negative_diff,
              "four_distinct_processes": len(set(pids)) == 4, "pids": pids,
              "executed_optimizer_steps": 10, "steps_per_final_path": 4, "checkpoint_step": 2,
              "same_initial_adapter": read("continuous/state.json")["initial_adapter_sha256"] == read("prefix/state.json")["initial_adapter_sha256"],
              "negative_first_resumed_loss_matches": negative["history"][2]["loss"] == continuous["history"][2]["loss"],
              "negative_final_loss_matches": negative["history"][3]["loss"] == continuous["history"][3]["loss"],
              "quality_evaluated": False, "promote_to_main": False,
              "published_reference_adapter_sha256": prior["files_sha256"]["adapter_model.safetensors"]}
    save_json(output / "comparison.json", report)
    print(json.dumps(report, ensure_ascii=False), flush=True)
    assert report["full_resume_exact"] and report["optimizer_reset_diverges"] and report["four_distinct_processes"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    parser.add_argument("--phase", choices=("continuous", "prefix", "resume", "optimizer-reset"))
    args = parser.parse_args()
    assert Path(args.run).name == args.run
    if args.phase:
        phase(args.run, args.phase)
        return
    output, artifact = ROOT / "results" / args.run, ROOT / "artifacts" / args.run
    output.mkdir(parents=True, exist_ok=False)
    artifact.mkdir(parents=True, exist_ok=False)
    protocol = json.loads((ROOT / "data/resume-protocol-v1.json").read_text(encoding="utf-8"))
    save_json(output / "protocol.json", protocol)
    reference = json.loads((ROOT / "results/2026-10-23-boundary-v2/candidate/upload.json").read_text(encoding="utf-8"))
    save_json(output / "manifest.json", {"protocol": protocol, "source_sha256": {name: digest(ROOT / name) for name in (
              "resume_experiment.py", "resume_state.py", "data/resume-protocol-v1.json", "data/pilot-v1.json", "train_smoke.py", "contract.py")},
              "published_reference_adapter_sha256": reference["files_sha256"]["adapter_model.safetensors"],
              "scope": "Engineering-only training on fresh base, not a new quality candidate. Checkpoints stay local."})
    save_json(output / "environment.json", {"packages": {name: version(name) for name in ("torch", "transformers", "peft", "numpy", "huggingface_hub")},
              "python": sys.version.split()[0], "CUBLAS_WORKSPACE_CONFIG": os.environ["CUBLAS_WORKSPACE_CONFIG"]})
    for name in protocol["phases"]:
        subprocess.run([sys.executable, "-X", "utf8", str(Path(__file__).resolve()), "--run", args.run, "--phase", name], check=True)
    compare(args.run)


if __name__ == "__main__":
    main()
