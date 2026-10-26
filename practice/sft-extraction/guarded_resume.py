"""Add identity rejection tests and a real guarded resume without editing frozen workers."""
import argparse
import copy
import json
from pathlib import Path
import subprocess
import sys
from importlib.metadata import version
# Imports the frozen worker's deterministic-process environment before CUDA use.
from resume_experiment import phase
import torch
from transformers import AutoTokenizer
from contract import load_pilot
from train_smoke import ROOT, BASE, REVISION, SNAPSHOT, fixture, save_json
from resume_state import tree_digest
from resume_guard import GuardRejected, file_digest, canonical_digest, validate_identity, guarded_load


def identity(run, data_path=None):
    protocol_path = ROOT / "results" / run / "protocol.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    data_path = data_path or ROOT / "data/pilot-v1.json"
    data = load_pilot(data_path)
    tokenizer = AutoTokenizer.from_pretrained(SNAPSHOT, local_files_only=True)
    train = [r for r in data["rows"] if r["split"] == "train"]
    token_fixtures = [fixture(tokenizer, data, r) for r in train]
    return {"base": BASE, "base_revision": REVISION,
            "base_files": {n: file_digest(SNAPSHOT / n) for n in ("config.json", "model.safetensors")},
            "data_sha256": file_digest(data_path), "sample_order": [r["id"] for r in train],
            "tokenized_training_sha256": canonical_digest([{k: r[k] for k in ("id", "input_ids", "labels")} for r in token_fixtures]),
            "tokenizer_files": {n: file_digest(SNAPSHOT / n) for n in ("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt")},
            "chat_template_sha256": canonical_digest(tokenizer.chat_template),
            "special_tokens": {"eos": tokenizer.eos_token_id, "pad": tokenizer.pad_token_id},
            "training_protocol_sha256": file_digest(protocol_path), "training_protocol": protocol,
            "code_sha256": {n: file_digest(ROOT / n) for n in ("resume_experiment.py", "resume_state.py", "train_smoke.py", "contract.py", "resume_guard.py", "guarded_resume.py")},
            "software": {n: version(n) for n in ("torch", "transformers", "peft", "numpy")},
            "python": sys.version.split()[0], "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0), "capability": list(torch.cuda.get_device_capability(0))}


def check_and_dispatch(run, fault=None):
    output, artifact = ROOT / "results" / run, ROOT / "artifacts" / run
    envelope = json.loads((output / "binding.json").read_text(encoding="utf-8"))
    checkpoint = artifact / "checkpoint-step-2.pt"
    data_path = None
    if fault in ("data-content", "data-order"):
        data_path = artifact / (fault + ".json")
        data = load_pilot()
        if fault == "data-content":
            data["rows"][0]["text"] += "（诊断副本）"
        else:
            data["rows"][0], data["rows"][1] = data["rows"][1], data["rows"][0]
        save_json(data_path, data)
    runtime = identity(run, data_path)
    if fault == "base-revision":
        runtime["base_revision"] = "0" * 40
    elif fault == "tokenizer-files":
        runtime["tokenizer_files"]["tokenizer_config.json"] = "0" * 64
    elif fault == "chat-template":
        runtime["chat_template_sha256"] = "0" * 64
    elif fault == "training-config":
        runtime["training_protocol"]["lora"]["r"] = 16
    elif fault == "missing-identity-field":
        del runtime["base_revision"]
    elif fault == "checkpoint-bytes":
        copied = artifact / "corrupted-checkpoint.pt"
        copied.write_bytes(checkpoint.read_bytes() + b"controlled corruption")
        checkpoint = copied
    report = {"fault": fault, "checkpoint_deserialized": False, "training_dispatched": False,
              "optimizer_updates": 0, "rejected": False}

    def deserialize(path):
        report["checkpoint_deserialized"] = True
        return torch.load(path, map_location="cpu", weights_only=True)

    try:
        guarded_load(envelope, checkpoint, runtime, deserialize)
    except GuardRejected as exc:
        report.update(rejected=True, reason=str(exc))
        save_json(output / "rejections" / (str(fault) + ".json"), report)
        print(json.dumps(report), flush=True)
        if fault is None:
            raise
        assert not report["checkpoint_deserialized"] and not report["training_dispatched"]
        return
    if fault is not None:
        save_json(output / "rejections" / (fault + ".json"), report)
        raise RuntimeError("negative identity was incorrectly accepted")
    report["training_dispatched"] = True
    phase(run, "resume")
    report["optimizer_updates"] = 2
    save_json(output / "valid-guard.json", report)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--fault")
    args = parser.parse_args()
    assert Path(args.run).name == args.run
    if args.check:
        check_and_dispatch(args.run, args.fault)
        return
    output, artifact = ROOT / "results" / args.run, ROOT / "artifacts" / args.run
    output.mkdir(parents=True, exist_ok=False)
    artifact.mkdir(parents=True, exist_ok=False)
    guard_protocol = json.loads((ROOT / "data/resume-guard-protocol-v1.json").read_text(encoding="utf-8"))
    protocol = json.loads((ROOT / guard_protocol["training_protocol"]).read_text(encoding="utf-8"))
    save_json(output / "protocol.json", protocol)
    save_json(output / "guard-protocol.json", guard_protocol)
    before = identity(args.run)
    subprocess.run([sys.executable, "-X", "utf8", str(ROOT / "resume_experiment.py"), "--run", args.run, "--phase", "prefix"], check=True)
    validate_identity(before, identity(args.run))
    checkpoint = artifact / "checkpoint-step-2.pt"
    prefix = json.loads((output / "prefix/state.json").read_text(encoding="utf-8"))
    assert file_digest(checkpoint) == prefix["checkpoint_sha256"]
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    binding = {"schema_version": 1, "checkpoint_sha256": file_digest(checkpoint), "identity": before,
               "parameter_order_sha256": canonical_digest(state["parameter_names"])}
    save_json(output / "binding.json", binding)
    for fault in guard_protocol["negative_cases"]:
        subprocess.run([sys.executable, "-X", "utf8", str(Path(__file__).resolve()), "--run", args.run, "--check", "--fault", fault], check=True)
    subprocess.run([sys.executable, "-X", "utf8", str(Path(__file__).resolve()), "--run", args.run, "--check"], check=True)
    reference_run = guard_protocol["reference_run"]
    reference = torch.load(ROOT / "artifacts" / reference_run / "continuous-final.pt", map_location="cpu", weights_only=True)
    actual = torch.load(artifact / "resume-final.pt", map_location="cpu", weights_only=True)
    checks = {k: tree_digest(reference[k]) == tree_digest(actual[k]) for k in ("adapter", "optimizer", "scheduler", "rng", "history", "completed_steps", "parameter_names")}
    errors = [json.loads((output / "rejections" / (n + ".json")).read_text(encoding="utf-8")) for n in guard_protocol["negative_cases"]]
    difference = max((v - actual["adapter"][k]).abs().max().item() for k, v in reference["adapter"].items())
    comparison = {"valid_resume_exact": all(checks.values()), "checks": checks, "adapter_max_abs_diff": difference,
                  "negative_cases": len(errors), "all_rejected_before_deserialization": all(r["rejected"] and not r["checkpoint_deserialized"] and not r["training_dispatched"] for r in errors),
                  "negative_optimizer_updates": sum(r["optimizer_updates"] for r in errors), "executed_optimizer_steps": 4,
                  "source_checkpoint_unchanged": file_digest(checkpoint) == binding["checkpoint_sha256"], "quality_evaluated": False,
                  "scope": guard_protocol["scope"]}
    save_json(output / "comparison.json", comparison)
    old_manifest = json.loads((ROOT / "results" / reference_run / "manifest.json").read_text(encoding="utf-8"))
    save_json(output / "manifest.json", {"guard_protocol": guard_protocol,
              "source_sha256": {n: file_digest(ROOT / n) for n in ("resume_guard.py", "guarded_resume.py", "data/resume-guard-protocol-v1.json")},
              "published_reference_adapter_sha256": old_manifest["published_reference_adapter_sha256"],
              "binding_sha256": file_digest(output / "binding.json")})
    print(json.dumps(comparison, ensure_ascii=False), flush=True)
    assert comparison["valid_resume_exact"] and comparison["all_rejected_before_deserialization"] and comparison["source_checkpoint_unchanged"]


if __name__ == "__main__":
    main()
