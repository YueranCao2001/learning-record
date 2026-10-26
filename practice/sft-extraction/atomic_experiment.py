"""Controlled writer exits, corrupted-file refusal and a real container-based GPU resume."""
import argparse
import copy
import hashlib
import io
import json
from pathlib import Path
import shutil
import struct
import subprocess
import sys
from checkpoint_container import MAGIC, MAX_HEADER, write_atomic, load_container, ContainerRejected
from resume_guard import file_digest

ROOT = Path(__file__).resolve().parent


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def gpu_resume(run):
    # Frozen worker sets deterministic environment before torch/CUDA use.
    from resume_experiment import phase
    import torch
    from guarded_resume import identity
    from resume_state import tree_digest
    output, artifact = ROOT / "results" / run, ROOT / "artifacts" / run
    slot = artifact / "writer-cases/normal/slot.ckpt"
    runtime = identity(run)
    verified_payload = []

    def deserialize(payload):
        verified_payload.append(payload)
        return torch.load(io.BytesIO(payload), map_location="cpu", weights_only=True)

    header, state = load_container(slot, runtime, deserialize)
    assert state["completed_steps"] == 2
    checkpoint = artifact / "checkpoint-step-2.pt"
    checkpoint.write_bytes(verified_payload[0])
    assert file_digest(checkpoint) == header["binding"]["checkpoint_sha256"]
    phase(run, "resume")
    actual = torch.load(artifact / "resume-final.pt", map_location="cpu", weights_only=True)
    reference = torch.load(ROOT / "artifacts/2026-10-26-exact-resume-v1/continuous-final.pt", map_location="cpu", weights_only=True)
    checks = {k: tree_digest(actual[k]) == tree_digest(reference[k]) for k in (
        "adapter", "optimizer", "scheduler", "rng", "history", "completed_steps", "parameter_names")}
    difference = max((v - reference["adapter"][k]).abs().max().item() for k, v in actual["adapter"].items())
    report = {"checks": checks, "exact": all(checks.values()), "adapter_max_abs_diff": difference,
              "optimizer_updates": 2, "container_generation": header["generation"],
              "loaded_verified_payload": True}
    save_json(output / "gpu-resume.json", report)
    assert report["exact"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    parser.add_argument("--writer-case")
    parser.add_argument("--gpu", action="store_true")
    args = parser.parse_args()
    assert args.run not in (".", "..") and Path(args.run).name == args.run
    output, artifact = ROOT / "results" / args.run, ROOT / "artifacts" / args.run
    protocol_path = ROOT / "data/atomic-checkpoint-protocol-v1.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    source_run = protocol["source_run"]
    source_payload = ROOT / "artifacts" / source_run / "checkpoint-step-2.pt"
    binding = json.loads((ROOT / "results" / source_run / "binding.json").read_text(encoding="utf-8"))
    if args.writer_case:
        assert args.writer_case in protocol["writer_cases"]
        write_atomic(artifact / "writer-cases" / args.writer_case / "slot.ckpt", source_payload, binding,
                     "new-generation", fault=None if args.writer_case == "normal" else args.writer_case)
        return
    if args.gpu:
        gpu_resume(args.run)
        return
    output.mkdir(parents=True, exist_ok=False)
    artifact.mkdir(parents=True, exist_ok=False)
    save_json(output / "atomic-protocol.json", protocol)
    source_output = ROOT / "results" / source_run
    shutil.copyfile(source_output / "protocol.json", output / "protocol.json")
    (output / "prefix").mkdir()
    shutil.copyfile(source_output / "prefix/state.json", output / "prefix/state.json")
    writer_reports = []
    for case in protocol["writer_cases"]:
        slot = artifact / "writer-cases" / case / "slot.ckpt"
        write_atomic(slot, source_payload, binding, "old-generation")
        old_sha = file_digest(slot)
        result = subprocess.run([sys.executable, "-X", "utf8", str(Path(__file__).resolve()), "--run", args.run, "--writer-case", case])
        assert result.returncode == protocol["expected_exit_codes"][case]
        header, _ = load_container(slot, binding["identity"], lambda payload: len(payload))
        expect_old = case in ("mid-payload", "before-replace")
        assert header["generation"] == ("old-generation" if expect_old else "new-generation")
        if expect_old:
            assert file_digest(slot) == old_sha
        row = {"case": case, "exit_code": result.returncode, "readable": True,
               "generation": header["generation"], "old_bytes_preserved": file_digest(slot) == old_sha,
               "pending_file_count": len(list(slot.parent.glob(".pending-*")))}
        writer_reports.append(row)
        print(row, flush=True)
    save_json(output / "writer-cases.json", writer_reports)
    valid = (artifact / "writer-cases/normal/slot.ckpt").read_bytes()
    corruption_reports = []
    for case in protocol["corruption_cases"]:
        raw, runtime = bytearray(valid), copy.deepcopy(binding["identity"])
        if case == "truncated":
            raw = raw[:-17]
        elif case == "payload-bitflip":
            raw[-1] ^= 1
        elif case == "malformed-header":
            raw[len(MAGIC) + 8] = ord(']')
        elif case == "oversized-header":
            raw[len(MAGIC):len(MAGIC) + 8] = struct.pack(">Q", MAX_HEADER + 1)
        else:
            runtime["base_revision"] = "0" * 40
        bad = artifact / "invalid" / (case + ".ckpt")
        bad.parent.mkdir(exist_ok=True)
        bad.write_bytes(raw)
        report = {"case": case, "deserializer_called": False, "optimizer_updates": 0, "rejected": False}

        def forbidden(payload):
            report["deserializer_called"] = True
            raise RuntimeError("invalid container reached deserialization")

        try:
            load_container(bad, runtime, forbidden)
        except ContainerRejected as exc:
            report.update(rejected=True, reason=str(exc))
        assert report["rejected"] and not report["deserializer_called"]
        corruption_reports.append(report)
        print(report, flush=True)
    save_json(output / "corruption-cases.json", corruption_reports)
    subprocess.run([sys.executable, "-X", "utf8", str(Path(__file__).resolve()), "--run", args.run, "--gpu"], check=True)
    gpu = json.loads((output / "gpu-resume.json").read_text(encoding="utf-8"))
    comparison = {"writer_cases_passed": 4, "invalid_containers_rejected": 5,
                  "all_invalid_rejected_before_deserialization": all(r["rejected"] and not r["deserializer_called"] for r in corruption_reports),
                  "gpu_resume_exact": gpu["exact"], "adapter_max_abs_diff": gpu["adapter_max_abs_diff"],
                  "executed_optimizer_steps": 2, "quality_evaluated": False, "real_power_failure_tested": False,
                  "scope": protocol["scope"]}
    save_json(output / "comparison.json", comparison)
    old_manifest = json.loads((ROOT / "results" / source_run / "manifest.json").read_text(encoding="utf-8"))
    save_json(output / "manifest.json", {"source_sha256": {n: file_digest(ROOT / n) for n in (
        "checkpoint_container.py", "atomic_experiment.py", "data/atomic-checkpoint-protocol-v1.json", "resume_guard.py", "guarded_resume.py", "resume_experiment.py")},
        "source_binding_sha256": file_digest(source_output / "binding.json"),
        "published_reference_adapter_sha256": old_manifest["published_reference_adapter_sha256"]})
    print(json.dumps(comparison, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
