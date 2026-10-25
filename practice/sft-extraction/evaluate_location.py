"""Fixed read-only minimal-pair diagnostic, reusing prior generation/hash helpers."""
import argparse
import gc
import json
from importlib.metadata import version
from pathlib import Path
import torch
from peft import PeftModel
from transformers import AutoTokenizer, set_seed
from audit_evaluation import state_hash
from location_minimal import compose, review
from train_smoke import ROOT, REVISION, SNAPSHOT, digest, generate, load_base, save_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    args = parser.parse_args()
    assert Path(args.run).name == args.run
    output = ROOT / "results" / args.run
    output.mkdir(parents=True, exist_ok=False)
    protocol = json.loads((ROOT / "data/location-minimal-protocol-v1.json").read_text(encoding="utf-8"))
    rows = compose()
    assert [r["id"] for r in rows] == protocol["case_ids"]
    assert protocol["training_steps"] == 0 and protocol["base_revision"] == REVISION
    assert protocol["conditions"] == ["base", "candidate"] and protocol["total_generation_calls"] == 12
    assert protocol["generation"] == {"do_sample": False, "max_new_tokens": 128}
    old_upload = json.loads((ROOT / "results/2026-10-23-boundary-v2/candidate/upload.json").read_text(encoding="utf-8"))
    assert old_upload["commit"] == protocol["adapter_revision"]
    snapshot = Path(old_upload["downloaded_snapshot"])
    assert all(digest(snapshot / name) == value for name, value in old_upload["files_sha256"].items())
    instruction = json.loads((ROOT / "data/pilot-v1.json").read_text(encoding="utf-8"))["instruction"]
    sources = ["data/location-minimal-v1.json", "data/location-minimal-protocol-v1.json", "data/audit-v1.json",
               "data/pilot-v1.json", "evaluate_location.py", "location_minimal.py", "audit_evaluation.py",
               "train_smoke.py", "contract.py", "annotation-rules-v1.md"]
    save_json(output / "manifest.json", {"protocol": protocol, "source_sha256": {n: digest(ROOT / n) for n in sources},
              "adapter_sha256": old_upload["files_sha256"]["adapter_model.safetensors"],
              "instruction": instruction, "training_steps": 0, "annotation_rules_in_prompt": False})
    save_json(output / "cases.json", rows)
    save_json(output / "protocol.json", protocol)
    set_seed(protocol["seed"])
    assert torch.cuda.is_available()
    save_json(output / "environment.json", {"packages": {n: version(n) for n in ("torch", "transformers", "peft", "huggingface_hub")},
              "gpu": torch.cuda.get_device_name(0), "cuda": torch.version.cuda, "new_dependencies_installed": False})
    tokenizer = AutoTokenizer.from_pretrained(SNAPSHOT, local_files_only=True)
    for condition in protocol["conditions"]:
        model = load_base()
        if condition == "candidate":
            model = PeftModel.from_pretrained(model, snapshot, local_files_only=True, is_trainable=False)
        model.eval()
        before = state_hash(model)
        print(condition, "generating six fixed variants", flush=True)
        save_json(output / (condition + "-extraction.json"), generate(model, tokenizer, {"instruction": instruction, "rows": rows}))
        after = state_hash(model)
        assert before == after
        save_json(output / (condition + "-weights.json"), {"before": before, "after": after, "unchanged": True, "training_steps": 0})
        del model
        gc.collect()
        torch.cuda.empty_cache()
    report = review(args.run)
    assert report["total_generation_calls"] == 12
    save_json(output / "review.json", report)
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
