"""120 preregistered read-only GPU generations with the existing helpers."""
import argparse
import gc
from importlib.metadata import version
from pathlib import Path
import torch
from peft import PeftModel
from transformers import AutoTokenizer, set_seed
from audit_evaluation import state_hash
from prompt_diagnostic import compose, read, review
from train_smoke import ROOT, REVISION, SNAPSHOT, digest, generate, load_base, prompt_ids, save_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    args = parser.parse_args()
    if Path(args.run).name != args.run:
        raise ValueError("single directory name required")
    protocol, prompts, rows = compose()
    assert protocol["training_steps"] == 0 and protocol["base_revision"] == REVISION
    assert protocol["generation"] == {"do_sample": False, "max_new_tokens": 128}
    upload = read(ROOT / "results/2026-10-23-boundary-v2/candidate/upload.json")
    assert upload["commit"] == protocol["adapter_revision"]
    snapshot = Path(upload["downloaded_snapshot"])
    assert all(digest(snapshot / name) == sha for name, sha in upload["files_sha256"].items())
    folder = ROOT / "results" / args.run
    folder.mkdir(parents=True, exist_ok=False)
    for name, data in (("protocol.json", protocol), ("prompts.json", prompts), ("cases.json", rows)):
        save_json(folder / name, data)
    sources = ("data/prompt-diagnostic-v1.json", "data/pilot-v1.json", "data/audit-v1.json", "prompt_diagnostic.py",
               "evaluate_prompt.py", "train_smoke.py", "contract.py", "audit_evaluation.py", "annotation-rules-v1.md")
    save_json(folder / "manifest.json", {"base_revision": REVISION, "adapter_revision": upload["commit"],
              "adapter_sha256": upload["files_sha256"]["adapter_model.safetensors"],
              "source_sha256": {n: digest(ROOT / n) for n in sources}, "training_steps": 0,
              "conditions_fixed_before_generation": True, "human_review": "pending"})
    set_seed(protocol["seed"])
    assert torch.cuda.is_available()
    save_json(folder / "environment.json", {"packages": {n: version(n) for n in ("torch", "transformers", "peft", "huggingface_hub")},
              "gpu": torch.cuda.get_device_name(0), "cuda": torch.version.cuda, "new_dependencies_installed": False})
    tokenizer = AutoTokenizer.from_pretrained(SNAPSHOT, local_files_only=True)
    save_json(folder / "prompt-tokens.json", {c: [{"id": r["id"], "input_ids": prompt_ids(tokenizer, {"instruction": p}, r)}
                                                   for r in rows] for c, p in prompts.items()})
    for model_name in protocol["models"]:
        model = load_base()
        if model_name == "candidate":
            model = PeftModel.from_pretrained(model, snapshot, local_files_only=True, is_trainable=False)
        model.eval()
        before = state_hash(model)
        hashes = {}
        for condition in protocol["conditions"]:
            print(model_name, condition, "20 fixed notices", flush=True)
            outputs = generate(model, tokenizer, {"instruction": prompts[condition], "rows": rows})
            save_json(folder / f"{model_name}-{condition}.json", outputs)
            hashes[condition] = state_hash(model)
            assert hashes[condition] == before
            print("exact", sum(o["score"]["exact"] for o in outputs), "/", len(outputs), flush=True)
        save_json(folder / f"{model_name}-weights.json", {"before": before, "after_each_condition": hashes, "unchanged": True})
        del model
        gc.collect()
        torch.cuda.empty_cache()
    report = review(folder)
    assert report["total_generation_calls"] == protocol["expected_generation_calls"]
    save_json(folder / "review.json", report)
    print("Complete:", report["total_generation_calls"], "generations; original replay", report["old_audit_tokens_reproduce"], flush=True)


if __name__ == "__main__":
    main()
