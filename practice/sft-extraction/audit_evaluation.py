"""Read-only GPU evaluation of the pinned base and yesterday's pinned adapter."""
import argparse
import gc
import hashlib
import json
from importlib.metadata import version
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoTokenizer, set_seed
from audit_contract import load_audit, retention_score
from train_smoke import ROOT, BASE, REVISION, SNAPSHOT, digest, generate, load_base, save_json


def state_hash(model):
    h = hashlib.sha256()
    for name, param in model.named_parameters():
        h.update(name.encode())
        h.update(param.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def retention(model, tokenizer, protocol):
    result = []
    model.eval()
    for row in protocol["retention"]["rows"]:
        messages = [{"role": "system", "content": protocol["retention"]["system"]},
                    {"role": "user", "content": row["prompt"]}]
        ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                    enable_thinking=False, return_tensors="pt").to("cuda")
        with torch.inference_mode():
            tokens = model.generate(ids, attention_mask=torch.ones_like(ids),
                        do_sample=False, max_new_tokens=128, pad_token_id=tokenizer.pad_token_id,
                        eos_token_id=tokenizer.eos_token_id, use_cache=True)
        new = tokens[0, ids.shape[1]:].cpu().tolist()
        raw = tokenizer.decode(new, skip_special_tokens=True)
        result.append({"id": row["id"], "raw": raw, "token_ids": new,
                       "ended_with_eos": new[-1] == tokenizer.eos_token_id,
                       "correct": retention_score(raw, row)})
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    args = parser.parse_args()
    assert Path(args.run).name == args.run
    output = ROOT / "results" / args.run
    output.mkdir(parents=True, exist_ok=False)
    audit, prior = load_audit()
    protocol = json.loads((ROOT / "data/audit-protocol-v1.json").read_text(encoding="utf-8"))
    assert protocol["base"] == BASE and protocol["base_revision"] == REVISION
    assert protocol["adapter_revision"] == "e3b24d1115c21cf69feaa5e6e9cf85a98f502c84"
    assert protocol["training_steps"] == 0 and len(audit["rows"]) == 12
    upload = json.loads((ROOT / "results/2026-10-23-boundary-v2/candidate/upload.json").read_text(encoding="utf-8"))
    assert upload["commit"] == protocol["adapter_revision"] and upload["repo"] == protocol["adapter_repo"]
    snapshot = Path(upload["downloaded_snapshot"])
    assert snapshot.name == upload["commit"]
    assert all(digest(snapshot / name) == expected for name, expected in upload["files_sha256"].items())
    # The old helper filters split='dev'; this view is inference-only, never training data.
    data = {"instruction": prior["instruction"], "rows": [r for r in prior["rows"] if r["split"] == "dev"]
            + [dict(r, split="dev") for r in audit["rows"]]}
    assert len(data["rows"]) == 21
    sources = ["data/audit-v1.json", "data/audit-protocol-v1.json", "annotation-rules-v1.md",
               "audit_evaluation.py", "audit_contract.py", "contract.py", "train_smoke.py",
               "results/2026-10-23-boundary-v2/data.json"]
    save_json(output / "manifest.json", {"protocol": protocol,
              "source_sha256": {name: digest(ROOT / name) for name in sources},
              "adapter_sha256": upload["files_sha256"]["adapter_model.safetensors"],
              "instruction": data["instruction"], "training_steps": 0,
              "annotation_rules_in_prompt": False, "human_review": "pending"})
    save_json(output / "extraction-data.json", data)
    save_json(output / "protocol.json", protocol)
    set_seed(protocol["seed"])
    assert torch.cuda.is_available()
    save_json(output / "environment.json", {"packages": {n: version(n) for n in (
              "torch", "transformers", "peft", "huggingface_hub")}, "gpu": torch.cuda.get_device_name(0),
              "cuda": torch.version.cuda, "new_dependencies_installed": False})
    tokenizer = AutoTokenizer.from_pretrained(SNAPSHOT, local_files_only=True)
    for condition in ("base", "candidate"):
        model = load_base()
        if condition == "candidate":
            model = PeftModel.from_pretrained(model, snapshot, local_files_only=True, is_trainable=False)
        model.eval()
        initial = state_hash(model)
        print(condition, "generating 21 extraction cases", flush=True)
        save_json(output / (condition + "-extraction.json"), generate(model, tokenizer, data))
        print(condition, "generating 5 retention probes", flush=True)
        save_json(output / (condition + "-retention.json"), retention(model, tokenizer, protocol))
        after = state_hash(model)
        assert initial == after
        save_json(output / (condition + "-weights.json"), {"before": initial, "after": after,
                  "unchanged": True, "training_steps": 0})
        del model
        gc.collect()
        torch.cuda.empty_cache()
    from review_audit import review
    report = review(args.run)
    save_json(output / "review.json", report)
    print(json.dumps(report["metrics"], ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
