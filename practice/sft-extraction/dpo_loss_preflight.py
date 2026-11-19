"""Score one frozen preference pair and independently verify the DPO loss."""
import argparse
import gc
from importlib.metadata import version
import math
from pathlib import Path

import torch
from peft import PeftModel
from torch.nn import functional as F
from transformers import AutoTokenizer

from audit_evaluation import state_hash
from dpo_pair_preflight import validate
from fit_metrics import target_statistics
from prompt_diagnostic import ROOT, read
from time_cue_order_diagnostic import ADAPTERS
from train_smoke import REVISION, SNAPSHOT, digest, fixture, load_base, save_json

PAIR_RUN = "2026-11-19-dpo-preference-pairs-v1"
ADAPTER_EVIDENCE = "results/2026-11-18-venue-stage-closeout-v1/manifest.json"
BETA = 0.1


def sequence_logp(model, sample):
    input_ids = torch.tensor([sample["input_ids"]], device="cuda")
    labels = torch.tensor(sample["labels"], device="cuda")
    logits = model(input_ids=input_ids, attention_mask=torch.ones_like(input_ids), use_cache=False).logits[0]
    stats = target_statistics(logits, labels)
    first = sample["prefix_tokens"]
    assert stats["positions"] == list(range(first, len(sample["input_ids"])))
    assert stats["target_ids"] == sample["input_ids"][first:]
    logp = math.fsum(stats["target_logps"])
    mask = labels[1:] != -100
    cross_entropy_sum = F.cross_entropy(logits[:-1][mask].float(), labels[1:][mask], reduction="sum").item()
    assert math.isclose(logp, -cross_entropy_sum, abs_tol=1e-5, rel_tol=0)
    return {**stats, "sequence_logp": logp, "cross_entropy_sum": cross_entropy_sum,
            "scored_tokens": len(stats["target_ids"])}


def dpo_loss(policy_margin, reference_margin, beta):
    logit = beta * (policy_margin - reference_margin)
    loss = -F.logsigmoid(torch.tensor(logit, dtype=torch.float64)).item()
    assert math.isclose(loss, math.log1p(math.exp(-logit)), abs_tol=1e-12, rel_tol=0)
    return {"policy_margin": policy_margin, "reference_margin": reference_margin, "beta": beta,
            "logit": logit, "loss": loss}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    args = parser.parse_args()
    if Path(args.run).name != args.run or args.run in (".", ".."):
        raise ValueError("new run directory required")
    prior = ROOT / "results" / PAIR_RUN
    pair_data = read(prior / "pairs.json")
    validate(pair_data)
    row = next(row for row in pair_data["rows"] if row["id"] == "dpo-01")
    assert row["split"] == "train"
    previous = next(record for record in read(prior / "review.json")["token_records"] if record["id"] == row["id"])
    tokenizer_manifest = read(prior / "manifest.json")
    assert tokenizer_manifest["tokenizer_revision"] == REVISION
    assert all(digest(SNAPSHOT / n) == expected for n, expected in tokenizer_manifest["tokenizer_files_sha256"].items())
    tokenizer = AutoTokenizer.from_pretrained(SNAPSHOT, local_files_only=True)
    samples = {name: fixture(tokenizer, pair_data, {**row, "expected": row[name]}) for name in ("chosen", "rejected")}
    for name, sample in samples.items():
        prefix_length = sample["prefix_tokens"]
        assert sample["input_ids"][:prefix_length] == previous["shared_prompt_token_ids"]
        assert sample["input_ids"][prefix_length:] == previous["completion_token_ids"][name]

    run, arm = ADAPTERS["target"]
    adapter = ROOT / "artifacts" / run / arm
    adapter_hash = digest(adapter / "adapter_model.safetensors")
    prior_adapter = read(ROOT / ADAPTER_EVIDENCE)
    assert adapter_hash == prior_adapter["local_adapter_file_sha256"] and REVISION == prior_adapter["base_revision"]
    sources = ["dpo_loss_preflight.py", "dpo_pair_preflight.py", "train_smoke.py", "contract.py", "fit_metrics.py",
               "prompt_diagnostic.py", "audit_evaluation.py", "time_cue_order_diagnostic.py", ADAPTER_EVIDENCE]
    sources += [f"results/{PAIR_RUN}/{name}" for name in ("pairs.json", "manifest.json", "review.json")]
    output = ROOT / "results" / args.run
    output.mkdir(parents=True, exist_ok=False)
    save_json(output / "pair.json", {"instruction": pair_data["instruction"], "row": row,
                                      "shared_prefix_tokens": previous["prefix_tokens"]})
    save_json(output / "manifest.json", {"planned_date": "2026-11-20", "executed_date": "2026-11-19",
              "source_sha256": {name: digest(ROOT / name) for name in sources}, "base_revision": REVISION,
              "local_adapter_file_sha256": adapter_hash,
              "published_reference_adapter_sha256": prior_adapter["published_reference_adapter_sha256"],
              "policy_and_reference_same_initial_adapter": True, "base_dtype": "bfloat16",
              "scoring_dtype": "float32", "gpu": torch.cuda.get_device_name(0), "cuda": torch.version.cuda,
              "packages": {name: version(name) for name in ("torch", "transformers", "peft")}})

    roles = {}
    for role in ("policy", "reference"):
        model = PeftModel.from_pretrained(load_base(), adapter, local_files_only=True, is_trainable=False).eval()
        assert not any(p.requires_grad for p in model.parameters())
        before = state_hash(model)
        with torch.inference_mode():
            responses = {name: sequence_logp(model, sample) for name, sample in samples.items()}
        after = state_hash(model)
        assert before == after
        roles[role] = {"parameters_before": before, "parameters_after": after, "responses": responses}
        del model
        gc.collect()
        torch.cuda.empty_cache()
    assert roles["policy"] == roles["reference"]
    margins = {role: data["responses"]["chosen"]["sequence_logp"] - data["responses"]["rejected"]["sequence_logp"]
               for role, data in roles.items()}
    measured = dpo_loss(margins["policy"], margins["reference"], BETA)
    assert measured["logit"] == 0 and math.isclose(measured["loss"], math.log(2), abs_tol=1e-12, rel_tol=0)
    sign_checks = {name: dpo_loss(margin, 0, BETA) for name, margin in (("positive", 1), ("negative", -1))}
    assert sign_checks["positive"]["loss"] < measured["loss"] < sign_checks["negative"]["loss"]
    save_json(output / "review.json", {"training_steps": 0, "human_review": "pending", "model_generations": 0,
              "forwards": 4, "pair_id": row["id"], "roles": roles, "measured_loss": measured,
              "synthetic_sign_checks": sign_checks})
    for role, data in roles.items():
        print(role, {name: r["sequence_logp"] for name, r in data["responses"].items()}, flush=True)
    print("DPO", measured, flush=True)


if __name__ == "__main__":
    main()
