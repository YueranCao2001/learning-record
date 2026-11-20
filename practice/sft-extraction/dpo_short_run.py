"""Run one DPO pass over three frozen training pairs and audit a fixed development set."""
import argparse
import gc
from importlib.metadata import version
import math
from pathlib import Path

import torch
from peft import PeftModel
from torch.nn import functional as F
from transformers import AutoTokenizer, set_seed

from audit_evaluation import state_hash
from confirm_prompt import aggregate
from dpo_loss_preflight import ADAPTER_EVIDENCE, BETA, PAIR_RUN, dpo_loss
from dpo_pair_preflight import validate
from prompt_diagnostic import ROOT, read
from review_repair import paired
from time_cue_order_diagnostic import ADAPTERS
from train_smoke import BASE, REVISION, SNAPSHOT, base_hash, digest, fixture, generate, load_base, save_json

PREFLIGHT = "2026-11-20-dpo-loss-preflight-v1"
REGRESSION = "2026-11-18-venue-stage-closeout-v1"
LR = 2e-5


def completion_logp(model, sample):
    input_ids = torch.tensor([sample["input_ids"]], device="cuda")
    labels = torch.tensor(sample["labels"], device="cuda")[1:]
    logits = model(input_ids=input_ids, attention_mask=torch.ones_like(input_ids), use_cache=False).logits[0, :-1]
    mask = labels != -100
    return -F.cross_entropy(logits[mask].float(), labels[mask], reduction="sum")


def pair_scores(model, samples):
    return {name: completion_logp(model, sample) for name, sample in samples.items()}


def margin(scores):
    return scores["chosen"] - scores["rejected"]


def score_pairs(model, fixtures, identities):
    with torch.inference_mode():
        return {identity: {name: value.item() for name, value in pair_scores(model, fixtures[identity]).items()}
                for identity in identities}


def evaluate(model, tokenizer, data, fixtures, dev_ids):
    model.eval()
    return {"preference_scores": score_pairs(model, fixtures, dev_ids), "generations": generate(model, tokenizer, data)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    args = parser.parse_args()
    if Path(args.run).name != args.run or args.run in (".", ".."):
        raise ValueError("new run directory required")
    pair_dir = ROOT / "results" / PAIR_RUN
    data = read(pair_dir / "pairs.json")
    validate(data)
    train = [r["id"] for r in data["rows"] if r["split"] == "train"]
    dev = [r["id"] for r in data["rows"] if r["split"] == "dev"]
    assert len(train) == len(dev) == 3 and set(train).isdisjoint(dev)
    tokenizer = AutoTokenizer.from_pretrained(SNAPSHOT, local_files_only=True)
    frozen = {row["id"]: row for row in read(pair_dir / "review.json")["token_records"]}
    fixtures = {}
    for row in data["rows"]:
        identity = row["id"]
        previous = frozen[identity]
        fixtures[identity] = {}
        for name in ("chosen", "rejected"):
            sample = fixture(tokenizer, data, {**row, "expected": row[name]})
            assert sample["input_ids"] == previous["shared_prompt_token_ids"] + previous["completion_token_ids"][name]
            assert sample["labels"] == [-100] * previous["prefix_tokens"] + previous["completion_token_ids"][name]
            fixtures[identity][name] = sample
    regression_dir = ROOT / "results" / REGRESSION
    prior_regression = read(regression_dir / "cases.json")
    assert data["instruction"] == prior_regression["prompts"]["original"]
    dev_rows = [r for r in data["rows"] if r["split"] == "dev"]
    evaluation = {"instruction": data["instruction"],
                  "rows": [{**r, "cohort": "dpo-dev"} for r in dev_rows] + prior_regression["rows"]}
    assert len(evaluation["rows"]) == 14
    assert set(train).isdisjoint(r["id"] for r in evaluation["rows"])
    adapter_run, arm = ADAPTERS["target"]
    start_adapter = ROOT / "artifacts" / adapter_run / arm
    prior_adapter = read(ROOT / ADAPTER_EVIDENCE)
    start_sha = digest(start_adapter / "adapter_model.safetensors")
    assert start_sha == prior_adapter["local_adapter_file_sha256"] and REVISION == prior_adapter["base_revision"]
    prior_numeric = read(ROOT / "results" / PREFLIGHT / "review.json")
    output = ROOT / "results" / args.run
    artifact = ROOT / "artifacts" / args.run / "policy"
    output.mkdir(parents=True, exist_ok=False)
    assert not artifact.exists()
    sources = ["dpo_short_run.py", "dpo_loss_preflight.py", "dpo_pair_preflight.py", "train_smoke.py",
               "contract.py", "audit_evaluation.py", "confirm_prompt.py", "prompt_diagnostic.py", "review_repair.py",
               "time_cue_order_diagnostic.py", ADAPTER_EVIDENCE, f"results/{PREFLIGHT}/review.json"]
    sources += [f"results/{PAIR_RUN}/{name}" for name in ("pairs.json", "review.json", "manifest.json")]
    sources += [f"results/{REGRESSION}/{name}" for name in ("cases.json", "review.json")]
    save_json(output / "manifest.json", {"executed_date": "2026-11-20", "planned_dates": ["2026-11-21", "2026-11-22"],
              "base_revision": REVISION, "local_initial_adapter_sha256": start_sha,
              "published_reference_adapter_sha256": prior_adapter["published_reference_adapter_sha256"],
              "source_sha256": {n: digest(ROOT / n) for n in sources}, "gpu": torch.cuda.get_device_name(0),
              "packages": {n: version(n) for n in ("torch", "transformers", "peft")},
              "train_order": train, "dev_ids": dev, "beta": BETA, "optimizer": "AdamW", "lr": LR,
              "batch_pairs": 1, "weight_decay": 0, "seed": 42, "base_dtype": "bfloat16",
              "gradient_clip": 1.0, "generation": {"greedy": True, "use_cache": True, "max_new_tokens": 128}})
    save_json(output / "evaluation-inputs.json", evaluation)

    set_seed(42)
    reference = PeftModel.from_pretrained(load_base(), start_adapter, local_files_only=True, is_trainable=False).eval()
    assert not any(p.requires_grad for p in reference.parameters())
    reference_before = state_hash(reference)
    assert reference_before == prior_numeric["roles"]["reference"]["parameters_after"]
    anchored = score_pairs(reference, fixtures, train + dev)
    for name, value in anchored["dpo-01"].items():
        earlier = prior_numeric["roles"]["reference"]["responses"][name]["sequence_logp"]
        assert math.isclose(value, earlier, abs_tol=1e-5, rel_tol=0)
    baseline = generate(reference, tokenizer, evaluation)
    past = read(regression_dir / "review.json")["regression"]["original"]["outputs"]
    assert [(r["id"], r["token_ids"]) for r in baseline[3:]] == [(r["id"], r["token_ids"]) for r in past]
    reference_after = state_hash(reference)
    assert reference_before == reference_after
    save_json(output / "before.json", {"preference_scores": {key: anchored[key] for key in dev}, "generations": baseline})
    save_json(output / "reference-scores.json", anchored)
    del reference
    gc.collect()
    torch.cuda.empty_cache()

    policy = PeftModel.from_pretrained(load_base(), start_adapter, local_files_only=True, is_trainable=True)
    assert state_hash(policy) == reference_before
    params = {name: p for name, p in policy.named_parameters() if p.requires_grad}
    assert params and all("lora_" in name for name in params)
    initial = {name: p.detach().cpu().clone() for name, p in params.items()}
    base_before = base_hash(policy)
    optimizer = torch.optim.AdamW(params.values(), lr=LR, weight_decay=0, foreach=False, fused=False)
    policy.train()
    steps = []
    for identity in train:
        optimizer.zero_grad(set_to_none=True)
        scores = pair_scores(policy, fixtures[identity])
        logit = BETA * (margin(scores) - margin(anchored[identity]))
        loss = -F.logsigmoid(logit)
        assert torch.isfinite(loss)
        numeric = dpo_loss(margin({n: v.item() for n, v in scores.items()}), margin(anchored[identity]), BETA)
        assert math.isclose(loss.item(), numeric["loss"], abs_tol=1e-6, rel_tol=0)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(params.values(), 1.0, error_if_nonfinite=True).item()
        assert norm > 0 and all(p.grad is None for p in policy.parameters() if not p.requires_grad)
        optimizer.step()
        steps.append({"id": identity, "loss": loss.item(), "logit": logit.item(), "gradient_norm_before_clip": norm})
        print("training", steps[-1], flush=True)
    assert abs(steps[0]["logit"]) < 1e-5
    assert {int(s["step"].item()) for s in optimizer.state.values()} == {3}
    base_after = base_hash(policy)
    assert base_before == base_after
    trained = {name: p.detach().cpu().clone() for name, p in params.items()}
    changed = sum(not torch.equal(initial[name], tensor) for name, tensor in trained.items())
    assert changed > 0
    trained_hash = state_hash(policy)
    after = evaluate(policy, tokenizer, evaluation, fixtures, dev)
    policy.peft_config["default"].base_model_name_or_path = BASE
    policy.peft_config["default"].revision = REVISION
    policy.save_pretrained(artifact, safe_serialization=True, save_embedding_layers=False)
    save_json(output / "after.json", after)
    save_json(output / "training.json", {"steps": steps, "changed_adapter_tensors": changed,
              "trainable_parameters": sum(p.numel() for p in params.values()), "initial_policy_sha256": reference_before,
              "trained_policy_sha256": trained_hash, "reference_before": reference_before, "reference_after": reference_after,
              "base_before": base_before, "base_after": base_after,
              "saved_adapter_sha256": digest(artifact / "adapter_model.safetensors")})
    del policy, optimizer, params, scores, loss, logit
    gc.collect()
    torch.cuda.empty_cache()

    reloaded = PeftModel.from_pretrained(load_base(), artifact, local_files_only=True, is_trainable=False)
    assert state_hash(reloaded) == trained_hash
    replay = evaluate(reloaded, tokenizer, evaluation, fixtures, dev)
    assert after == replay
    save_json(output / "reloaded.json", replay)

    preferences = {}
    for identity in dev:
        before_margin = margin(anchored[identity])
        after_scores = after["preference_scores"][identity]
        preferences[identity] = {"before_scores": anchored[identity], "after_scores": after_scores,
                                 "margin_change": margin(after_scores) - before_margin,
                                 "before_margin": before_margin, "after_margin": margin(after_scores)}
    cohorts = {}
    for name, start, stop in (("dpo-dev", 0, 3), ("historical-regression", 3, 14)):
        rows = evaluation["rows"][start:stop]
        a, b = baseline[start:stop], after["generations"][start:stop]
        before_summary, _, before_errors = aggregate(rows, a)
        after_summary, _, after_errors = aggregate(rows, b)
        cohorts[name] = {"before": before_summary, "after": after_summary, "changes": paired(rows, a, b),
                         "before_errors": before_errors, "after_errors": after_errors}
    save_json(output / "review.json", {"training_steps": len(steps), "human_review": "pending",
              "generation_calls": len(baseline) + len(after["generations"]) + len(replay["generations"]),
              "preference_forwards": 2 * (len(anchored) + len(steps) + len(dev) * 2), "history_replayed": len(past),
              "reload_exact": True, "reference_unchanged": True, "base_unchanged": True,
              "preference_deltas": preferences, "cohorts": cohorts, "train_weights_uploaded": False})


if __name__ == "__main__":
    main()
