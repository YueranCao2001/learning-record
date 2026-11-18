"""Measure cache-dependent output changes across three existing venue pairs."""
import argparse
from importlib.metadata import version
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoTokenizer

from audit_evaluation import state_hash
from confirm_prompt import aggregate
from contract import score
from fit_metrics import first_difference
from prompt_diagnostic import ROOT, read
from time_cue_order_diagnostic import ADAPTERS
from train_smoke import REVISION, SNAPSHOT, digest, fixture, load_base, save_json

PROMPT_RUN = "2026-11-16-venue-field-prompt-v1"
CACHE_RUN = "2026-11-17-venue-location-cache-check-v1"


def compare(model, tokenizer, prompt, row):
    case = fixture(tokenizer, {"instruction": prompt}, row)
    prefix_length = case["prefix_tokens"]
    gold = case["input_ids"][prefix_length:]
    ids = torch.tensor([case["input_ids"][:prefix_length]], device="cuda")
    outputs, scores = {}, {}
    for name, use_cache in (("cached", True), ("uncached", False)):
        with torch.inference_mode():
            generated = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids), do_sample=False,
                                       max_new_tokens=128, use_cache=use_cache, pad_token_id=tokenizer.pad_token_id,
                                       eos_token_id=tokenizer.eos_token_id, return_dict_in_generate=True,
                                       output_scores=True)
        tokens = generated.sequences[0, prefix_length:].cpu().tolist()
        raw = tokenizer.decode(tokens, skip_special_tokens=True)
        outputs[name] = {"id": row["id"], "raw": raw, "token_ids": tokens, "score": score(raw, row["expected"]),
                         "first_gold_difference_zero_based": first_difference(gold, tokens)}
        scores[name] = generated.scores

    cached, uncached = outputs["cached"]["token_ids"], outputs["uncached"]["token_ids"]
    index = first_difference(cached, uncached)
    divergence = None
    if index is not None:
        divergence = {"index_zero_based": index, "shared_prefix": tokenizer.decode(cached[:index])}
        if index < min(len(cached), len(uncached)):
            candidates = {"cached_choice": cached[index], "uncached_choice": uncached[index]}
            divergence["candidate_ids"] = candidates
            divergence["candidate_text"] = {name: tokenizer.decode([token]) for name, token in candidates.items()}
            divergence["probabilities"] = {}
            for name, steps in scores.items():
                probabilities = torch.softmax(steps[index][0].float(), dim=-1)
                divergence["probabilities"][name] = {key: probabilities[token].item()
                                                      for key, token in candidates.items()}
    return {**outputs, "divergence": divergence}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    args = parser.parse_args()
    if Path(args.run).name != args.run or args.run in (".", ".."):
        raise ValueError("new run directory required")
    prior = ROOT / "results" / PROMPT_RUN
    old_cache = ROOT / "results" / CACHE_RUN
    rows = read(prior / "cases.json")["rows"]
    prompts = read(prior / "protocol.json")["prompts"]
    previous = read(old_cache / "review.json")
    previous_manifest = read(old_cache / "manifest.json")
    assert len(rows) == 6 and len({r["group"] for r in rows}) == 3
    assert set(prompts) == {"original", "field-independence"}
    run, arm = ADAPTERS["target"]
    adapter = ROOT / "artifacts" / run / arm
    adapter_hash = digest(adapter / "adapter_model.safetensors")
    assert adapter_hash == previous_manifest["local_adapter_file_sha256"] and REVISION == previous_manifest["base_revision"]
    sources = ["venue_cache_scope.py", "time_cue_order_diagnostic.py", "time_regression.py",
               "train_smoke.py", "audit_evaluation.py", "contract.py", "prompt_diagnostic.py", "confirm_prompt.py",
               "fit_metrics.py"]
    sources += [f"results/{PROMPT_RUN}/{name}" for name in ("cases.json", "protocol.json", *(f"{c}-generation.json" for c in prompts))]
    sources += [f"results/{CACHE_RUN}/{name}" for name in ("cases.json", "review.json", "manifest.json")]
    manifest = {"source_sha256": {name: digest(ROOT / name) for name in sources},
                "base_revision": REVISION, "local_adapter_file_sha256": adapter_hash,
                "published_reference_adapter_sha256": previous_manifest["published_reference_adapter_sha256"],
                "generation": previous_manifest["generation"], "gpu": torch.cuda.get_device_name(0),
                "cuda": torch.version.cuda, "packages": {name: version(name) for name in ("torch", "transformers", "peft")}}
    output = ROOT / "results" / args.run
    output.mkdir(parents=True, exist_ok=False)
    save_json(output / "cases.json", {"rows": rows, "prompts": prompts})
    save_json(output / "manifest.json", manifest)

    tokenizer = AutoTokenizer.from_pretrained(SNAPSHOT, local_files_only=True)
    model = PeftModel.from_pretrained(load_base(), adapter, local_files_only=True, is_trainable=False)
    model.eval()
    assert not any(p.requires_grad for p in model.parameters())
    before = state_hash(model)
    assert before == previous["parameters_after"]
    cache_references = {(pair["prompt"], pair["id"]): pair["uncached"]["token_ids"] for pair in previous["pairs"]}
    results = {}
    for name, prompt in prompts.items():
        cached_references = {row["id"]: row["token_ids"] for row in read(prior / f"{name}-generation.json")}
        pairs = []
        for row in rows:
            comparison = compare(model, tokenizer, prompt, row)
            cached_equal = comparison["cached"]["token_ids"] == cached_references[row["id"]]
            uncached_reference = cache_references.get((name, row["id"]))
            uncached_equal = None if uncached_reference is None else comparison["uncached"]["token_ids"] == uncached_reference
            assert cached_equal and uncached_equal is not False
            pairs.append({"id": row["id"], "group": row["group"], "time_state": row["time_state"],
                          "cached_replay_equal": cached_equal, "prior_uncached_replay_equal": uncached_equal, **comparison})
            print(name, row["id"], comparison["cached"]["raw"], comparison["uncached"]["raw"], flush=True)
        summaries, errors = {}, {}
        for path in ("cached", "uncached"):
            summaries[path], _, errors[path] = aggregate(rows, [pair[path] for pair in pairs])
        results[name] = {"summaries": summaries, "errors": errors, "pairs": pairs}
    after = state_hash(model)
    assert before == after
    pairs = [pair for condition in results.values() for pair in condition["pairs"]]
    save_json(output / "review.json", {"date": "2026-11-18", "training_steps": 0, "human_review": "pending",
              "promote_to_main": False, "weights_unchanged": True, "parameters_before": before, "parameters_after": after,
              "generation_calls": 2 * len(pairs), "source_groups": 3, "new_source_groups": 0,
              "known_cached_replays": sum(pair["cached_replay_equal"] for pair in pairs),
              "known_uncached_replays": sum(pair["prior_uncached_replay_equal"] is True for pair in pairs), "results": results})


if __name__ == "__main__":
    main()
