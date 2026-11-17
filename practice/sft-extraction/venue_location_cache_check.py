"""Compare cached and uncached generation on the blue-printing venue pair."""
import argparse
from importlib.metadata import version
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoTokenizer

from audit_evaluation import state_hash
from contract import score
from prompt_diagnostic import ROOT, read
from time_cue_order_diagnostic import ADAPTERS
from train_smoke import REVISION, SNAPSHOT, digest, load_base, prompt_ids, save_json

PRIOR = "2026-11-17-venue-location-token-trace-v1"


def generate_with_scores(model, tokenizer, prompt, row, reference, use_cache):
    ids = torch.tensor([prompt_ids(tokenizer, {"instruction": prompt}, row)], device="cuda")
    with torch.inference_mode():
        generated = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids), do_sample=False,
                                   max_new_tokens=128, use_cache=use_cache, pad_token_id=tokenizer.pad_token_id,
                                   eos_token_id=tokenizer.eos_token_id, return_dict_in_generate=True,
                                   output_scores=True)
    tokens = generated.sequences[0, ids.shape[1]:].cpu().tolist()
    position = reference["value_token_index_zero_based"]
    old_tokens = reference["generated_token_ids"]
    assert tokens[:position] == old_tokens[:position] and len(generated.scores) > position
    replay_equal = tokens == old_tokens
    if use_cache:
        assert replay_equal
    probabilities = torch.softmax(generated.scores[position][0].float(), dim=-1)
    gold_id, bare_id = reference["gold_token_id"], reference["bare_colon_token_id"]
    raw = tokenizer.decode(tokens, skip_special_tokens=True)
    return {"use_cache": use_cache, "raw": raw, "token_ids": tokens, "replay_equal": replay_equal,
            "score": score(raw, row["expected"]), "chosen_token_id": tokens[position],
            "gold_token_p": probabilities[gold_id].item(), "bare_colon_token_p": probabilities[bare_id].item()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    args = parser.parse_args()
    if Path(args.run).name != args.run or args.run in (".", ".."):
        raise ValueError("new run directory required")
    prior = ROOT / "results" / PRIOR
    data = read(prior / "cases.json")
    previous = read(prior / "review.json")
    previous_manifest = read(prior / "manifest.json")
    run, arm = ADAPTERS["target"]
    adapter = ROOT / "artifacts" / run / arm
    adapter_hash = digest(adapter / "adapter_model.safetensors")
    assert adapter_hash == previous_manifest["local_adapter_file_sha256"] and REVISION == previous_manifest["base_revision"]
    assert len(data["rows"]) == 2 and set(data["prompts"]) == {"original", "field-independence"}
    sources = ["venue_location_cache_check.py", "time_cue_order_diagnostic.py", "time_regression.py",
               "train_smoke.py", "audit_evaluation.py", "contract.py", "prompt_diagnostic.py"]
    sources += [f"results/{PRIOR}/{name}" for name in ("cases.json", "review.json", "manifest.json")]
    manifest = {"source_sha256": {name: digest(ROOT / name) for name in sources},
                "base_revision": REVISION, "local_adapter_file_sha256": adapter_hash,
                "published_reference_adapter_sha256": previous_manifest["published_reference_adapter_sha256"],
                "generation": {"do_sample": False, "max_new_tokens": 128, "enable_thinking": False,
                               "use_cache": [True, False]},
                "gpu": torch.cuda.get_device_name(0), "cuda": torch.version.cuda,
                "packages": {name: version(name) for name in ("torch", "transformers", "peft")}}
    output = ROOT / "results" / args.run
    output.mkdir(parents=True, exist_ok=False)
    save_json(output / "cases.json", data)
    save_json(output / "manifest.json", manifest)

    tokenizer = AutoTokenizer.from_pretrained(SNAPSHOT, local_files_only=True)
    model = PeftModel.from_pretrained(load_base(), adapter, local_files_only=True, is_trainable=False)
    model.eval()
    assert not any(p.requires_grad for p in model.parameters())
    before = state_hash(model)
    assert before == previous["parameters_after"]
    pairs = []
    for prompt_name, prompt in data["prompts"].items():
        references = {r["id"]: r for r in previous["results"][prompt_name]}
        for row in data["rows"]:
            reference = references[row["id"]]
            cached, uncached = [generate_with_scores(model, tokenizer, prompt, row, reference, use_cache)
                                for use_cache in (True, False)]
            pairs.append({"prompt": prompt_name, "id": row["id"], "time_state": row["time_state"],
                          "value_token_index_zero_based": reference["value_token_index_zero_based"],
                          "shared_output_prefix": reference["shared_output_prefix"],
                          "cached": cached, "uncached": uncached})
            print(prompt_name, row["time_state"], cached["raw"], uncached["raw"], flush=True)
    after = state_hash(model)
    assert before == after
    save_json(output / "review.json", {"training_steps": 0, "human_review": "pending", "promote_to_main": False,
              "weights_unchanged": True, "parameters_before": before, "parameters_after": after,
              "generation_calls": 2 * len(pairs), "known_token_replays": sum(p["cached"]["replay_equal"] for p in pairs),
              "source_groups": 1, "new_source_groups": 0, "pairs": pairs})


if __name__ == "__main__":
    main()
