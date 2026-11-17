"""Trace the location value token for the remaining blue-printing venue omission."""
import argparse
from importlib.metadata import version
from pathlib import Path

from fit_metrics import first_difference
from prompt_diagnostic import ROOT, read
from time_cue_order_diagnostic import ADAPTERS

PRIOR = "2026-11-16-venue-field-prompt-v1"
GROUP = "voc-01"


def trace(model, tokenizer, data, row, reference, null_token):
    import torch

    from evaluate_fit import teacher_forced
    from train_smoke import fixture

    case = fixture(tokenizer, data, row)
    prefix_length = case["prefix_tokens"]
    gold = case["input_ids"][prefix_length:]
    position = next(i for i in range(len(gold)) if tokenizer.decode(gold[:i]).endswith(',"location'))
    assert tokenizer.decode([gold[position]]) == '":"'
    ids = torch.tensor([case["input_ids"][:prefix_length]], device="cuda")
    with torch.inference_mode():
        result = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids), do_sample=False,
                                max_new_tokens=128, use_cache=True, pad_token_id=tokenizer.pad_token_id,
                                eos_token_id=tokenizer.eos_token_id, return_dict_in_generate=True,
                                output_scores=True)
    generated = result.sequences[0, prefix_length:].cpu().tolist()
    raw = tokenizer.decode(generated, skip_special_tokens=True)
    assert generated == reference["token_ids"] and raw == reference["raw"]
    assert generated[:position] == gold[:position]
    difference = first_difference(gold, generated)
    assert difference is None or difference == position
    assert generated[position] in (gold[position], null_token)
    cached = torch.log_softmax(result.scores[position][0].float(), dim=-1)
    teacher = teacher_forced(model, tokenizer, case, reference)
    token = teacher["tokens"][position]
    assert teacher["first_difference_index"] == difference and token["target_id"] == gold[position]
    return {"id": row["id"], "time_state": row["time_state"], "raw": raw, "generated_token_ids": generated,
            "replay_equal": True, "value_token_index_zero_based": position,
            "first_difference_index_zero_based": difference, "shared_output_prefix": tokenizer.decode(gold[:position]),
            "gold_token_id": gold[position], "bare_colon_token_id": null_token, "generated_token_id": generated[position],
            "cached_p": {"gold": cached[gold[position]].exp().item(), "bare_colon": cached[null_token].exp().item()},
            "teacher_forced": teacher}


def main():
    import torch
    from peft import PeftModel
    from transformers import AutoTokenizer

    from audit_evaluation import state_hash
    from train_smoke import REVISION, SNAPSHOT, digest, load_base, save_json

    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    args = parser.parse_args()
    if Path(args.run).name != args.run or args.run in (".", ".."):
        raise ValueError("new run directory required")
    prior = ROOT / "results" / PRIOR
    frozen = read(prior / "cases.json")
    rows = [row for row in frozen["rows"] if row["group"] == GROUP]
    assert {row["time_state"] for row in rows} == {"undetermined", "explicit"} and len(rows) == 2
    prompts = read(prior / "protocol.json")["prompts"]
    assert set(prompts) == {"original", "field-independence"}
    start = read(prior / "manifest.json")
    adapter_run, arm = ADAPTERS["target"]
    adapter = ROOT / "artifacts" / adapter_run / arm
    adapter_hash = digest(adapter / "adapter_model.safetensors")
    assert adapter_hash == start["local_adapter_file_sha256"] and REVISION == start["base_revision"]
    sources = ["venue_location_token_trace.py", "time_cue_order_diagnostic.py", "time_regression.py", "train_smoke.py",
               "audit_evaluation.py", "evaluate_fit.py", "fit_metrics.py", "contract.py", "prompt_diagnostic.py"]
    sources += [f"results/{PRIOR}/{name}" for name in ("cases.json", "protocol.json", "manifest.json", "review.json",
                                                       "original-generation.json", "field-independence-generation.json")]
    manifest = {"source_sha256": {name: digest(ROOT / name) for name in sources}, "base_revision": REVISION,
                "local_adapter_file_sha256": adapter_hash, "generation": start["generation"],
                "published_reference_adapter_sha256": start["published_reference_adapter_sha256"],
                "gpu": torch.cuda.get_device_name(0), "cuda": torch.version.cuda,
                "packages": {name: version(name) for name in ("torch", "transformers", "peft")}}
    output = ROOT / "results" / args.run
    output.mkdir(parents=True, exist_ok=False)
    save_json(output / "manifest.json", manifest)
    save_json(output / "cases.json", {"rows": rows, "prompts": prompts})

    tokenizer = AutoTokenizer.from_pretrained(SNAPSHOT, local_files_only=True)
    bare_colon = tokenizer.encode('":', add_special_tokens=False)
    assert len(bare_colon) == 1 and tokenizer.decode(bare_colon) == '":'
    model = PeftModel.from_pretrained(load_base(), adapter, local_files_only=True, is_trainable=False)
    model.eval()
    assert not any(p.requires_grad for p in model.parameters())
    before = state_hash(model)
    assert before == read(prior / "review.json")["parameters_before"]
    results = {}
    for name, prompt in prompts.items():
        references = {r["id"]: r for r in read(prior / f"{name}-generation.json")}
        results[name] = [trace(model, tokenizer, {"instruction": prompt}, row, references[row["id"]], bare_colon[0])
                         for row in rows]
        print(name, [(r["time_state"], r["cached_p"]) for r in results[name]], flush=True)
    after = state_hash(model)
    assert before == after
    calls = sum(len(records) for records in results.values())
    save_json(output / "review.json", {"training_steps": 0, "human_review": "pending", "promote_to_main": False,
              "weights_unchanged": True, "parameters_before": before, "parameters_after": after,
              "source_groups": 1, "new_source_groups": 0, "generation_calls": calls, "teacher_forced_forwards": calls,
              "known_token_replays": sum(r["replay_equal"] for records in results.values() for r in records),
              "results": results})


if __name__ == "__main__":
    main()
