"""Close the venue diagnosis with a precision probe and a fixed open regression panel."""
import argparse
import gc
from importlib.metadata import version
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from audit_evaluation import state_hash
from confirm_prompt import aggregate
from contract import score
from fit_metrics import first_difference
from prompt_diagnostic import ROOT, read
from time_cue_order_diagnostic import ADAPTERS
from train_smoke import REVISION, SNAPSHOT, digest, fixture, generate, load_base, save_json

PROMPT_RUN = "2026-11-16-venue-field-prompt-v1"
CACHE_RUN = "2026-11-18-venue-cache-scope-v1"
PROBE = "vtc-02-wording-0-venue-first"
HISTORY = {
    "2026-11-06-time-template-confirmation-v1": (
        "ttc-01-cue0-missing-location", "ttc-03-cue0-missing-time", "ttc-03-cue1-missing-time"),
    "2026-11-05-time-regression-v1": ("confirm-02-missing-location", "time-regression-time-first"),
}


def precision_probe(model, tokenizer, prompt, row):
    case = fixture(tokenizer, {"instruction": prompt}, row)
    prefix_length = case["prefix_tokens"]
    gold = case["input_ids"][prefix_length:]
    index = next(i for i in range(len(gold)) if tokenizer.decode(gold[:i]).endswith(',"location'))
    string_id = gold[index]
    colon_ids = tokenizer.encode('":', add_special_tokens=False)
    assert tokenizer.decode([string_id]) == '":"' and len(colon_ids) == 1
    ids = torch.tensor([case["input_ids"][:prefix_length]], device="cuda")
    paths = {}
    for name, use_cache in (("cached", True), ("uncached", False)):
        with torch.inference_mode():
            output = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids), do_sample=False,
                                    max_new_tokens=128, use_cache=use_cache, pad_token_id=tokenizer.pad_token_id,
                                    eos_token_id=tokenizer.eos_token_id, return_dict_in_generate=True,
                                    output_scores=True)
        tokens = output.sequences[0, prefix_length:].cpu().tolist()
        raw = tokenizer.decode(tokens, skip_special_tokens=True)
        matching_prefix = len(tokens) > index and tokens[:index] == gold[:index]
        candidates = None
        if matching_prefix:
            scores = output.scores[index][0].float()
            probabilities = torch.softmax(scores, dim=-1)
            colon_id = colon_ids[0]
            candidates = {"chosen_id": tokens[index], "string_id": string_id, "colon_id": colon_id,
                          "string_p": probabilities[string_id].item(), "colon_p": probabilities[colon_id].item(),
                          "string_minus_colon_logit": (scores[string_id] - scores[colon_id]).item()}
        paths[name] = {"raw": raw, "token_ids": tokens, "score": score(raw, row["expected"]),
                       "first_gold_difference_zero_based": first_difference(gold, tokens),
                       "gold_prefix_reached": matching_prefix, "candidates": candidates}
    return {"value_token_index_zero_based": index, "paths": paths}


def regression(model, tokenizer, prompts, rows, references):
    results = {}
    for name, prompt in prompts.items():
        outputs = generate(model, tokenizer, {"instruction": prompt, "rows": rows})
        summary, records, errors = aggregate(rows, outputs)
        replay = {out["id"]: out["token_ids"] == references[name][out["id"]]
                  for out in outputs if out["id"] in references[name]}
        assert len(replay) == len(references[name]) and all(replay.values())
        results[name] = {"summary": summary, "records": records, "errors": errors, "outputs": outputs,
                         "known_token_replays": replay}
        print(name, summary, flush=True)
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    args = parser.parse_args()
    if Path(args.run).name != args.run or args.run in (".", ".."):
        raise ValueError("new run directory required")
    prompt_dir = ROOT / "results" / PROMPT_RUN
    cache_dir = ROOT / "results" / CACHE_RUN
    rows = read(prompt_dir / "cases.json")["rows"]
    prompts = read(prompt_dir / "protocol.json")["prompts"]
    sources = ["venue_stage_closeout.py", "time_cue_order_diagnostic.py", "time_regression.py",
               "train_smoke.py", "audit_evaluation.py", "contract.py", "prompt_diagnostic.py", "confirm_prompt.py",
               "fit_metrics.py"]
    sources += [f"results/{PROMPT_RUN}/{name}" for name in ("cases.json", "protocol.json", *(f"{c}-generation.json" for c in prompts))]
    sources += [f"results/{CACHE_RUN}/{name}" for name in ("review.json", "manifest.json")]
    references = {name: {r["id"]: r["token_ids"] for r in read(prompt_dir / f"{name}-generation.json")}
                  for name in prompts}
    for run, identities in HISTORY.items():
        prior = ROOT / "results" / run
        lookup = {r["id"]: r for r in read(prior / "cases.json")["rows"]}
        rows += [lookup[identity] for identity in identities]
        old = {r["id"]: r["token_ids"] for r in read(prior / "target-generation.json")}
        references["original"].update({identity: old[identity] for identity in identities})
        sources += [f"results/{run}/{name}" for name in ("cases.json", "target-generation.json")]
    assert len(rows) == len({r["id"] for r in rows}) == 11
    probe = next(r for r in rows if r["id"] == PROBE)
    prior_cache = read(cache_dir / "review.json")
    prior_probe = next(p for p in prior_cache["results"]["field-independence"]["pairs"] if p["id"] == PROBE)
    prior_manifest = read(cache_dir / "manifest.json")
    run, arm = ADAPTERS["target"]
    adapter = ROOT / "artifacts" / run / arm
    adapter_hash = digest(adapter / "adapter_model.safetensors")
    assert adapter_hash == prior_manifest["local_adapter_file_sha256"] and REVISION == prior_manifest["base_revision"]
    output = ROOT / "results" / args.run
    output.mkdir(parents=True, exist_ok=False)
    save_json(output / "cases.json", {"rows": rows, "prompts": prompts, "precision_probe_id": PROBE})
    save_json(output / "manifest.json", {"source_sha256": {n: digest(ROOT / n) for n in sources},
              "base_revision": REVISION, "local_adapter_file_sha256": adapter_hash,
              "published_reference_adapter_sha256": prior_manifest["published_reference_adapter_sha256"],
              "generation": prior_manifest["generation"], "regression_base_dtype": "bfloat16",
              "regression_use_cache": True, "float32_matmul_precision": "highest", "attention": "sdpa",
              "gpu": torch.cuda.get_device_name(0), "cuda": torch.version.cuda,
              "packages": {n: version(n) for n in ("torch", "transformers", "peft")}})

    torch.set_float32_matmul_precision("highest")
    tokenizer = AutoTokenizer.from_pretrained(SNAPSHOT, local_files_only=True)
    model = PeftModel.from_pretrained(load_base(), adapter, local_files_only=True, is_trainable=False).eval()
    before = state_hash(model)
    assert before == prior_cache["parameters_after"] and not any(p.requires_grad for p in model.parameters())
    bf16 = precision_probe(model, tokenizer, prompts["field-independence"], probe)
    assert all(record["token_ids"] == prior_probe[name]["token_ids"] for name, record in bf16["paths"].items())
    results = regression(model, tokenizer, prompts, rows, references)
    after = state_hash(model)
    assert before == after
    bf16.update(parameters_before=before, parameters_after=after)
    del model
    gc.collect()
    torch.cuda.empty_cache()

    base = AutoModelForCausalLM.from_pretrained(SNAPSHOT, local_files_only=True, torch_dtype=torch.float32,
                                               attn_implementation="sdpa").to("cuda")
    model = PeftModel.from_pretrained(base, adapter, local_files_only=True, is_trainable=False).eval()
    before = state_hash(model)
    assert all(p.dtype == torch.float32 and not p.requires_grad for p in model.parameters())
    fp32 = precision_probe(model, tokenizer, prompts["field-independence"], probe)
    after = state_hash(model)
    assert before == after
    fp32.update(parameters_before=before, parameters_after=after)
    print("precision", {dtype: {name: (r["raw"], r["candidates"]) for name, r in data["paths"].items()}
                        for dtype, data in (("bfloat16", bf16), ("float32", fp32))}, flush=True)
    replay_count = 2 + sum(len(r["known_token_replays"]) for r in results.values())
    save_json(output / "review.json", {"date": "2026-11-18", "training_steps": 0, "human_review": "pending",
              "promote_to_main": False, "weights_unchanged": True, "regression_cases": len(rows),
              "generation_calls": 4 + sum(r["summary"]["count"] for r in results.values()), "known_token_replays": replay_count,
              "precision_probe": {"bfloat16": bf16, "float32": fp32}, "regression": results})


if __name__ == "__main__":
    main()
