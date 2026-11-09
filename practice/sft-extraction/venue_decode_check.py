"""Compare the first divergent token with and without incremental KV caching."""
import argparse
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoTokenizer

from audit_evaluation import state_hash
from time_regression import TRAINED
from train_smoke import ROOT, SNAPSHOT, digest, load_base, prompt_ids, save_json
from prompt_diagnostic import read

PROBES = ("venue-cue0-original-time-first", "venue-cue1-original-venue-first", "ttc-01-cue0-missing-location")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    args = parser.parse_args()
    if Path(args.run).name != args.run or args.run in (".", ".."):
        raise ValueError("run must be a directory name")
    run = ROOT / "results" / args.run
    output = run / "decode-check.json"
    if output.exists():
        raise FileExistsError(output)
    data = read(run / "cases.json")
    rows = {r["id"]: r for r in data["rows"]}
    generated = {r["id"]: r for r in read(run / "target-generation.json")}
    teachers = {r["id"]: r for r in read(run / "target-teacher.json")}
    adapter = ROOT / "artifacts" / TRAINED / "target"
    adapter_hash = digest(adapter / "adapter_model.safetensors")
    assert adapter_hash == read(run / "manifest.json")["adapter_files_sha256"]["target"]
    tokenizer = AutoTokenizer.from_pretrained(SNAPSHOT, local_files_only=True)
    model = PeftModel.from_pretrained(load_base(), adapter, local_files_only=True, is_trainable=False)
    model.eval()
    before = state_hash(model)
    prior = read(run / "review.json")["results"]["target"]
    assert before == prior["parameters_before"]
    probes = {}
    for identity in PROBES:
        ids = torch.tensor([prompt_ids(tokenizer, data, rows[identity])], device="cuda")
        teacher = teachers[identity]
        position = teacher["first_difference_index"]
        gold_id = teacher["tokens"][position]["target_id"]
        free_id = generated[identity]["token_ids"][position]
        variants = {}
        for cache in (True, False):
            with torch.inference_mode():
                result = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids),
                                        do_sample=False, max_new_tokens=128, use_cache=cache,
                                        pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id,
                                        return_dict_in_generate=True, output_scores=True)
            tokens = result.sequences[0, ids.shape[1]:].cpu().tolist()
            shared_prefix = tokens[:position] == generated[identity]["token_ids"][:position]
            assert shared_prefix and len(result.scores) > position
            scores = torch.log_softmax(result.scores[position][0].float(), dim=-1)
            if cache:
                assert tokens == generated[identity]["token_ids"]
            variants["cached" if cache else "uncached"] = {
                "gold_token_p": scores[gold_id].exp().item(), "cached_choice_p": scores[free_id].exp().item(),
                "chosen_id": tokens[position], "token_ids": tokens, "replay_equal": tokens == generated[identity]["token_ids"],
                "raw": tokenizer.decode(tokens, skip_special_tokens=True)}
        probes[identity] = {"position_zero_based": position, "gold_id": gold_id, "cached_choice_id": free_id,
                            "teacher_argmax_id": teacher["tokens"][position]["argmax_id"], "paths": variants}
    after = state_hash(model)
    assert before == after
    inputs = ["cases.json", "manifest.json", "review.json", "target-generation.json", "target-teacher.json"]
    save_json(output, {"source_sha256": digest(Path(__file__)), "inputs_sha256": {n: digest(run / n) for n in inputs},
                       "adapter_sha256": adapter_hash, "parameter_sha256_before": before, "parameter_sha256_after": after,
                       "generation_calls": len(PROBES) * 2, "probes": probes})


if __name__ == "__main__":
    main()
