"""Validate synthetic preference labels and shared Qwen chat prefixes."""
import argparse
from collections import Counter
from importlib.metadata import version
import json
from pathlib import Path

from contract import score, validate_target
from fit_metrics import first_difference
from prompt_diagnostic import ROOT, read

DATA = "data/dpo-preference-pairs-v1.json"
ANNOTATION = "annotation-rules-v1.md"
HF_REFERENCE = "results/2026-11-18-venue-stage-closeout-v1/manifest.json"


def validate(data):
    identities, texts, group_splits = set(), set(), {}
    for row in data["rows"]:
        identity, group, split = row["id"], row["group"], row["split"]
        if split not in {"train", "dev"} or not group or not identity or not row["text"].strip():
            raise ValueError("invalid row identity or split")
        if group_splits.setdefault(group, split) != split:
            raise ValueError("source group crosses splits")
        if identity in identities or row["text"] in texts:
            raise ValueError("duplicate pair ID or prompt")
        identities.add(identity)
        texts.add(row["text"])
        expected, chosen, rejected = (row[key] for key in ("expected", "chosen", "rejected"))
        for answer in (expected, chosen, rejected):
            validate_target(answer)
            if any(value is not None and value not in row["text"] for value in answer.values()):
                raise ValueError("answer contains a nonliteral value")
        if not score(json.dumps(chosen), expected)["exact"]:
            raise ValueError("chosen must match the reference")
        changed = {key for key in chosen if chosen[key] != rejected[key]}
        if changed != {row["field"]}:
            raise ValueError("preference pair must change exactly the documented field")
        key = row["field"]
        error = "missing-to-value" if chosen[key] is None else "value-to-null" if rejected[key] is None else "wrong-span"
        if row["error"] != error or not row["reason"].strip():
            raise ValueError("preference reason or error label missing")


def main():
    from transformers import AutoTokenizer
    from train_smoke import REVISION, SNAPSHOT, digest, fixture, save_json

    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    args = parser.parse_args()
    if Path(args.run).name != args.run or args.run in (".", ".."):
        raise ValueError("new run directory required")
    data = read(ROOT / DATA)
    validate(data)
    tokenizer = AutoTokenizer.from_pretrained(SNAPSHOT, local_files_only=True)
    records = []
    for row in data["rows"]:
        fixtures = {name: fixture(tokenizer, data, {**row, "expected": row[name]})
                    for name in ("chosen", "rejected")}
        chosen, rejected = fixtures["chosen"], fixtures["rejected"]
        prefix_length = chosen["prefix_tokens"]
        prefix = chosen["input_ids"][:prefix_length]
        assert prefix_length == rejected["prefix_tokens"] and prefix == rejected["input_ids"][:prefix_length]
        answers = {name: item["input_ids"][prefix_length:] for name, item in fixtures.items()}
        first = first_difference(answers["chosen"], answers["rejected"])
        assert first is not None and first < min(map(len, answers.values()))
        assert all(token_ids[-1] == tokenizer.eos_token_id for token_ids in answers.values())
        records.append({"id": row["id"], "group": row["group"], "split": row["split"],
                        "field": row["field"], "error": row["error"], "reason": row["reason"],
                        "shared_prompt_token_ids": prefix, "completion_token_ids": answers,
                        "prefix_tokens": prefix_length, "completion_tokens": {n: len(ids) for n, ids in answers.items()},
                        "first_difference_zero_based": first,
                        "first_difference_text": {n: tokenizer.decode([ids[first]]) for n, ids in answers.items()}})
    output = ROOT / "results" / args.run
    output.mkdir(parents=True, exist_ok=False)
    save_json(output / "pairs.json", data)
    save_json(output / "manifest.json", {
        "source_sha256": {n: digest(ROOT / n) for n in (DATA, ANNOTATION, HF_REFERENCE, "dpo_pair_preflight.py",
                                                      "train_smoke.py", "contract.py", "fit_metrics.py", "prompt_diagnostic.py")},
        "tokenizer_revision": REVISION, "transformers": version("transformers"),
        "tokenizer_files_sha256": {n: digest(SNAPSHOT / n) for n in ("tokenizer.json", "tokenizer_config.json")},
        "enable_thinking": False, "eos_id": tokenizer.eos_token_id, "max_sequence_tokens": 512,
        "published_reference_adapter_sha256": read(ROOT / HF_REFERENCE)["published_reference_adapter_sha256"]})
    save_json(output / "review.json", {
        "training_steps": 0, "human_review": "pending", "model_generations": 0, "pairs": len(records),
        "split_counts": dict(Counter(row["split"] for row in records)),
        "error_counts": dict(Counter(row["error"] for row in records)),
        "shared_prefix_pairs": len(records), "distinct_completion_pairs": len(records), "token_records": records})
    for row in records:
        print(row["id"], row["prefix_tokens"], row["completion_tokens"], row["first_difference_text"])


if __name__ == "__main__":
    main()
