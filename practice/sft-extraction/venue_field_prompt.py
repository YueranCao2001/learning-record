"""Compare one field-independence sentence against the original extraction prompt."""
import argparse
from importlib.metadata import version
from pathlib import Path

from confirm_prompt import aggregate
from prompt_diagnostic import ROOT, read
from time_cue_order_diagnostic import ADAPTERS

ORDER_RUN = "2026-11-16-venue-clause-order-v1"
TITLE_RUN = "2026-11-12-venue-title-confirmation-v1"
RULE = "逐个判断字段，只把原文未明确给出的字段填为 null。"


def prepare(order_data, title_data):
    anchors = [r for r in order_data["rows"] if r["order"] == "venue-first" and r["wording"] == "后续发布"]
    assert len(anchors) == len({r["group"] for r in anchors}) == 3
    assert order_data["instruction"] == title_data["instruction"]
    explicit = {r["group"]: r for r in title_data["rows"] if r["time_state"] == "explicit"}
    rows = []
    for anchor in anchors:
        control = explicit[anchor["group"]]
        time = control["expected"]["time"]
        assert time and anchor["expected"] == {**control["expected"], "time": None}
        clause = "｜时间：后续发布\n"
        assert anchor["text"].count(clause) == 1
        rows.append(anchor)
        rows.append({**control, "id": control["id"] + "-venue-first", "order": "venue-first",
                     "text": anchor["text"].replace(clause, f"｜时间：{time}\n", 1),
                     "paired_undetermined_id": anchor["id"]})
    assert len({r["id"] for r in rows}) == len({r["text"] for r in rows}) == 6
    assert all(r["split"] == "dev" and r["cohort"] for r in rows)
    return {"instruction": order_data["instruction"], "rows": rows}


def main():
    import torch
    from peft import PeftModel
    from transformers import AutoTokenizer

    from audit_evaluation import state_hash
    from train_smoke import REVISION, SNAPSHOT, digest, generate, load_base, save_json

    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    args = parser.parse_args()
    if Path(args.run).name != args.run or args.run in (".", ".."):
        raise ValueError("new run directory required")
    prior = ROOT / "results" / ORDER_RUN
    data = prepare(read(prior / "cases.json"), read(ROOT / "results" / TITLE_RUN / "cases.json"))
    previous = read(prior / "manifest.json")
    run, arm = ADAPTERS["target"]
    adapter = ROOT / "artifacts" / run / arm
    adapter_hash = digest(adapter / "adapter_model.safetensors")
    assert adapter_hash == previous["adapter_files_sha256"]["target"] and REVISION == previous["base_revision"]
    sources = ["venue_field_prompt.py", "time_cue_order_diagnostic.py", "time_regression.py", "train_smoke.py",
               "confirm_prompt.py", "prompt_diagnostic.py", "audit_evaluation.py", "contract.py",
               f"results/{TITLE_RUN}/cases.json"]
    sources += [f"results/{ORDER_RUN}/{name}" for name in ("cases.json", "manifest.json", "target-generation.json")]
    prompts = {"original": data["instruction"], "field-independence": data["instruction"] + "\n" + RULE}
    manifest = {"source_sha256": {n: digest(ROOT / n) for n in sources}, "base_revision": REVISION,
                "local_adapter_file_sha256": adapter_hash, "generation": previous["generation"],
                "published_reference_adapter_sha256": previous["published_reference_adapter_sha256"],
                "gpu": torch.cuda.get_device_name(0), "cuda": torch.version.cuda,
                "packages": {n: version(n) for n in ("torch", "transformers", "peft")}}
    protocol = {"date": "2026-11-16", "prompts": prompts, "source_groups": 3, "new_source_groups": 0,
                "question": "Does appending one field-independence sentence restore the three omitted venues?",
                "completion": "Score both prompts on all six notices, replay three original outputs, and compare location and time by pair.",
                "human_review": "pending"}
    output = ROOT / "results" / args.run
    output.mkdir(parents=True, exist_ok=False)
    for name, value in (("cases", data), ("protocol", protocol), ("manifest", manifest)):
        save_json(output / f"{name}.json", value)

    tokenizer = AutoTokenizer.from_pretrained(SNAPSHOT, local_files_only=True)
    model = PeftModel.from_pretrained(load_base(), adapter, local_files_only=True, is_trainable=False)
    model.eval()
    assert not any(p.requires_grad for p in model.parameters())
    before = state_hash(model)
    results, outputs = {}, {}
    for name, prompt in prompts.items():
        generated = generate(model, tokenizer, {**data, "instruction": prompt})
        summary, records, errors = aggregate(data["rows"], generated)
        save_json(output / f"{name}-generation.json", generated)
        results[name] = {"summary": summary, "records": records, "errors": errors}
        outputs[name] = generated
        print(name, summary, flush=True)
    after = state_hash(model)
    assert before == after

    history = {r["id"]: r["token_ids"] for r in read(prior / "target-generation.json")}
    replays = {row["id"]: generated["token_ids"] == history[row["id"]]
               for row, generated in zip(data["rows"], outputs["original"]) if row["time_state"] == "undetermined"}
    assert len(replays) == 3 and all(replays.values())
    pairs = [{"id": row["id"], "time_state": row["time_state"], "original": original, "field_independence": variant}
             for row, original, variant in zip(data["rows"], results["original"]["records"],
                                               results["field-independence"]["records"])]
    save_json(output / "review.json", {"training_steps": 0, "human_review": "pending", "promote_to_main": False,
              "weights_unchanged": True, "parameters_before": before, "parameters_after": after,
              "source_groups": 3, "new_source_groups": 0, "generation_calls": sum(map(len, outputs.values())),
              "token_replays": replays, "known_token_replays": len(replays), "pairs": pairs, "results": results})


if __name__ == "__main__":
    main()
