"""Compare three undetermined-time phrases in frozen title notices."""
import argparse
import gc
from pathlib import Path

from confirm_prompt import aggregate
from prompt_diagnostic import ROOT, read
from time_cue_order_diagnostic import ADAPTERS

PRIOR = "2026-11-12-venue-title-confirmation-v1"
WORDINGS = ("后续发布", "待定", "尚未确定")


def prepare(previous):
    anchors = [r for r in previous["rows"] if r["time_state"] == "undetermined"]
    assert len(anchors) == len({r["group"] for r in anchors}) == 3
    rows = []
    for anchor in anchors:
        assert anchor["split"] == "dev" and anchor["order"] == "time-first"
        assert anchor["expected"]["time"] is None and anchor["expected"]["location"]
        clause = f"时间：{WORDINGS[0]}｜"
        assert anchor["text"].count(clause) == 1
        for index, wording in enumerate(WORDINGS):
            notice = anchor["text"].replace(clause, f"时间：{wording}｜", 1)
            rows.append({**anchor, "id": f"{anchor['group']}-wording-{index}", "text": notice,
                         "wording": wording, "replay_id": anchor["id"] if index == 0 else None})
    assert len({r["id"] for r in rows}) == len({r["text"] for r in rows}) == 9
    return {"instruction": previous["instruction"], "rows": rows}


def main():
    from importlib.metadata import version

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
    prior = ROOT / "results" / PRIOR
    data = prepare(read(prior / "cases.json"))
    previous = read(prior / "manifest.json")
    paths = {name: ROOT / "artifacts" / run / arm for name, (run, arm) in ADAPTERS.items()}
    hashes = {name: digest(path / "adapter_model.safetensors") for name, path in paths.items()}
    assert hashes == previous["adapter_files_sha256"] and REVISION == previous["base_revision"]
    sources = ["venue_time_wording.py", "time_cue_order_diagnostic.py", "time_regression.py", "train_smoke.py",
               "confirm_prompt.py", "prompt_diagnostic.py", "audit_evaluation.py", "contract.py"]
    sources += [f"results/{PRIOR}/{name}" for name in ("cases.json", "manifest.json", *(f"{c}-generation.json" for c in ADAPTERS))]
    manifest = {"source_sha256": {n: digest(ROOT / n) for n in sources}, "base_revision": REVISION,
                "adapter_files_sha256": hashes, "generation": previous["generation"],
                "published_reference_adapter_sha256": previous["published_reference_adapter_sha256"],
                "gpu": torch.cuda.get_device_name(0), "cuda": torch.version.cuda,
                "packages": {n: version(n) for n in ("torch", "transformers", "peft")}}
    protocol = {"date": "2026-11-13", "wordings": WORDINGS, "source_groups": 3, "new_source_groups": 0,
                "question": "Does replacing the undetermined time phrase restore the explicit venue?",
                "completion": "Score all nine inputs for each of three adapters and reproduce all nine frozen outputs.",
                "human_review": "pending"}
    output = ROOT / "results" / args.run
    output.mkdir(parents=True, exist_ok=False)
    for name, value in (("cases", data), ("protocol", protocol), ("manifest", manifest)):
        save_json(output / f"{name}.json", value)

    tokenizer = AutoTokenizer.from_pretrained(SNAPSHOT, local_files_only=True)
    results = {}
    for condition, path in paths.items():
        model = PeftModel.from_pretrained(load_base(), path, local_files_only=True, is_trainable=False)
        model.eval()
        assert not any(p.requires_grad for p in model.parameters())
        before = state_hash(model)
        generated = generate(model, tokenizer, data)
        summary, details, errors = aggregate(data["rows"], generated)
        after = state_hash(model)
        assert before == after
        old = {r["id"]: r["token_ids"] for r in read(prior / f"{condition}-generation.json")}
        replays = {r["id"]: o["token_ids"] == old[r["replay_id"]]
                   for r, o in zip(data["rows"], generated) if r["replay_id"]}
        assert len(replays) == 3 and all(replays.values())
        by_wording = {}
        for wording in WORDINGS:
            indexes = [i for i, r in enumerate(data["rows"]) if r["wording"] == wording]
            by_wording[wording] = {field: sum(details[i]["fields"][field] for i in indexes)
                                   for field in ("event", "time", "location")}
        save_json(output / f"{condition}-generation.json", generated)
        results[condition] = {"summary": summary, "by_wording": by_wording, "details": details, "errors": errors,
                              "token_replays": replays, "parameters_before": before, "parameters_after": after}
        print(condition, by_wording, flush=True)
        del model
        gc.collect()
    save_json(output / "review.json", {"training_steps": 0, "human_review": "pending", "promote_to_main": False,
              "weights_unchanged": True, "source_groups": 3, "new_source_groups": 0,
              "generation_calls": sum(c["summary"]["count"] for c in results.values()),
              "known_token_replays": sum(len(c["token_replays"]) for c in results.values()), "results": results})


if __name__ == "__main__":
    main()
