"""Cross clause order and cue wording for a known explicit-time omission."""
import argparse
import gc
from pathlib import Path

from confirm_prompt import aggregate
from contract import validate_target
from prompt_diagnostic import ROOT, read
from time_regression import TRAINED, WARM

SOURCE_RUN = "2026-11-11-venue-omission-confirmation-v1"
ADAPTERS = {"warm-start": (WARM, "extended"), "canonical": (TRAINED, "canonical"), "target": (TRAINED, "target")}
TIME_CUES = ("日程", "时间")
VENUE_CUES = ("会址", "地点")
ORDERS = ("time-first", "venue-first")


def prepare(source):
    lookup = {row["id"]: row for row in source["rows"]}
    rows = []
    for order in ORDERS:
        original = lookup[f"voc-02-explicit-{order}"]
        validate_target(original["expected"])
        assert original["split"] == "dev" and original["cohort"]
        assert original["text"].count("日程：") == original["text"].count("会址：") == 1
        assert all(v and v in original["text"] for v in original["expected"].values())
        for time_index, time_cue in enumerate(TIME_CUES):
            for venue_index, venue_cue in enumerate(VENUE_CUES):
                notice = original["text"].replace("日程：", time_cue + "：").replace("会址：", venue_cue + "：")
                rows.append({**original, "id": f"{order}-time{time_index}-venue{venue_index}",
                             "time_cue": time_cue, "venue_cue": venue_cue, "text": notice,
                             "replay_id": original["id"] if notice == original["text"] else None})
    assert len({r["text"] for r in rows}) == 8
    assert all(row["expected"] == rows[0]["expected"] for row in rows)
    return {"instruction": source["instruction"], "rows": rows}


def main():
    from peft import PeftModel
    from transformers import AutoTokenizer
    from audit_evaluation import state_hash
    from train_smoke import SNAPSHOT, digest, generate, load_base, save_json

    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    args = parser.parse_args()
    if Path(args.run).name != args.run or args.run in (".", ".."):
        raise ValueError("new run directory required")
    source = ROOT / "results" / SOURCE_RUN
    data = prepare(read(source / "cases.json"))
    previous = read(source / "manifest.json")
    paths = {name: ROOT / "artifacts" / run / arm for name, (run, arm) in ADAPTERS.items()}
    hashes = {name: digest(path / "adapter_model.safetensors") for name, path in paths.items()}
    assert hashes == previous["adapter_files_sha256"]
    evidence = ["cases.json", "manifest.json", "environment.json"] + [f"{c}-generation.json" for c in ADAPTERS]
    sources = ["time_cue_order_diagnostic.py", "time_regression.py", "train_smoke.py", "contract.py",
               "confirm_prompt.py", "prompt_diagnostic.py", "audit_evaluation.py"]
    sources += [f"results/{SOURCE_RUN}/{name}" for name in evidence]
    manifest = {"source_sha256": {name: digest(ROOT / name) for name in sources},
                "base_revision": previous["base_revision"], "adapter_files_sha256": hashes,
                "published_reference_adapter_sha256": previous["published_reference_adapter_sha256"],
                "generation": previous["generation"]}
    protocol = {"question": "Which tested cue replacements restore the known explicit time after changing clause order?",
                "time_cues": TIME_CUES, "venue_cues": VENUE_CUES, "orders": ORDERS, "source_groups": 1,
                "completion": "Replay both frozen orders, score all eight cases on each fixed adapter, and record field-level outcomes.",
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
        prior_outputs = {r["id"]: r["token_ids"] for r in read(source / f"{condition}-generation.json")}
        replays = {r["id"]: o["token_ids"] == prior_outputs[r["replay_id"]]
                   for r, o in zip(data["rows"], generated) if r["replay_id"]}
        assert len(replays) == 2 and all(replays.values())
        save_json(output / f"{condition}-generation.json", generated)
        results[condition] = {"summary": summary, "details": details, "errors": errors, "token_replays": replays,
                              "parameters_before": before, "parameters_after": after}
        print(condition, summary, flush=True)
        del model
        gc.collect()
    save_json(output / "review.json", {"training_steps": 0, "human_review": "pending", "promote_to_main": False,
              "weights_unchanged": True, "source_groups": 1, "new_source_groups": 0,
              "generation_calls": sum(c["summary"]["count"] for c in results.values()),
              "known_token_replays": sum(len(c["token_replays"]) for c in results.values()), "results": results})


if __name__ == "__main__":
    main()
