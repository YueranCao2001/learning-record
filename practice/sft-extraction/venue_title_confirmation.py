"""Replicate a venue omission with the same title template and new activity content."""
import argparse
import gc
from pathlib import Path

from confirm_prompt import aggregate
from contract import validate_target
from prompt_diagnostic import ROOT, read
from time_cue_order_diagnostic import ADAPTERS, SOURCE_RUN

SPEC = "data/venue-title-confirmation-v1.json"
ANCHOR_SPEC = "data/venue-omission-confirmation-v1.json"
STATES = ("explicit-time-first", "undetermined-time-first", "removed-venue-only")


def prepare(spec, anchor_spec, prior):
    source = next(r for r in anchor_spec["sources"] if r["id"] == spec["anchor_group"])
    lookup = {r["id"]: r for r in prior["rows"]}
    anchors = [lookup[f"{source['id']}-{state}"] for state in STATES]
    rows = []
    for new in (source, *spec["sources"]):
        venue_clause = f"{source['venue_cue']}：{new['venue']}"
        for state, anchor in zip(STATES, anchors):
            explicit = state == "explicit-time-first"
            time = new["time"] if explicit else anchor_spec["missing_status"]
            fields = [venue_clause]
            if state != "removed-venue-only":
                fields.insert(0, f"{source['time_cue']}：{time}")
            notice = source["template"].format(event=new["event"], fields=source["separator"].join(fields))
            expected = {"event": new["event"], "time": time if explicit else None, "location": new["venue"]}
            validate_target(expected)
            assert all(value is None or value in notice for value in expected.values())
            if new is source:
                assert anchor["text"] == notice and anchor["expected"] == expected
                rows.append(anchor)
            else:
                rows.append({**anchor, "id": f"{new['id']}-{state}", "group": new["id"], "text": notice,
                             "cohort": "title-template-confirmation", "expected": expected})
    assert len({r["id"] for r in rows}) == len({r["text"] for r in rows}) == 9
    assert all(r["split"] == "dev" and r["cohort"] for r in rows)
    for key in ("id", "group", "text"):
        assert not {r[key] for r in rows[3:]} & {r[key] for r in prior["rows"]}
    return {"instruction": prior["instruction"], "rows": rows}


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
    prior = ROOT / "results" / SOURCE_RUN
    spec = read(ROOT / SPEC)
    data = prepare(spec, read(ROOT / ANCHOR_SPEC), read(prior / "cases.json"))
    previous = read(prior / "manifest.json")
    paths = {name: ROOT / "artifacts" / run / arm for name, (run, arm) in ADAPTERS.items()}
    hashes = {name: digest(path / "adapter_model.safetensors") for name, path in paths.items()}
    assert hashes == previous["adapter_files_sha256"]
    sources = [SPEC, ANCHOR_SPEC, "venue_title_confirmation.py", "time_cue_order_diagnostic.py", "time_regression.py",
               "train_smoke.py", "contract.py", "confirm_prompt.py", "prompt_diagnostic.py", "audit_evaluation.py"]
    sources += [f"results/{SOURCE_RUN}/{name}" for name in ("cases.json", "manifest.json", "environment.json")]
    sources += [f"results/{SOURCE_RUN}/{c}-generation.json" for c in ADAPTERS]
    manifest = {"source_sha256": {name: digest(ROOT / name) for name in sources},
                "base_revision": previous["base_revision"], "adapter_files_sha256": hashes,
                "published_reference_adapter_sha256": previous["published_reference_adapter_sha256"],
                "generation": previous["generation"]}
    output = ROOT / "results" / args.run
    output.mkdir(parents=True, exist_ok=False)
    for name, value in (("cases", data), ("protocol", spec), ("manifest", manifest)):
        save_json(output / f"{name}.json", value)

    tokenizer = AutoTokenizer.from_pretrained(SNAPSHOT, local_files_only=True)
    results = {}
    for condition, path in paths.items():
        model = PeftModel.from_pretrained(load_base(), path, local_files_only=True, is_trainable=False)
        model.eval()
        assert not any(p.requires_grad for p in model.parameters())
        before = state_hash(model)
        generated = generate(model, tokenizer, data)
        _, details, errors = aggregate(data["rows"], generated)
        new_summary, _, _ = aggregate(data["rows"][3:], generated[3:])
        after = state_hash(model)
        assert before == after
        history = {r["id"]: r["token_ids"] for r in read(prior / f"{condition}-generation.json")}
        replays = {o["id"]: o["token_ids"] == history[o["id"]] for o in generated[:3]}
        assert len(replays) == 3 and all(replays.values())
        save_json(output / f"{condition}-generation.json", generated)
        results[condition] = {"new_summary": new_summary, "details": details, "errors": errors,
                              "token_replays": replays, "generation_calls": len(generated),
                              "parameters_before": before, "parameters_after": after}
        print(condition, new_summary, flush=True)
        del model
        gc.collect()
    save_json(output / "review.json", {"training_steps": 0, "human_review": "pending", "promote_to_main": False,
              "weights_unchanged": True, "new_source_groups": len(spec["sources"]), "new_cases": len(data["rows"]) - 3,
              "generation_calls": sum(c["generation_calls"] for c in results.values()),
              "known_token_replays": sum(len(c["token_replays"]) for c in results.values()), "results": results})


if __name__ == "__main__":
    main()
