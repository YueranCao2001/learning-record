"""Compare both clause orders for the same nine undetermined-time notices."""
import argparse
import gc
from pathlib import Path

from confirm_prompt import aggregate
from prompt_diagnostic import ROOT, read
from time_cue_order_diagnostic import ADAPTERS
from venue_time_wording import WORDINGS

PRIOR = "2026-11-13-venue-time-wording-v1"


def prepare(prior):
    controls = prior["rows"]
    assert len(controls) == 9
    assert len({(row["group"], row["wording"]) for row in controls}) == 9
    variants = []
    for row in controls:
        heading, line, closing = row["text"].splitlines()
        time_clause, venue_clause = line.split("｜")
        assert row["order"] == "time-first" and row["split"] == "dev" and row["cohort"]
        assert row["expected"]["time"] is None and row["expected"]["location"]
        assert time_clause == f"时间：{row['wording']}" and row["wording"] in WORDINGS
        assert venue_clause == f"地点：{row['expected']['location']}"
        variants.append({**row, "id": row["id"] + "-venue-first", "order": "venue-first",
                         "text": "\n".join((heading, venue_clause + "｜" + time_clause, closing)),
                         "replay_id": None, "paired_control_id": row["id"]})
    rows = controls + variants
    assert len({r["id"] for r in rows}) == len({r["text"] for r in rows}) == 18
    return {"instruction": prior["instruction"], "rows": rows}


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
    sources = ["venue_clause_order.py", "venue_time_wording.py", "time_cue_order_diagnostic.py", "time_regression.py",
               "train_smoke.py", "confirm_prompt.py", "prompt_diagnostic.py", "audit_evaluation.py", "contract.py"]
    sources += [f"results/{PRIOR}/{name}" for name in ("cases.json", "manifest.json", *(f"{c}-generation.json" for c in ADAPTERS))]
    manifest = {"source_sha256": {n: digest(ROOT / n) for n in sources}, "base_revision": REVISION,
                "adapter_files_sha256": hashes, "generation": previous["generation"],
                "published_reference_adapter_sha256": previous["published_reference_adapter_sha256"],
                "gpu": torch.cuda.get_device_name(0), "cuda": torch.version.cuda,
                "packages": {n: version(n) for n in ("torch", "transformers", "peft")}}
    protocol = {"date": "2026-11-16", "source_run": PRIOR, "wordings": WORDINGS, "source_groups": 3,
                "new_source_groups": 0, "question": "Does placing the venue clause first restore the explicit venue?",
                "completion": "Score all 27 paired comparisons, verify 27 original token replays and report time and venue changes.",
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
        _, details, errors = aggregate(data["rows"], generated)
        after = state_hash(model)
        assert before == after
        old = {r["id"]: r["token_ids"] for r in read(prior / f"{condition}-generation.json")}
        replays = {o["id"]: o["token_ids"] == old[o["id"]] for o in generated[:9]}
        assert len(replays) == 9 and all(replays.values())
        pairs = [{"group": row["group"], "wording": row["wording"], "time_first": control, "venue_first": variant}
                 for row, control, variant in zip(data["rows"][:9], details[:9], details[9:])]
        by_wording = {}
        for wording in WORDINGS:
            selected = [p for p in pairs if p["wording"] == wording]
            by_wording[wording] = {
                order: {field: sum(p[order]["fields"][field] for p in selected) for field in ("event", "time", "location")}
                for order in ("time_first", "venue_first")
            }
        save_json(output / f"{condition}-generation.json", generated)
        results[condition] = {"by_wording": by_wording, "pairs": pairs, "errors": errors, "token_replays": replays,
                              "generation_calls": len(generated), "parameters_before": before, "parameters_after": after}
        print(condition, by_wording, flush=True)
        del model
        gc.collect()
    save_json(output / "review.json", {"training_steps": 0, "human_review": "pending", "promote_to_main": False,
              "weights_unchanged": True, "source_groups": 3, "new_source_groups": 0,
              "generation_calls": sum(c["generation_calls"] for c in results.values()),
              "known_token_replays": sum(len(c["token_replays"]) for c in results.values()), "results": results})


if __name__ == "__main__":
    main()
