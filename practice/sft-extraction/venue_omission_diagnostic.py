"""Measure venue omissions and locate token divergence under the correct prefix."""
import argparse
import gc
from importlib.metadata import version
from pathlib import Path

from confirm_prompt import aggregate
from contract import validate_target
from prompt_diagnostic import ROOT, read
from time_regression import TRAINED, WARM

SPEC = "data/venue-omission-diagnostic-v1.json"


def prepare(spec, source):
    lookup = {r["id"]: r for r in source["rows"]}
    rows = []
    for cue_index in (0, 1):
        prefix = f"{spec['anchor_group']}-cue{cue_index}"
        missing = lookup[f"{prefix}-missing-time"]
        complete = lookup[f"{prefix}-complete"]
        heading, fields, closing = missing["text"].splitlines()
        time_clause, venue_clause = fields.split("｜")
        assert time_clause == f"时间：{spec['original_status']}"
        assert venue_clause == f"{missing['cue']}：{missing['expected']['location']}"
        assert missing["expected"] == {**complete["expected"], "time": None}
        assert missing["expected"]["time"] is None
        expressions = {
            "original": time_clause,
            "alternate": f"时间：{spec['alternate_status']}",
            "explicit": f"时间：{complete['expected']['time']}",
        }
        for status, clause in expressions.items():
            base = complete if status == "explicit" else missing
            for order in ("time-first", "venue-first"):
                fields = (clause, venue_clause) if order == "time-first" else (venue_clause, clause)
                notice = "\n".join((heading, "｜".join(fields), closing))
                replay = base["id"] if order == "time-first" and status != "alternate" else None
                if replay:
                    assert notice == base["text"]
                rows.append({**base, "id": f"venue-cue{cue_index}-{status}-{order}", "text": notice,
                             "time_status": status, "order": order, "replay_id": replay})
        rows.append({**missing, "id": f"venue-cue{cue_index}-drop-time",
                     "text": "\n".join((heading, venue_clause, closing)), "time_status": "removed",
                     "order": "venue-only", "replay_id": None})
    reverse = lookup[spec["reverse_probe_id"]]
    rows.append({**reverse, "time_status": "reverse-probe", "order": "venue-first", "replay_id": reverse["id"]})
    assert len({r["id"] for r in rows}) == len({r["text"] for r in rows}) == 15
    for row in rows:
        validate_target(row["expected"])
        assert row["split"] == "dev" and row["cohort"]
        assert all(v is None or v in row["text"] for v in row["expected"].values())
    return {"instruction": source["instruction"], "rows": rows}


def main():
    import torch
    from peft import PeftModel
    from transformers import AutoTokenizer
    from audit_evaluation import state_hash
    from evaluate_fit import teacher_forced
    from train_smoke import REVISION, SNAPSHOT, digest, fixture, generate, load_base, save_json

    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    args = parser.parse_args()
    if Path(args.run).name != args.run or args.run in (".", ".."):
        raise ValueError("new run directory required")
    spec = read(ROOT / SPEC)
    prior = ROOT / "results" / spec["source_run"]
    data = prepare(spec, read(prior / "cases.json"))
    prior_manifest = read(prior / "manifest.json")
    runs = {"warm-start": (WARM, "extended"), "canonical": (TRAINED, "canonical"), "target": (TRAINED, "target")}
    paths = {c: ROOT / "artifacts" / run / arm for c, (run, arm) in runs.items()}
    hashes = {c: digest(path / "adapter_model.safetensors") for c, path in paths.items()}
    assert hashes == prior_manifest["adapter_files_sha256"]
    sources = [SPEC, "venue_omission_diagnostic.py", "time_regression.py", "evaluate_fit.py", "fit_metrics.py",
               "train_smoke.py", "contract.py", "confirm_prompt.py", "prompt_diagnostic.py", "audit_evaluation.py",
               "annotation-rules-v1.md"]
    sources += [(prior / n).relative_to(ROOT).as_posix() for n in ("manifest.json", "cases.json", "review.json")]
    sources += [(prior / f"{c}-generation.json").relative_to(ROOT).as_posix() for c in runs]
    manifest = {"source_sha256": {n: digest(ROOT / n) for n in sources}, "base_revision": REVISION,
                "adapter_files_sha256": hashes,
                "published_reference_adapter_sha256": prior_manifest["published_reference_adapter_sha256"],
                "generation": prior_manifest["generation"]}
    tokenizer = AutoTokenizer.from_pretrained(SNAPSHOT, local_files_only=True)
    fixtures = [fixture(tokenizer, data, row) for row in data["rows"]]
    environment = {"gpu": torch.cuda.get_device_name(0), "cuda": torch.version.cuda,
                   "packages": {n: version(n) for n in ("torch", "transformers", "peft", "huggingface_hub")}}
    output = ROOT / "results" / args.run
    output.mkdir(parents=True, exist_ok=False)
    for name, value in (("protocol", spec), ("cases", data), ("manifest", manifest), ("environment", environment)):
        save_json(output / f"{name}.json", value)

    results = {}
    generation_calls = teacher_calls = 0
    for condition in spec["conditions"]:
        model = PeftModel.from_pretrained(load_base(), paths[condition], local_files_only=True, is_trainable=False)
        model.eval()
        assert not any(p.requires_grad for p in model.parameters())
        before = state_hash(model)
        generated = generate(model, tokenizer, data)
        summary, details, errors = aggregate(data["rows"], generated)
        previous = {r["id"]: r["token_ids"] for r in read(prior / f"{condition}-generation.json")}
        replays = {r["id"]: o["token_ids"] == previous[r["replay_id"]]
                   for r, o in zip(data["rows"], generated) if r["replay_id"] is not None}
        assert len(replays) == 5 and all(replays.values())
        teachers = [teacher_forced(model, tokenizer, f, o) for f, o in zip(fixtures, generated)]
        after = state_hash(model)
        assert before == after
        save_json(output / f"{condition}-generation.json", generated)
        save_json(output / f"{condition}-teacher.json", teachers)
        generation_calls += len(generated)
        teacher_calls += len(teachers)
        results[condition] = {"summary": summary, "details": details, "errors": errors, "token_replays": replays,
                              "parameters_before": before, "parameters_after": after}
        print(condition, summary, flush=True)
        del model
        gc.collect()
        torch.cuda.empty_cache()
    save_json(output / "review.json", {"training_steps": 0, "human_review": "pending", "promote_to_main": False,
              "weights_unchanged": True, "generation_calls": generation_calls, "teacher_forced_forwards": teacher_calls,
              "known_token_replays": sum(len(c["token_replays"]) for c in results.values()), "new_source_groups": 0,
              "results": results})


if __name__ == "__main__":
    main()
