"""Paired review of the frozen repair run, reusing existing strict field metrics."""
from pathlib import Path
from confirm_prompt import aggregate
from prompt_diagnostic import read
from repair_data import evaluation_rows


def paired(rows, first, second):
    _, a, _ = aggregate(rows, first)
    _, b, _ = aggregate(rows, second)
    return {"fixed_ids": [y["id"] for x, y in zip(a, b) if not x["exact"] and y["exact"]],
            "regressed_ids": [y["id"] for x, y in zip(a, b) if x["exact"] and not y["exact"]]}


def review(folder):
    folder = Path(folder)
    rows = evaluation_rows(read(folder / "data.json"))
    execution = read(folder / "execution-protocol.json")
    outputs = {"baseline": read(folder / "baseline.json")}
    for arm in execution["arms"]:
        outputs[arm] = read(folder / arm / "after.json")
    summaries, details, transitions = {}, {}, {}
    for name, generated in outputs.items():
        aggregate(rows, generated)  # Reject altered scores and missing/reordered identities.
        summaries[name], details[name] = {}, {}
        for cohort in ("new-dev", "historical-regression"):
            selected = [(r, o) for r, o in zip(rows, generated) if r["cohort"] == cohort]
            subset, actual = map(list, zip(*selected))
            summary, records, errors = aggregate(subset, actual)
            if cohort == "new-dev":
                summary["by_state"] = {}
                for state in ("complete", "missing-time", "missing-location", "missing-both"):
                    members = [(r, o) for r, o in selected if r["state"] == state]
                    summary["by_state"][state] = aggregate(*map(list, zip(*members)))[0]
            summaries[name][cohort] = summary
            details[name][cohort] = {"records": records, "errors": errors}
    for first, second in (("baseline", "control"), ("baseline", "repair"), ("control", "repair")):
        transitions[first + "-to-" + second] = {}
        for cohort in ("new-dev", "historical-regression"):
            subset = [(r, a, b) for r, a, b in zip(rows, outputs[first], outputs[second]) if r["cohort"] == cohort]
            transitions[first + "-to-" + second][cohort] = paired(*map(list, zip(*subset)))
    arms = {a: read(folder / a / "summary.json") for a in execution["arms"]}
    reloaded = {a: read(folder / a / "reloaded.json") for a in execution["arms"]}
    calls = len(outputs["baseline"]) + sum(len(outputs[a]) + len(reloaded[a]) for a in execution["arms"])
    return {"training_steps": sum(s["steps"] for s in arms.values()), "total_generation_calls": calls,
            "human_review": "pending", "promote_to_main": False, "trained_weights_uploaded": False,
            "same_initial_adapter": len({s["initial_adapter_sha256"] for s in arms.values()}) == 1,
            "same_initial_base": len({s["frozen_base_sha256_before"] for s in arms.values()}) == 1,
            "both_base_unchanged": all(s["base_unchanged"] for s in arms.values()),
            "both_fresh_optimizers": all(s["optimizer_state_entries_before"] == 0 for s in arms.values()),
            "both_reload_exact": all(s["adapter_reload_exact"] and s["reload_generation_tokens_equal"] for s in arms.values()),
            "both_reload_tokens_recomputed_equal": all([o["token_ids"] for o in outputs[a]] == [o["token_ids"] for o in reloaded[a]] for a in arms),
            "trained_target_tokens": {a: s["trained_target_tokens"] for a, s in arms.items()},
            "first_six_training_records_equal": read(folder / "control/training.json")[:6] == read(folder / "repair/training.json")[:6],
            "summaries": summaries, "transitions": transitions, "details": details,
            "scope": execution["scope"]}
