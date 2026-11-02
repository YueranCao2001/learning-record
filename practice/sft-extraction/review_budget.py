"""Compare budget endpoints using existing strict metrics and target-NLL replay."""
from pathlib import Path
from confirm_prompt import aggregate
from fit_metrics import condition_summary
from prompt_diagnostic import read
from repair_data import evaluation_rows
from review_repair import paired


def rows_for_evaluation(data):
    return [r for r in data["rows"] if r["cohort"] == "new-train"] + evaluation_rows(data)


def review(folder):
    folder = Path(folder)
    data, protocol = read(folder / "data.json"), read(folder / "protocol.json")
    rows = rows_for_evaluation(data)
    fixtures = {f["id"]: f for f in read(folder / "token-fixtures.json")}
    outputs = {c: read(folder / c / "after.json") for c in protocol["conditions"]}
    summaries, details, changes, fitting = {}, {}, {}, {}
    for condition, actual in outputs.items():
        aggregate(rows, actual)
        summaries[condition], details[condition] = {}, {}
        for cohort in protocol["evaluation_order"]:
            selected = [(r, o) for r, o in zip(rows, actual) if r["cohort"] == cohort]
            subset, generated = map(list, zip(*selected))
            summary, records, errors = aggregate(subset, generated)
            summaries[condition][cohort] = summary
            details[condition][cohort] = {"records": records, "errors": errors}
            if cohort == "new-train":
                fitting[condition] = condition_summary(subset, generated, read(folder / condition / "teacher.json"), fixtures)
    for cohort in protocol["evaluation_order"]:
        selected = [(r, a, b) for r, a, b in zip(rows, outputs["short"], outputs["extended"]) if r["cohort"] == cohort]
        changes[cohort] = paired(*map(list, zip(*selected)))
    summary = read(folder / "extended/summary.json")
    replay = read(folder / "short/replay.json")
    comparison = read(folder / "prefix-comparison.json")
    reloaded = read(folder / "extended/reloaded.json")
    return {"training_steps": len(read(folder / "extended/training.json")),
            "total_generation_calls": sum(len(v) for v in outputs.values()) + len(reloaded),
            "teacher_forced_forwards": sum(len(read(folder / c / "teacher.json")) for c in protocol["conditions"]),
            "human_review": "pending", "promote_to_main": False, "trained_weights_uploaded": False,
            "short_tokens_reproduce": replay["all_tokens_equal"], "initial_adapter_equal": comparison["initial_adapter_equal"],
            "first_twelve_records_equal": comparison["first_twelve_records_equal"], "extended_base_unchanged": summary["base_unchanged"],
            "extended_reload_exact": summary["adapter_reload_exact"] and summary["reload_generation_tokens_equal"],
            "reloaded_tokens_recomputed_equal": [o["token_ids"] for o in reloaded] == [o["token_ids"] for o in outputs["extended"]],
            "optimizer_state_entries_before": summary["optimizer_state_entries_before"], "optimizer_final_steps": summary["optimizer_final_steps"],
            "trained_target_tokens": summary["trained_target_tokens"],
            "summaries": summaries, "fitting": fitting, "short_to_extended": changes, "details": details,
            "scope": protocol["scope"]}
