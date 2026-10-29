"""Versioned repair-data inventory and split checks, without model dependencies."""
import copy
from collections import Counter
from contract import validate_pilot_data
from prompt_diagnostic import ROOT, read

CORE = ("id", "group", "split", "text", "expected")


def historical_rows():
    paths = ("results/2026-10-23-boundary-v2/data.json", "results/2026-10-27-prompt-diagnostic-v1/cases.json",
             "results/2026-10-28-prompt-confirmation-v1/cases.json")
    found = {}
    for name in paths:
        value = read(ROOT / name)
        rows = value["rows"] if isinstance(value, dict) else value
        for row in rows:
            canonical = {k: row[k] for k in CORE}
            if row["id"] in found and found[row["id"]] != canonical:
                raise ValueError("historical-identity-conflict")
            found.setdefault(row["id"], canonical)
    return list(found.values())


def compose():
    additions = read(ROOT / "data/repair-data-v1.json")
    protocol = read(ROOT / "data/repair-protocol-v1.json")
    rows = [dict(r, cohort="old-train" if r["split"] == "train" else "historical-regression", family=None)
            for r in historical_rows()]
    rows += [dict(r, cohort="new-train") for r in additions["train"]]
    for group in additions["dev_groups"]:
        for state in protocol["states"]:
            time_present = state in ("complete", "missing-location")
            location_present = state in ("complete", "missing-time")
            rows.append({"id": group["id"] + "-" + state, "group": group["group"], "family": group["family"],
                         "split": "dev", "cohort": "new-dev", "state": state,
                         "text": group["template"].format(time_clause=group["time_present" if time_present else "time_missing"],
                                                          location_clause=group["location_present" if location_present else "location_missing"]),
                         "expected": {"event": group["event"], "time": group["time"] if time_present else None,
                                      "location": group["location"] if location_present else None}})
    data = {"version": "repair-data-v1", "instruction": read(ROOT / "data/pilot-v1.json")["instruction"],
            "provenance": additions["provenance"], "human_review": "pending", "rows": rows}
    return data, protocol


def evaluation_rows(data):
    return [r for cohort in ("new-dev", "historical-regression") for r in data["rows"] if r["cohort"] == cohort]


def validate(data, protocol):
    historical = {r["id"]: r for r in historical_rows()}
    rows = data["rows"]
    lookup = {r["id"]: r for r in rows}
    for identity, old in historical.items():
        if identity not in lookup or {k: lookup[identity][k] for k in CORE} != old:
            raise ValueError("historical-mutation")
    reserved = {r["group"] for r in historical.values()}
    if any(r["group"] in reserved for r in rows if r["id"] not in historical):
        raise ValueError("reserved-source-group")
    group_splits = {}
    for row in rows:
        if group_splits.setdefault(row["group"], row["split"]) != row["split"]:
            raise ValueError("group-cross-split")
    train_families = {r["family"] for r in rows if r["cohort"] == "new-train"}
    dev_families = {r["family"] for r in rows if r["cohort"] == "new-dev"}
    if train_families & dev_families:
        raise ValueError("family-cross-split")
    if len({r["text"] for r in rows}) != len(rows):
        raise ValueError("duplicate-text")
    if any(v is not None and v not in r["text"] for r in rows for v in r["expected"].values()):
        raise ValueError("nonliteral-label")
    validate_pilot_data(data)
    for row in rows:
        if row.get("state"):
            state = (row["expected"]["time"] is not None, row["expected"]["location"] is not None)
            expected = {"complete": (True, True), "missing-time": (False, True),
                        "missing-location": (True, False), "missing-both": (False, False)}[row["state"]]
            if state != expected:
                raise ValueError("state-label-mismatch")
    for arm in ("control", "repair"):
        order = protocol[arm + "_order"]
        if len(order) != protocol["planned_steps_per_arm"]:
            raise ValueError("schedule-budget")
        if any(n not in lookup or lookup[n]["split"] != "train" for n in order):
            raise ValueError("schedule-non-train")
    counts = Counter(r["cohort"] for r in rows)
    if counts != {"old-train": 6, "new-train": 6, "new-dev": 12, "historical-regression": 45}:
        raise ValueError("cohort-count")
    return {"counts": dict(counts), "source_group_count": len(group_splits),
            "new_train_groups": sorted({r["group"] for r in rows if r["cohort"] == "new-train"}),
            "new_dev_groups": sorted({r["group"] for r in rows if r["cohort"] == "new-dev"}),
            "new_train_states": dict(Counter(r["state"] for r in rows if r["cohort"] == "new-train")),
            "historical_labels_unchanged": True, "source_groups_disjoint": True,
            "new_family_names_disjoint": True, "semantic_independence_proven": False,
            "all_labels_literal": True, "human_review": "pending"}


def rejection_checks(data, protocol):
    cases = {"historical-dev-relabel": "historical-mutation", "reuse-historical-group": "reserved-source-group",
             "group-leak": "group-cross-split", "family-leak": "family-cross-split",
             "duplicate-text": "duplicate-text", "nonliteral-label": "nonliteral-label",
             "dev-in-schedule": "schedule-non-train"}
    reports = []
    for name, expected in cases.items():
        changed, plan = copy.deepcopy(data), copy.deepcopy(protocol)
        train = next(r for r in changed["rows"] if r["cohort"] == "new-train")
        dev = next(r for r in changed["rows"] if r["cohort"] == "new-dev")
        old = next(r for r in changed["rows"] if r["cohort"] == "historical-regression")
        if name == "historical-dev-relabel": old["split"] = "train"
        elif name == "reuse-historical-group": train["group"] = old["group"]
        elif name == "group-leak": dev["group"] = train["group"]
        elif name == "family-leak": dev["family"] = train["family"]
        elif name == "duplicate-text": dev["text"] = train["text"]
        elif name == "nonliteral-label": train["expected"]["location"] = "不存在的测试场地"
        elif name == "dev-in-schedule": plan["repair_order"][-1] = dev["id"]
        try:
            validate(changed, plan)
        except ValueError as error:
            if str(error) != expected: raise
            reports.append({"case": name, "rejected": True, "reason": str(error), "optimizer_updates": 0,
                            "scope": "in-memory fault injection into disposable data/protocol copies"})
        else:
            raise AssertionError("fault accepted: " + name)
    return reports
