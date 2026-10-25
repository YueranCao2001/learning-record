"""Construct one-factor surface variants of the existing audit-03 source."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def compose():
    source = json.loads((ROOT / "data/location-minimal-v1.json").read_text(encoding="utf-8"))
    rows = []
    for variant in source["variants"]:
        keys = set(variant) - {"id", "factor"}
        allowed = {"anchor": set(), "time-wording": {"time_clause"},
                   "time-present": {"time_clause", "expected_time"}, "clause-order": {"reverse_clauses"},
                   "location-wording": {"location_clause"}, "header-shortening": {"header"}}
        if variant["factor"] not in allowed or keys != allowed[variant["factor"]]:
            raise ValueError("variant changes undeclared factors")
        header = variant.get("header", source["header"])
        location = variant.get("location_clause", source["location_clause"])
        time = variant.get("time_clause", source["time_clause"])
        clauses = [time, location] if variant.get("reverse_clauses") else [location, time]
        text = header + "，".join(clauses) + "。"
        expected = {"event": source["event"], "time": variant.get("expected_time"), "location": source["location"]}
        if any(v is not None and v not in text for v in expected.values()):
            raise ValueError("nonliteral minimal-pair target")
        rows.append({"id": variant["id"], "group": source["source_group"], "parent_id": source["parent_id"],
                     "factor": variant["factor"], "split": "dev", "text": text, "expected": expected})
    if len({r["id"] for r in rows}) != len(rows) or len({r["text"] for r in rows}) != len(rows):
        raise ValueError("duplicate variants")
    audit = json.loads((ROOT / "data/audit-v1.json").read_text(encoding="utf-8"))
    parent = next(r for r in audit["rows"] if r["id"] == source["parent_id"])
    if rows[0]["text"] != parent["text"] or rows[0]["expected"] != parent["expected"]:
        raise ValueError("anchor differs from original failure")
    return rows


def review(run):
    from contract import score
    folder = ROOT / "results" / run
    read = lambda name: json.loads((folder / name).read_text(encoding="utf-8"))
    rows = read("cases.json")
    conditions = {n: read(n + "-extraction.json") for n in ("base", "candidate")}
    table = []
    for index, row in enumerate(rows):
        item = {"id": row["id"], "factor": row["factor"]}
        for name, outputs in conditions.items():
            actual = outputs[index]
            assert actual["id"] == row["id"] and actual["score"] == score(actual["raw"], row["expected"])
            parsed = json.loads(actual["raw"]) if actual["score"]["schema_valid"] else {}
            item[name] = {"location": parsed.get("location"), "location_correct": bool(parsed) and parsed["location"] == row["expected"]["location"],
                          "exact": actual["score"]["exact"]}
        table.append(item)
    prior_folder = ROOT / "results/2026-10-24-open-audit-v1"
    reproduced = {}
    for name, outputs in conditions.items():
        prior = json.loads((prior_folder / (name + "-extraction.json")).read_text(encoding="utf-8"))
        anchor = next(r for r in prior if r["id"] == "audit-03")
        reproduced[name] = outputs[0]["token_ids"] == anchor["token_ids"]
    return {"table": table, "anchor_tokens_reproduce": reproduced,
            "total_generation_calls": sum(len(v) for v in conditions.values()),
            "source_groups": 1, "training_steps": 0, "human_review": "pending", "promote_to_main": False}
