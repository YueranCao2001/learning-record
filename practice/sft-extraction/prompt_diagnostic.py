"""Frozen prompt interventions and paired review; no model dependencies."""
import json
from collections import Counter
from pathlib import Path
from contract import score, validate_target

ROOT = Path(__file__).resolve().parent


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def compose():
    protocol = read(ROOT / "data/prompt-diagnostic-v1.json")
    instruction = read(ROOT / "data/pilot-v1.json")["instruction"]
    change = protocol["interventions"]["key-order"]
    if instruction.count(change["old"]) != 1:
        raise ValueError("key order intervention must replace one exact span")
    prompts = {"original": instruction, "key-order": instruction.replace(change["old"], change["new"]),
               "field-independence": instruction + protocol["interventions"]["field-independence"]}
    rows = [dict(r, split="dev", cohort="old-audit") for r in read(ROOT / "data/audit-v1.json")["rows"]]
    prior = read(ROOT / "results/2026-10-23-boundary-v2/data.json")["rows"]
    old_groups = {r["group"] for r in prior + rows}
    used_groups = set()
    for pair in protocol["pairs"]:
        if pair["group"] in old_groups | used_groups:
            raise ValueError("new pair source overlaps prior source")
        used_groups.add(pair["group"])
        for state in ("missing", "present"):
            rows.append({"id": pair["id"] + "-" + state, "group": pair["group"], "pair": pair["id"],
                         "time_state": state, "split": "dev", "cohort": "new-pairs",
                         "text": pair["header"] + pair["location_clause"] + "，" + pair[state + "_clause"] + "。",
                         "expected": {"event": pair["event"], "time": None if state == "missing" else pair["expected_time"],
                                      "location": pair["location"]}})
    if len(rows) != protocol["expected_rows"] or len({r["id"] for r in rows}) != len(rows):
        raise ValueError("unexpected row count or duplicate id")
    texts = [r["text"] for r in prior + rows]
    if len(set(texts)) != len(texts):
        raise ValueError("duplicate text")
    for row in rows:
        validate_target(row["expected"])
        if any(v is not None and v not in row["text"] for v in row["expected"].values()):
            raise ValueError("nonliteral target")
    if len(protocol["models"]) * len(prompts) * len(rows) != protocol["expected_generation_calls"]:
        raise ValueError("generation budget mismatch")
    return protocol, prompts, rows


def metrics(rows, outputs):
    if [r["id"] for r in rows] != [r["id"] for r in outputs]:
        raise ValueError("missing, reordered or duplicate output")
    records = []
    for row, output in zip(rows, outputs):
        actual = score(output["raw"], row["expected"])
        if output["score"] != actual:
            raise ValueError("stored score differs from raw output")
        value = json.loads(output["raw"]) if actual["schema_valid"] else None
        fields = {k: value is not None and value[k] == v for k, v in row["expected"].items()}
        records.append({"id": row["id"], "cohort": row["cohort"], "exact": actual["exact"],
                        "fields": fields, "parsed": value,
                        "location_omitted": value is not None and row["expected"]["location"] is not None and value["location"] is None})
    summary = {}
    for cohort in ("old-audit", "new-pairs"):
        selected = [r for r in records if r["cohort"] == cohort]
        summary[cohort] = {"count": len(selected), "exact": sum(r["exact"] for r in selected),
                           "fields_correct": {k: sum(r["fields"][k] for r in selected) for k in ("event", "time", "location")},
                           "location_omitted_ids": [r["id"] for r in selected if r["location_omitted"]]}
    summary["json_valid"] = sum(r["score"]["json_valid"] for r in outputs)
    summary["schema_valid"] = sum(r["score"]["schema_valid"] for r in outputs)
    summary["emitted_key_orders"] = dict(Counter(",".join(r["parsed"]) if r["parsed"] else "invalid" for r in records))
    return summary, records


def review(folder):
    folder = Path(folder)
    rows, protocol = read(folder / "cases.json"), read(folder / "protocol.json")
    summaries, detail, transitions, pair_table, reproduction = {}, {}, {}, [], {}
    for model in protocol["models"]:
        summaries[model], detail[model], transitions[model] = {}, {}, {}
        original = None
        for condition in protocol["conditions"]:
            outputs = read(folder / f"{model}-{condition}.json")
            summary, records = metrics(rows, outputs)
            summaries[model][condition], detail[model][condition] = summary, records
            if condition == "original":
                original = records
                prior = read(ROOT / f"results/2026-10-24-open-audit-v1/{model}-extraction.json")
                prior_by_id = {r["id"]: r for r in prior}
                reproduction[model] = all(o["token_ids"] == prior_by_id[o["id"]]["token_ids"] for o in outputs[:12])
            else:
                transitions[model][condition] = {
                    "fixed_ids": [b["id"] for a, b in zip(original, records) if not a["exact"] and b["exact"]],
                    "regressed_ids": [b["id"] for a, b in zip(original, records) if a["exact"] and not b["exact"]]}
            by_id = {r["id"]: r for r in records}
            for pair in protocol["pairs"]:
                pair_table.append({"model": model, "condition": condition, "pair": pair["id"],
                                   "group": pair["group"], **{s: by_id[pair["id"] + "-" + s] for s in ("missing", "present")}})
    return {"training_steps": 0, "human_review": "pending", "promote_to_main": False,
            "total_generation_calls": sum(s["old-audit"]["count"] + s["new-pairs"]["count"] for m in summaries.values() for s in m.values()),
            "old_audit_tokens_reproduce": reproduction, "new_source_groups": len(protocol["pairs"]),
            "summaries": summaries, "transitions_from_original": transitions,
            "pair_table": pair_table, "detail": detail}
