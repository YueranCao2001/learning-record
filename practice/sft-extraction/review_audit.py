"""Recompute the open audit with frozen scorers and retain paired errors."""
import json
from pathlib import Path
from contract import score
from audit_contract import retention_score

ROOT = Path(__file__).resolve().parent


def review(run):
    folder = ROOT / "results" / run
    read = lambda name: json.loads((folder / name).read_text(encoding="utf-8"))
    data = read("extraction-data.json")
    targets = {r["id"]: r["expected"] for r in data["rows"]}
    conditions = {name: read(name + "-extraction.json") for name in ("base", "candidate")}
    for rows in conditions.values():
        assert [r["id"] for r in rows] == list(targets)
        for row in rows:
            assert row["score"] == score(row["raw"], targets[row["id"]])
    subsets = {"old_regression": [r for r in data["rows"] if not r["id"].startswith("audit-")],
               "new_open_audit": [r for r in data["rows"] if r["id"].startswith("audit-")]}
    for category in sorted({r["category"] for r in subsets["new_open_audit"]}):
        subsets[category] = [r for r in subsets["new_open_audit"] if r["category"] == category]
    metrics = {}
    for subset, selected in subsets.items():
        identities = {r["id"] for r in selected}
        metrics[subset] = {name: {"n": len(selected),
              "exact": sum(r["score"]["exact"] for r in rows if r["id"] in identities),
              "fields_correct": sum(r["score"]["fields_correct"] for r in rows if r["id"] in identities)}
              for name, rows in conditions.items()}
    protocol = read("protocol.json")
    metrics["retention"] = {}
    for name in conditions:
        rows = read(name + "-retention.json")
        assert [r["id"] for r in rows] == [r["id"] for r in protocol["retention"]["rows"]]
        for row, target in zip(rows, protocol["retention"]["rows"]):
            assert row["correct"] == retention_score(row["raw"], target)
        metrics["retention"][name] = {"n": len(rows), "correct": sum(r["correct"] for r in rows)}
    paired = []
    for base, candidate in zip(conditions["base"], conditions["candidate"]):
        identity = base["id"]
        if not candidate["score"]["exact"] or base["score"]["exact"] != candidate["score"]["exact"]:
            paired.append({"id": identity, "expected": targets[identity], "base_raw": base["raw"],
                "candidate_raw": candidate["raw"], "base_score": base["score"], "candidate_score": candidate["score"]})
    previous = json.loads((ROOT / "results/2026-10-23-boundary-v2/candidate/after.json").read_text(encoding="utf-8"))
    old = conditions["candidate"][:9]
    return {"metrics": metrics, "paired_errors_or_differences": paired,
            "old_candidate_tokens_reproduce": [r["token_ids"] for r in old] == [r["token_ids"] for r in previous],
            "total_generation_calls": sum(len(v) for v in conditions.values()) + 10,
            "training_steps": 0, "human_review": "pending", "promote_to_main": False,
            "scope": protocol["scope"]}
