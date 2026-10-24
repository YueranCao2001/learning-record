"""Open-audit validation, sharing the frozen extraction scorer without changing it."""
import json
from pathlib import Path
from contract import validate_target

ROOT = Path(__file__).resolve().parent


def load_audit():
    data = json.loads((ROOT / "data/audit-v1.json").read_text(encoding="utf-8"))
    prior = json.loads((ROOT / "results/2026-10-23-boundary-v2/data.json").read_text(encoding="utf-8"))
    validate_audit(data, prior)
    return data, prior


def validate_audit(data, prior):
    ids = {r["id"] for r in prior["rows"]}
    groups = {r["group"] for r in prior["rows"]}
    texts = {r["text"] for r in prior["rows"]}
    for row in data["rows"]:
        if not row["id"] or not row["group"] or row["id"] in ids or row["group"] in groups or row["text"] in texts:
            raise ValueError("duplicate or overlapping audit identity/source/text")
        if not isinstance(row["text"], str) or not row["text"].strip():
            raise ValueError("empty audit text")
        validate_target(row["expected"])
        if any(v is not None and v not in row["text"] for v in row["expected"].values()):
            raise ValueError("nonliteral audit target")
        ids.add(row["id"])
        groups.add(row["group"])
        texts.add(row["text"])


def retention_score(raw, row):
    if row["kind"] == "text":
        return raw.strip() == row["expected"]
    if row["kind"] == "json-array":
        try:
            value = json.loads(raw)
        except (ValueError, TypeError):
            return False
        return (isinstance(value, list) and all(type(v) is int for v in value)
                and value == row["expected"])
    raise ValueError("unknown retention scorer")
