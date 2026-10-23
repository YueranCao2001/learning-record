"""Small diagnostic contract; deliberately independent of training libraries."""
import json
from pathlib import Path

KEYS = {"event", "time", "location"}


def validate_target(value):
    if not isinstance(value, dict) or set(value) != KEYS:
        raise ValueError("exactly event, time, location required")
    if any(v is not None and (not isinstance(v, str) or not v.strip())
           for v in value.values()):
        raise ValueError("nonblank string or null required")


def load_pilot(path=None):
    path = path or Path(__file__).parent / "data/pilot-v1.json"
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return validate_pilot_data(data)


def validate_pilot_data(data):
    """Validate in-memory compositions with the same original sample rules."""
    seen, groups, texts = set(), {}, set()
    counts = {"train": 0, "dev": 0}
    for row in data["rows"]:
        if row["id"] in seen or not row["id"] or not row["group"]:
            raise ValueError("invalid identity")
        seen.add(row["id"])
        split = row["split"]
        if split not in counts:
            raise ValueError("unknown split")
        if groups.setdefault(row["group"], split) != split:
            raise ValueError("source group crosses splits")
        if not isinstance(row["text"], str) or not row["text"].strip() or row["text"] in texts:
            raise ValueError("empty or duplicate text")
        texts.add(row["text"])
        validate_target(row["expected"])
        if any(v is not None and v not in row["text"] for v in row["expected"].values()):
            raise ValueError("target must be a literal input span")
        counts[split] += 1
    if not all(counts.values()):
        raise ValueError("both splits required")
    return data


def score(raw, expected):
    validate_target(expected)
    try:
        value = json.loads(raw)
    except (ValueError, TypeError):
        return {"json_valid": False, "schema_valid": False, "exact": False, "fields_correct": 0}
    try:
        validate_target(value)
    except ValueError:
        return {"json_valid": True, "schema_valid": False, "exact": False, "fields_correct": 0}
    count = sum(value[k] == expected[k] for k in KEYS)
    return {"json_valid": True, "schema_valid": True, "exact": count == 3, "fields_correct": count}


if __name__ == "__main__":
    data = load_pilot()
    print(json.dumps({"version": data["version"], "counts": {
        split: sum(r["split"] == split for r in data["rows"])
        for split in ("train", "dev")}, "model_run": False}, ensure_ascii=False))
