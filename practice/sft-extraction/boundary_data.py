"""Compose immutable pilot data with explicit, separately reviewed additions."""
import json
from pathlib import Path
from contract import load_pilot, validate_pilot_data

ROOT = Path(__file__).resolve().parent


def compose():
    original = load_pilot()
    additions = json.loads((ROOT / "data/boundary-additions-v2.json").read_text(encoding="utf-8"))
    return validate_pilot_data({"version": "boundary-v2", "instruction": original["instruction"],
            "provenance": additions["provenance"], "rows": original["rows"] + additions["rows"]})


def validate_protocol(data, protocol):
    rows = {r["id"]: r for r in data["rows"]}
    for arm in ("control", "candidate"):
        order = protocol[arm + "_order"]
        if len(order) != protocol["steps_per_arm"]:
            raise ValueError("step budget mismatch")
        if any(identity not in rows or rows[identity]["split"] != "train" for identity in order):
            raise ValueError("non-training row in optimizer schedule")
    actual = [r["id"] for r in data["rows"] if r["split"] == "dev"]
    if actual != protocol["dev_ids"]:
        raise ValueError("evaluation identities/order changed")


if __name__ == "__main__":
    data = compose()
    protocol = json.loads((ROOT / "data/boundary-protocol-v2.json").read_text(encoding="utf-8"))
    validate_protocol(data, protocol)
    print({"train": sum(r["split"] == "train" for r in data["rows"]),
           "dev": sum(r["split"] == "dev" for r in data["rows"]), "model_run": False})
