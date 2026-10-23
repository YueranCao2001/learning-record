"""Recompute paired diagnostics without another generation or training run."""
import argparse
import json
from pathlib import Path
from contract import load_pilot, score

ROOT = Path(__file__).resolve().parent


def review(run):
    output = ROOT / "results" / run
    data = load_pilot(output / "data.json")
    targets = {r["id"]: r["expected"] for r in data["rows"] if r["split"] == "dev"}
    paths = {"base": output / "baseline.json", "control": output / "control/after.json",
             "candidate": output / "candidate/after.json"}
    conditions = {name: json.loads(path.read_text(encoding="utf-8")) for name, path in paths.items()}
    for rows in conditions.values():
        assert [r["id"] for r in rows] == list(targets)
        for r in rows:
            assert score(r["raw"], targets[r["id"]]) == r["score"]
    paired = [{"id": identity, **{name: next(r["score"] for r in rows if r["id"] == identity)
                                    for name, rows in conditions.items()}}
              for identity in targets]
    subsets = {"old_regression": [k for k in targets if k.startswith("dev-")],
               "new_diagnostic": [k for k in targets if k.startswith("boundary-dev-")],
               "all": list(targets)}
    metrics = {}
    for subset, identities in subsets.items():
        metrics[subset] = {}
        for name, rows in conditions.items():
            selected = [r for r in rows if r["id"] in identities]
            event_correct = sum(json.loads(r["raw"]).get("event") == targets[r["id"]]["event"]
                                for r in selected if r["score"]["schema_valid"])
            metrics[subset][name] = {"n": len(selected),
                "exact": sum(r["score"]["exact"] for r in selected),
                "fields_correct": sum(r["score"]["fields_correct"] for r in selected),
                "event_correct": event_correct}
    yesterday = json.loads((ROOT / "results/2026-10-22-lora-smoke-v1/after.json").read_text(encoding="utf-8"))
    control = conditions["control"][:3]
    reproduced = [r["token_ids"] for r in yesterday] == [r["token_ids"] for r in control]
    return {"metrics": metrics, "paired": paired,
            "control_old_tokens_reproduce_yesterday": reproduced,
            "interpretation": "One fixed-seed, targeted synthetic diagnostic comparison; not independent test accuracy. Exposure/order and supervised token counts differ between arms."}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    args = parser.parse_args()
    assert Path(args.run).name == args.run
    report = review(args.run)
    (ROOT / "results" / args.run / "review.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["metrics"], ensure_ascii=False))
    print("control_old_tokens_reproduce_yesterday:", report["control_old_tokens_reproduce_yesterday"])
