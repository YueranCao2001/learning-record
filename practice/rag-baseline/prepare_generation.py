"""Prepare controlled-context requests only; no model call and no semantic scoring."""
import hashlib
import json
from pathlib import Path

from citations import load_snapshots, make_evidence
from generation_contract import build_request

HERE = Path(__file__).resolve().parent


def prepare(candidate, source_data):
    if candidate["schema_version"] != 1 or candidate["synthetic"] is not True:
        raise ValueError("expected synthetic candidate v1")
    snapshots = load_snapshots(source_data)
    allowed = set(candidate["allowed_sources"])
    cases = candidate["cases"]
    if len({c["id"] for c in cases}) != len(cases):
        raise ValueError("duplicate case ID")
    requests = []
    for case in cases:
        keys = [tuple(key) for key in case["context_sources"]]
        if len(set(keys)) != len(keys):
            raise ValueError("duplicate context source version")
        evidence = []
        for source_id, version in keys:
            if source_id not in allowed:
                raise ValueError("source outside allowed scope")
            text = snapshots.get((source_id, version))
            if text is None:
                raise ValueError("missing source version")
            evidence.append(make_evidence(source_id, version, text))
        # Only query and context enter the request; rubric remains in the candidate file.
        requests.append(build_request(case["id"], case["query"], tuple(evidence)))
    return requests


def report():
    candidate_path = HERE / "data/generation-pilot-candidate-v1.json"
    sources_path = HERE / "data/generation-pilot-sources-v1.json"
    candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
    sources = json.loads(sources_path.read_text(encoding="utf-8"))
    files = {"candidate": candidate_path, "sources": sources_path,
             **{name: HERE / name for name in
                ("prepare_generation.py", "generation_contract.py", "citations.py", "retrieval.py")}}
    return {"schema_version": 1, "execution_kind": "request_preparation_only",
            "model_calls": 0, "dataset_version": candidate["dataset_version"],
            "review_status": candidate["review_status"], "context_mode": candidate["context_mode"],
            "sha256": {name: hashlib.sha256(path.read_bytes()).hexdigest()
                       for name, path in files.items()},
            "requests": prepare(candidate, sources)}


if __name__ == "__main__":
    print(json.dumps(report(), ensure_ascii=False, indent=2, allow_nan=False))
