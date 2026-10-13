"""Fixed synthetic retrieval-to-citation exercise; JSON report to stdout."""
import hashlib
import json
import platform
from dataclasses import asdict
from pathlib import Path

from citations import assemble, check_quote, full_quote, load_snapshots, make_evidence
from retrieval import words
from run import validate

HERE = Path(__file__).resolve().parent


def report():
    dataset_file = HERE / "data/baseline-v1.json"
    source_file = HERE / "data/source-snapshots-v1.json"
    dataset = json.loads(dataset_file.read_text(encoding="utf-8"))
    source_data = json.loads(source_file.read_text(encoding="utf-8"))
    docs = validate(dataset)
    snapshots = load_snapshots(source_data)
    versions = {d.id: "v1" for d in docs}
    user, query = "alice", "普通设备退货"
    evidence = assemble(docs, query, user, words, "overlap", snapshots, versions, 2)
    allowed = {d.id for d in docs if user in d.readers}
    first = evidence[0]
    good = full_quote(first)
    warranty = make_evidence("warranty", "v1", snapshots[("warranty", "v1")])
    cases = [
        ("valid", good, evidence, snapshots, allowed, True),
        ("forged_id", dict(good, chunk_id="invented"), evidence, snapshots, allowed, False),
        ("altered_quote", dict(good, quote="所有设备支持退货。"), evidence, snapshots, allowed, False),
        ("wrong_version", dict(good, version="v2"), evidence, snapshots, allowed, False),
        ("wrong_source", dict(good, source_id="warranty"), evidence, snapshots, allowed, False),
        ("not_in_context", full_quote(warranty), evidence, snapshots, allowed, False),
        ("outside_range", dict(good, end=first.end + 1), evidence, snapshots, allowed, False),
        ("changed_snapshot", good, evidence,
         dict(snapshots) | {("rule", "v1"): "被改写的政策。"}, allowed, False),
        ("revoked", good, evidence, snapshots, allowed - {"rule"}, False),
        ("contradictory_claim", good, evidence, snapshots, allowed, True),
    ]
    rows = []
    for name, citation, packet, sources, permission, expected in cases:
        try:
            passed = check_quote(citation, packet, sources, permission)
            reason = None
        except ValueError as error:
            passed, reason = False, str(error)
        if passed != expected:
            raise AssertionError(name)
        row = {"case_id": name, "citation": citation, "provenance_passed": passed,
               "expected_provenance": expected, "reason": reason,
               "semantic_support": "not_evaluated"}
        if name == "contradictory_claim":
            row["claim"] = "所有设备都支持30天内退货。"
            row["manual_note"] = "The claim broadens ordinary equipment to all equipment; exact quote checking does not catch it."
        rows.append(row)
    files = {"dataset": dataset_file, "snapshots": source_file,
             **{f: HERE / f for f in ("retrieval.py", "run.py", "citations.py", "citation_demo.py")}}
    return {"schema_version": 1, "synthetic": True, "python_version": platform.python_version(),
            "query": query, "user": user, "config": "word_overlap", "k": 2,
            "snapshot_version": source_data["snapshot_version"],
            "sha256": {key: hashlib.sha256(path.read_bytes()).hexdigest() for key, path in files.items()},
            "supplied_evidence": [asdict(c) for c in evidence],
            "rows": rows, "cases_matched_expectation": len(rows),
            "provenance_pass_count": sum(row["provenance_passed"] for row in rows),
            "semantic_support_evaluated": False}


if __name__ == "__main__":
    print(json.dumps(report(), ensure_ascii=False, indent=2, allow_nan=False))
