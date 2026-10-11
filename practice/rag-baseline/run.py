"""Run the fixed synthetic retrieval baseline; emit a JSON report to stdout."""
import hashlib
import json
import platform
from fractions import Fraction
from pathlib import Path

from retrieval import Doc, bigrams, rank, words

HERE = Path(__file__).resolve().parent
CONFIGS = (("word_overlap", words, "overlap"),
           ("word_bm25", words, "bm25"),
           ("bigram_bm25", bigrams, "bm25"))


def validate(data):
    if data["schema_version"] != 1 or data["synthetic"] is not True:
        raise ValueError("expected synthetic schema v1")
    doc_ids = [d["id"] for d in data["documents"]]
    case_ids = [c["id"] for c in data["cases"]]
    if len(set(doc_ids)) != len(doc_ids) or len(set(case_ids)) != len(case_ids):
        raise ValueError("duplicate IDs")
    docs = [Doc(d["id"], d["text"], frozenset(d["readers"])) for d in data["documents"]]
    for case in data["cases"]:
        required = set(case["required"])
        visible = {d.id for d in docs if case["user"] in d.readers}
        if len(required) != len(case["required"]) or not required <= visible:
            raise ValueError("required evidence must be unique and visible")
    return docs


def evaluate(data, k=2):
    docs = validate(data)
    results = []
    for name, analyzer, method in CONFIGS:
        rows = []
        complete = []
        for case in data["cases"]:
            # rank receives no gold labels.
            hits = rank(docs, case["query"], case["user"], analyzer, method, k)
            ids = {hit["id"] for hit in hits}
            required = set(case["required"])
            recall = str(Fraction(len(ids & required), len(required))) if required else None
            covered = required <= ids if required else None
            if covered is not None:
                complete.append(covered)
            rows.append({"case_id": case["id"], "slice": case["slice"],
                         "query": case["query"], "user": case["user"],
                         "required": sorted(required),
                         "hits": list(hits), "evidence_recall": recall,
                         "complete_evidence": covered,
                         "unanswerable_returned": bool(hits) if not required else None})
        results.append({"config": name, "k": k, "answerable_count": len(complete),
                        "complete_count": sum(complete),
                        "coverage": str(Fraction(sum(complete), len(complete))) if complete else None,
                        "rows": rows})
    return results


def report():
    raw = (HERE / "data" / "baseline-v1.json").read_bytes()
    data = json.loads(raw.decode("utf-8"))
    code_files = ("retrieval.py", "run.py")
    return {"report_schema_version": 1, "dataset_version": data["dataset_version"],
            "synthetic": True, "python_version": platform.python_version(),
            "data_sha256": hashlib.sha256(raw).hexdigest(),
            "code_sha256": {f: hashlib.sha256((HERE / f).read_bytes()).hexdigest()
                            for f in code_files},
            "idf": "log(N/df); visible documents only", "k1": 1.2, "b": 0.75,
            "query_term_frequency": "unique", "results": evaluate(data)}


if __name__ == "__main__":
    print(json.dumps(report(), ensure_ascii=False, indent=2, allow_nan=False))
