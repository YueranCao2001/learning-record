import copy
import json
import unittest
from datetime import date
from pathlib import Path

from prepare_generation import prepare, report

HERE = Path(__file__).parent
CANDIDATE = json.loads((HERE / "data/generation-pilot-candidate-v1.json").read_text(encoding="utf-8"))
SOURCES = json.loads((HERE / "data/generation-pilot-sources-v1.json").read_text(encoding="utf-8"))


class PreparationTests(unittest.TestCase):
    def test_requests_and_no_labels(self):
        requests = prepare(CANDIDATE, SOURCES)
        self.assertEqual(len(requests), 5)
        self.assertEqual([len(r["evidence"]) for r in requests], [1, 2, 0, 1, 2])
        for request in requests:
            self.assertEqual(set(request), {"contract_version", "request_id", "query",
                                            "context_sha256", "evidence"})
        self.assertEqual(report()["model_calls"], 0)

    def test_rubric_change_does_not_change_requests(self):
        changed = copy.deepcopy(CANDIDATE)
        for case in changed["cases"]:
            case["rubric"] = {"wrong_label": "not visible to model"}
        self.assertEqual(prepare(CANDIDATE, SOURCES), prepare(changed, SOURCES))

    def test_missing_and_unauthorized_source(self):
        missing = copy.deepcopy(CANDIDATE)
        missing["cases"][0]["context_sources"] = [["rule", "missing"]]
        with self.assertRaises(ValueError):
            prepare(missing, SOURCES)
        denied = copy.deepcopy(CANDIDATE)
        denied["allowed_sources"] = []
        with self.assertRaises(ValueError):
            prepare(denied, SOURCES)

    def test_duplicate_case_and_context(self):
        changed = copy.deepcopy(CANDIDATE)
        changed["cases"].append(changed["cases"][0])
        with self.assertRaises(ValueError):
            prepare(changed, SOURCES)
        changed = copy.deepcopy(CANDIDATE)
        changed["cases"][0]["context_sources"] *= 2
        with self.assertRaises(ValueError):
            prepare(changed, SOURCES)

    def test_effective_date_boundary_fixture(self):
        dated = [s for s in SOURCES["sources"] if s["source_id"] == "dated-policy"]
        def applicable(day):
            return [s["version"] for s in dated
                    if date.fromisoformat(s["effective_from"]) <= day and
                    (s["effective_until_exclusive"] is None or
                     day < date.fromisoformat(s["effective_until_exclusive"]))]
        self.assertEqual(applicable(date(2026, 10, 14)), ["v1"])
        self.assertEqual(applicable(date(2026, 10, 15)), ["v2"])
        self.assertEqual(applicable(date(2026, 10, 16)), ["v2"])


if __name__ == "__main__":
    unittest.main()
