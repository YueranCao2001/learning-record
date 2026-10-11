import copy
import json
import unittest
from pathlib import Path

from retrieval import Doc, bigrams, search, spans, words
from run import CONFIGS, evaluate, validate

DATA = json.loads((Path(__file__).parent / "data" / "baseline-v1.json").read_text(encoding="utf-8"))


class BaselineTests(unittest.TestCase):
    def test_previous_results(self):
        for result in evaluate(DATA):
            rows = {r["case_id"]: r for r in result["rows"]}
            self.assertEqual(result["coverage"], "3/4")
            self.assertEqual([h["id"] for h in rows["custom"]["hits"]], ["exception", "rule"])
            self.assertEqual(rows["synonym"]["hits"], [])
            self.assertEqual(rows["missing"]["hits"], [])
            self.assertIsNone(rows["missing"]["evidence_recall"])
            self.assertIsNone(rows["missing"]["complete_evidence"])
            self.assertFalse(rows["missing"]["unanswerable_returned"])

    def test_reordering(self):
        reversed_data = copy.deepcopy(DATA)
        reversed_data["documents"].reverse()
        self.assertEqual(evaluate(DATA), evaluate(reversed_data))

    def test_permission_and_negation(self):
        docs = validate(DATA)
        for _, analyzer, method in CONFIGS:
            self.assertEqual(search(docs, "退货", "unknown", analyzer, method), ())
            self.assertNotIn("private", search(docs, "退货", "alice", analyzer, method))
            self.assertEqual(search(docs, "定制设备不退货", "alice", analyzer, method),
                             search(docs, "定制设备退货", "alice", analyzer, method))

    def test_gold_not_used_for_ranking(self):
        changed = copy.deepcopy(DATA)
        changed["cases"][0]["required"] = ["exception"]
        for old, new in zip(evaluate(DATA), evaluate(changed)):
            self.assertEqual(old["rows"][0]["hits"], new["rows"][0]["hits"])

    def test_analysis_boundaries(self):
        self.assertEqual(spans("ＡＢＣ，设备"), ["abc", "设备"])
        self.assertEqual(words("定制设备退货"), ["定制设备", "退货"])
        self.assertEqual(bigrams("退货，规则"), ["退货", "规则"])
        self.assertEqual(bigrams("退"), [])

    def test_empty_inputs(self):
        self.assertEqual(search([], "退货", "alice", words, "bm25"), ())
        self.assertEqual(search([Doc("empty", "。", frozenset({"alice"}))],
                                "退货", "alice", words, "bm25"), ())

    def test_invalid_parameters_and_ids(self):
        docs = validate(DATA)
        for k in (0, True):
            with self.assertRaises(ValueError):
                search(docs, "退货", "alice", words, "bm25", k)
        with self.assertRaises(ValueError):
            search(docs, "退货", "alice", words, "unknown")
        with self.assertRaises(ValueError):
            search(docs + [docs[0]], "退货", "alice", words, "bm25")
        for key in ("documents", "cases"):
            duplicate = copy.deepcopy(DATA)
            duplicate[key].append(duplicate[key][0])
            with self.assertRaises(ValueError):
                evaluate(duplicate)

    def test_inaccessible_gold_rejected(self):
        changed = copy.deepcopy(DATA)
        changed["cases"][0]["required"] = ["private"]
        with self.assertRaises(ValueError):
            evaluate(changed)


if __name__ == "__main__":
    unittest.main()
