import copy
import json
import unittest
from pathlib import Path

from citations import assemble, full_quote, load_snapshots
from generation_contract import build_request, validate_response
from retrieval import words
from run import validate

HERE = Path(__file__).parent


class ContractTests(unittest.TestCase):
    def setUp(self):
        data = json.loads((HERE / "data/baseline-v1.json").read_text(encoding="utf-8"))
        sources = json.loads((HERE / "data/source-snapshots-v1.json").read_text(encoding="utf-8"))
        docs = validate(data)
        self.snapshots = load_snapshots(sources)
        self.packet = assemble(docs, "普通设备退货", "alice", words, "overlap",
                               self.snapshots, {d.id: "v1" for d in docs})
        self.allowed = {d.id for d in docs if "alice" in d.readers}
        self.request = build_request("fixture-1", "普通设备能退货吗", self.packet)
        answer = self.packet[0].text
        self.response = {"contract_version": 1, "request_id": "fixture-1",
                         "context_sha256": self.request["context_sha256"],
                         "status": "completed", "answer": answer, "reason": None,
                         "claims": [{"start": 0, "end": len(answer),
                                     "citations": [full_quote(self.packet[0])]}]}

    def check(self, response):
        return validate_response(response, self.request, self.packet, self.snapshots, self.allowed)

    def test_completed(self):
        result = self.check(self.response)
        self.assertTrue(result["eligible_for_answer_review"])
        self.assertEqual(result["checked_citations"], 1)
        self.assertEqual(result["semantic_support"], "not_evaluated")

    def test_wrong_request_or_context(self):
        for key in ("request_id", "context_sha256"):
            with self.assertRaises(ValueError):
                self.check(dict(self.response, **{key: "wrong"}))
        with self.assertRaises(ValueError):
            validate_response(self.response, self.request, self.packet[:1], self.snapshots, self.allowed)

    def test_schema_and_status(self):
        for response in (dict(self.response, extra=True), dict(self.response, status="unknown"),
                         dict(self.response, contract_version=True), dict(self.response, claims=[])):
            with self.assertRaises(ValueError):
                self.check(response)

    def test_noncompleted_states(self):
        for status in ("insufficient_evidence", "failed", "truncated"):
            response = dict(self.response, status=status, claims=[], reason="fixture reason")
            self.assertFalse(self.check(response)["eligible_for_answer_review"])
            with self.assertRaises(ValueError):
                self.check(dict(response, reason=None))

    def test_invalid_claim_range(self):
        for start, end in ((True, 2), (-1, 2), (0, 1000), (0, 0)):
            response = copy.deepcopy(self.response)
            response["claims"][0].update(start=start, end=end)
            with self.assertRaises(ValueError):
                self.check(response)

    def test_citation_check_is_reused(self):
        response = copy.deepcopy(self.response)
        response["claims"][0]["citations"][0]["quote"] = "错误原文"
        with self.assertRaises(ValueError):
            self.check(response)
        with self.assertRaises(ValueError):
            validate_response(self.response, self.request, self.packet, self.snapshots, set())

    def test_contradiction_still_requires_review(self):
        answer = "所有设备都支持30天内退货。"
        response = copy.deepcopy(self.response)
        response["answer"] = answer
        response["claims"][0]["end"] = len(answer)
        result = self.check(response)
        self.assertTrue(result["eligible_for_answer_review"])
        self.assertEqual(result["semantic_support"], "not_evaluated")


if __name__ == "__main__":
    unittest.main()
