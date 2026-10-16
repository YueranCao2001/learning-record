import copy
import json
import unittest

from citations import full_quote
from ollama_smoke import normalize, payload_for
import test_generation_contract as fixture_module


class AdapterTests(unittest.TestCase):
    def setUp(self):
        fixture = fixture_module.ContractTests()
        fixture.setUp()
        self.request, self.evidence = fixture.request, fixture.packet
        quote = full_quote(self.evidence[0])
        self.output = {"status": "completed", "answer": quote["quote"], "reason": None,
                       "citations": [{"chunk_id": quote["chunk_id"], "quote": quote["quote"]}]}

    def raw(self, output):
        return {"done": True, "done_reason": "stop", "message": {"content": json.dumps(output)}}

    def test_payload(self):
        payload = payload_for(self.request)
        self.assertFalse(payload["stream"])
        self.assertFalse(payload["think"])
        self.assertEqual(payload["options"]["num_predict"], 512)

    def test_exact_normalization(self):
        result = normalize(self.raw(self.output), self.request, self.evidence)
        self.assertEqual(result["claims"][0]["citations"][0], full_quote(self.evidence[0]))

    def test_no_repair(self):
        for field, value in (("chunk_id", "invented"), ("quote", "错误引文")):
            output = copy.deepcopy(self.output)
            output["citations"][0][field] = value
            with self.assertRaises(ValueError):
                normalize(self.raw(output), self.request, self.evidence)

    def test_truncation_and_bad_json(self):
        raw = self.raw(self.output)
        raw["done_reason"] = "length"
        with self.assertRaises(ValueError):
            normalize(raw, self.request, self.evidence)
        with self.assertRaises(ValueError):
            normalize({"done": True, "done_reason": "stop", "message": {"content": "not JSON"}},
                      self.request, self.evidence)


if __name__ == "__main__":
    unittest.main()
