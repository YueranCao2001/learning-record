import copy
import json
import unittest
from pathlib import Path
from unittest.mock import patch
from contract import load_pilot, score


class ContractTests(unittest.TestCase):
    def setUp(self):
        self.data = load_pilot()
        self.target = self.data["rows"][0]["expected"]

    def test_fixtures(self):
        self.assertEqual(len(self.data["rows"]), 6)

    def test_exact(self):
        self.assertTrue(score(json.dumps(self.target), self.target)["exact"])

    def test_fenced_json_is_not_raw_json(self):
        self.assertFalse(score('```json\n{}\n```', self.target)["json_valid"])

    def test_extra_key(self):
        self.assertFalse(score(json.dumps(dict(self.target, extra=1)), self.target)["schema_valid"])

    def test_wrong_type(self):
        self.assertFalse(score(json.dumps(dict(self.target, time=123)), self.target)["schema_valid"])

    def test_wrong_fact(self):
        result = score(json.dumps(dict(self.target, time=None)), self.target)
        self.assertTrue(result["schema_valid"])
        self.assertFalse(result["exact"])
        self.assertEqual(result["fields_correct"], 2)

    def test_cross_split_group(self):
        data = copy.deepcopy(self.data)
        data["rows"][3]["group"] = data["rows"][0]["group"]
        with patch.object(Path, "read_text", return_value=json.dumps(data)):
            with self.assertRaisesRegex(ValueError, "crosses splits"):
                load_pilot()


if __name__ == "__main__":
    unittest.main()
