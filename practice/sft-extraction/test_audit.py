import copy
import unittest
from audit_contract import load_audit, validate_audit, retention_score
from contract import score


class AuditTests(unittest.TestCase):
    def setUp(self):
        self.data, self.prior = load_audit()

    def test_counts_and_human_status(self):
        self.assertEqual(len(self.data["rows"]), 12)
        self.assertEqual(self.data["human_review"], "pending")

    def test_source_overlap(self):
        self.data["rows"][0]["group"] = self.prior["rows"][0]["group"]
        with self.assertRaisesRegex(ValueError, "overlapping"):
            validate_audit(self.data, self.prior)

    def test_nonliteral_target(self):
        self.data["rows"][0]["expected"]["event"] = "不在原文的活动名"
        with self.assertRaisesRegex(ValueError, "nonliteral"):
            validate_audit(self.data, self.prior)

    def test_duplicate_audit(self):
        self.data["rows"].append(copy.deepcopy(self.data["rows"][0]))
        with self.assertRaises(ValueError):
            validate_audit(self.data, self.prior)

    def test_retention_array_types(self):
        row = {"kind": "json-array", "expected": [1, 4, 9]}
        self.assertTrue(retention_score('[1,4,9]', row))
        self.assertFalse(retention_score('[true,4,9]', row))
        self.assertFalse(retention_score('[1.0,4,9]', row))

    def test_json_null_distinct(self):
        expected = {"event": "例子", "time": None, "location": None}
        result = score('{"event":"例子","time":"null","location":null}', expected)
        self.assertTrue(result["schema_valid"])
        self.assertFalse(result["exact"])
        self.assertEqual(result["fields_correct"], 2)


if __name__ == "__main__":
    unittest.main()
