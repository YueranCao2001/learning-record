import copy
import json
from pathlib import Path
import unittest
from boundary_data import compose, validate_protocol
from contract import validate_pilot_data

ROOT = Path(__file__).resolve().parent


class BoundaryTests(unittest.TestCase):
    def setUp(self):
        self.data = compose()
        self.protocol = json.loads((ROOT / "data/boundary-protocol-v2.json").read_text(encoding="utf-8"))

    def test_composition(self):
        self.assertEqual(sum(r["split"] == "train" for r in self.data["rows"]), 6)
        self.assertEqual(sum(r["split"] == "dev" for r in self.data["rows"]), 9)
        original = json.loads((ROOT / "data/pilot-v1.json").read_text(encoding="utf-8"))
        self.assertEqual(self.data["rows"][:6], original["rows"])
        self.assertEqual(self.data["instruction"], original["instruction"])

    def test_fixed_protocol(self):
        validate_protocol(self.data, self.protocol)
        self.assertEqual(self.protocol["steps_per_arm"], 6)
        self.assertEqual(self.protocol["control_order"][:3], self.protocol["candidate_order"][:3])

    def test_eval_cannot_train(self):
        self.protocol["candidate_order"][0] = "dev-03"
        with self.assertRaisesRegex(ValueError, "non-training"):
            validate_protocol(self.data, self.protocol)

    def test_group_leak_rejected(self):
        data = copy.deepcopy(self.data)
        data["rows"][-1]["group"] = data["rows"][0]["group"]
        with self.assertRaisesRegex(ValueError, "crosses"):
            validate_pilot_data(data)

    def test_budget_change_rejected(self):
        self.protocol["candidate_order"].append("train-01")
        with self.assertRaisesRegex(ValueError, "budget"):
            validate_protocol(self.data, self.protocol)

    def test_evaluation_order_rejected(self):
        self.protocol["dev_ids"].reverse()
        with self.assertRaisesRegex(ValueError, "identities/order"):
            validate_protocol(self.data, self.protocol)


if __name__ == "__main__":
    unittest.main()
