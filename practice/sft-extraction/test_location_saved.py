import hashlib
import json
from pathlib import Path
import unittest
from location_minimal import review

ROOT = Path(__file__).resolve().parent
RUN = ROOT / "results/2026-10-25-location-minimal-v1"


def read(name):
    return json.loads((RUN / name).read_text(encoding="utf-8"))


class LocationSavedTests(unittest.TestCase):
    def test_source_fingerprints(self):
        for name, expected in read("manifest.json")["source_sha256"].items():
            self.assertEqual(hashlib.sha256((ROOT / name).read_bytes()).hexdigest(), expected)

    def test_review_and_source_group(self):
        report = review("2026-10-25-location-minimal-v1")
        self.assertEqual(report, read("review.json"))
        self.assertEqual(report["source_groups"], 1)
        self.assertEqual(report["total_generation_calls"], 12)
        self.assertFalse(report["promote_to_main"])
        self.assertTrue(all(report["anchor_tokens_reproduce"].values()))

    def test_no_parameter_update(self):
        for name in ("base", "candidate"):
            weights = read(name + "-weights.json")
            self.assertEqual(weights["training_steps"], 0)
            self.assertEqual(weights["before"], weights["after"])
            self.assertTrue(weights["unchanged"])

    def test_hf_documents_only(self):
        report = read("hf-audit-upload.json")
        self.assertEqual(len(report["files_sha256"]), 7)
        self.assertFalse(report["weight_upload"])
        self.assertTrue(report["download_hashes_equal"] and report["adapter_weights_unchanged"] and report["private"])
        self.assertEqual(report["main_before"], report["main_after"])
        self.assertEqual(report["tested_adapter_revision"], "e3b24d1115c21cf69feaa5e6e9cf85a98f502c84")


if __name__ == "__main__":
    unittest.main()
