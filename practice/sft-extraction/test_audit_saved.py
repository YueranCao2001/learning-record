"""Offline evidence checks, not independent re-evaluation or human review."""
import hashlib
import json
from pathlib import Path
import unittest
from review_audit import review

ROOT = Path(__file__).resolve().parent
RUN = ROOT / "results/2026-10-24-open-audit-v1"


def read(name):
    return json.loads((RUN / name).read_text(encoding="utf-8"))


class AuditSavedTests(unittest.TestCase):
    def test_fingerprints(self):
        manifest = read("manifest.json")
        for name, value in manifest["source_sha256"].items():
            self.assertEqual(hashlib.sha256((ROOT / name).read_bytes()).hexdigest(), value)
        self.assertFalse(manifest["annotation_rules_in_prompt"])
        self.assertEqual(manifest["human_review"], "pending")

    def test_read_only_weights(self):
        for name in ("base", "candidate"):
            weights = read(name + "-weights.json")
            self.assertEqual(weights["before"], weights["after"])
            self.assertTrue(weights["unchanged"])
            self.assertEqual(weights["training_steps"], 0)

    def test_scores_and_call_budget(self):
        actual = review("2026-10-24-open-audit-v1")
        self.assertEqual(actual, read("review.json"))
        self.assertEqual(actual["total_generation_calls"], 52)
        self.assertEqual(actual["training_steps"], 0)
        self.assertFalse(actual["promote_to_main"])
        self.assertTrue(actual["old_candidate_tokens_reproduce"])
        calls = sum(len(read(name + suffix)) for name in ("base", "candidate")
                    for suffix in ("-extraction.json", "-retention.json"))
        self.assertEqual(calls, 52)

    def test_paired_regression_preserved(self):
        error = next(r for r in read("review.json")["paired_errors_or_differences"] if r["id"] == "audit-03")
        self.assertTrue(error["base_score"]["exact"])
        self.assertFalse(error["candidate_score"]["exact"])
        self.assertEqual(error["expected"]["location"], "潮汐教室")

    def test_metadata_only_sync(self):
        report = read("hf-audit-upload.json")
        self.assertTrue(report["download_hashes_equal"] and report["adapter_weights_unchanged"] and report["private"])
        self.assertFalse(report["weight_upload"])
        self.assertEqual(report["main_before"], report["main_after"])
        self.assertEqual(report["main_after"], "def6e17e1d38ae6d618c72dc55ddaaf961a4fb8f")
        self.assertEqual(len(report["files_sha256"]), 9)
        self.assertFalse(any(name.endswith("safetensors") for name in report["files_sha256"]))


if __name__ == "__main__":
    unittest.main()
