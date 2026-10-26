import copy
import hashlib
import json
from pathlib import Path
import unittest
from unittest.mock import Mock, patch
from resume_guard import canonical_digest, guarded_load, GuardRejected

ROOT = Path(__file__).resolve().parent
RUN = ROOT / "results/2026-10-26-resume-identity-guard-v1"


def read(name):
    return json.loads((RUN / name).read_text(encoding="utf-8"))


class GuardUnitTests(unittest.TestCase):
    def setUp(self):
        self.identity = {"training_protocol_sha256": "protocol", "data_sha256": "data"}
        self.envelope = {"schema_version": 1, "checkpoint_sha256": "checkpoint", "identity": self.identity,
                         "parameter_order_sha256": canonical_digest(["parameter"])}
        self.state = {"adapter": {}, "optimizer": {}, "scheduler": {}, "rng": {}, "history": [{}, {}],
                      "parameter_names": ["parameter"], "completed_steps": 2, "protocol_sha256": "protocol"}

    @patch("resume_guard.file_digest", return_value="checkpoint")
    def test_valid(self, _):
        loader = Mock(return_value=self.state)
        self.assertIs(guarded_load(self.envelope, "unused", self.identity, loader), self.state)
        loader.assert_called_once()

    @patch("resume_guard.file_digest", return_value="checkpoint")
    def test_mismatch_before_load(self, _):
        loader = Mock()
        actual = dict(self.identity, data_sha256="changed")
        with self.assertRaisesRegex(GuardRejected, "data_sha256"):
            guarded_load(self.envelope, "unused", actual, loader)
        loader.assert_not_called()

    @patch("resume_guard.file_digest", return_value="corrupt")
    def test_bytes_before_load(self, _):
        loader = Mock()
        with self.assertRaisesRegex(GuardRejected, "checkpoint_sha256"):
            guarded_load(self.envelope, "unused", self.identity, loader)
        loader.assert_not_called()

    @patch("resume_guard.file_digest", return_value="checkpoint")
    def test_missing_field_before_load(self, _):
        loader = Mock()
        with self.assertRaisesRegex(GuardRejected, "identity_fields"):
            guarded_load(self.envelope, "unused", {}, loader)
        loader.assert_not_called()

    @patch("resume_guard.file_digest", return_value="checkpoint")
    def test_parameter_order_after_safe_load(self, _):
        wrong = copy.deepcopy(self.state)
        wrong["parameter_names"] = ["another"]
        loader = Mock(return_value=wrong)
        with self.assertRaisesRegex(GuardRejected, "parameter_order"):
            guarded_load(self.envelope, "unused", self.identity, loader)
        loader.assert_called_once()


class GuardSavedTests(unittest.TestCase):
    def test_sources_and_binding(self):
        manifest = read("manifest.json")
        for name, expected in manifest["source_sha256"].items():
            self.assertEqual(hashlib.sha256((ROOT / name).read_bytes()).hexdigest(), expected)
        self.assertEqual(hashlib.sha256((RUN / "binding.json").read_bytes()).hexdigest(), manifest["binding_sha256"])

    def test_eight_preload_rejections(self):
        cases = read("guard-protocol.json")["negative_cases"]
        self.assertEqual(len(cases), 8)
        for name in cases:
            report = read("rejections/" + name + ".json")
            self.assertTrue(report["rejected"])
            self.assertFalse(report["checkpoint_deserialized"] or report["training_dispatched"])
            self.assertEqual(report["optimizer_updates"], 0)

    def test_guarded_gpu_matches_reference(self):
        report = read("comparison.json")
        self.assertTrue(report["valid_resume_exact"] and all(report["checks"].values()))
        self.assertEqual(report["adapter_max_abs_diff"], 0)
        reference = json.loads((ROOT / "results/2026-10-26-exact-resume-v1/continuous/history.json").read_text(encoding="utf-8"))
        self.assertEqual(read("resume/history.json"), reference)
        self.assertTrue(read("valid-guard.json")["training_dispatched"])
        self.assertEqual(report["executed_optimizer_steps"], 4)
        self.assertEqual(sum(read(n + "/state.json")["executed_steps"] for n in ("prefix", "resume")), 4)

    def test_hf_json_only(self):
        report = read("hf-audit-upload.json")
        self.assertEqual(len(report["files_sha256"]), 18)
        self.assertTrue(all(name.endswith(".json") for name in report["files_sha256"]))
        self.assertTrue(report["download_hashes_equal"] and report["adapter_weights_unchanged"] and report["private"])
        self.assertFalse(report["weight_upload"])
        self.assertEqual(report["main_before"], report["main_after"])


if __name__ == "__main__":
    unittest.main()
