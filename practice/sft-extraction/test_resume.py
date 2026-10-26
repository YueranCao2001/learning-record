import hashlib
import io
import json
from pathlib import Path
import random
import unittest
from unittest.mock import patch
import numpy as np
import torch
from resume_state import cpu_tree, tree_digest, capture_rng, restore_rng

ROOT = Path(__file__).resolve().parent
RUN = ROOT / "results/2026-10-26-exact-resume-v1"


def read(name):
    return json.loads((RUN / name).read_text(encoding="utf-8"))


class ResumeStateTests(unittest.TestCase):
    def test_scalar_and_dictionary_digest(self):
        self.assertEqual(tree_digest({"a": torch.tensor(3.0), "b": 1}), tree_digest({"b": 1, "a": torch.tensor(3.0)}))
        self.assertNotEqual(tree_digest(torch.tensor(3.0)), tree_digest(torch.tensor(3)))

    def test_cpu_tree_clones(self):
        original = {"a": [torch.ones(2)]}
        cloned = cpu_tree(original)
        cloned["a"][0][0] = 9
        self.assertEqual(original["a"][0][0].item(), 1)

    def test_rng_roundtrip(self):
        with patch.object(torch.cuda, "get_rng_state_all", return_value=[]), patch.object(torch.cuda, "set_rng_state_all"):
            original = capture_rng()
            try:
                checkpoint = capture_rng()
                first = (random.random(), float(np.random.random()), torch.rand(1).item())
                restore_rng(checkpoint)
                self.assertEqual(first, (random.random(), float(np.random.random()), torch.rand(1).item()))
            finally:
                restore_rng(original)

    def test_weights_only_container_load(self):
        state = {"tensor": torch.tensor(1.0), "names": ["lora_A"], "step": 2, "options": (0.9, 0.999)}
        buffer = io.BytesIO()
        torch.save(state, buffer)
        buffer.seek(0)
        self.assertEqual(tree_digest(state), tree_digest(torch.load(buffer, weights_only=True)))


class ResumeSavedTests(unittest.TestCase):
    def test_source_fingerprints(self):
        for name, expected in read("manifest.json")["source_sha256"].items():
            self.assertEqual(hashlib.sha256((ROOT / name).read_bytes()).hexdigest(), expected)

    def test_complete_resume(self):
        report = read("comparison.json")
        self.assertTrue(report["full_resume_exact"] and all(report["checks"].values()))
        self.assertEqual(report["full_resume_adapter_max_abs_diff"], 0)
        self.assertTrue(report["same_initial_adapter"] and report["four_distinct_processes"])
        self.assertEqual(read("continuous/history.json"), read("resume/history.json"))
        self.assertEqual(sum(read(name + "/state.json")["executed_steps"] for name in ("continuous", "prefix", "resume", "optimizer-reset")), 10)

    def test_optimizer_reset_diverges(self):
        report = read("comparison.json")
        self.assertTrue(report["optimizer_reset_diverges"] and report["negative_first_resumed_loss_matches"])
        self.assertFalse(report["negative_final_loss_matches"])
        reference, negative = read("continuous/history.json"), read("optimizer-reset/history.json")
        for key in ("id", "lr_before", "lr_after", "numpy_probe", "cpu_probe"):
            self.assertEqual([r[key] for r in reference], [r[key] for r in negative])
        self.assertEqual(read("continuous/state.json")["state_sha256"]["rng"], read("optimizer-reset/state.json")["state_sha256"]["rng"])

    def test_documents_only_publication(self):
        upload = read("hf-audit-upload.json")
        self.assertEqual(len(upload["files_sha256"]), 12)
        self.assertFalse(upload["weight_upload"])
        self.assertTrue(upload["download_hashes_equal"] and upload["adapter_weights_unchanged"] and upload["private"])
        self.assertEqual(upload["main_before"], upload["main_after"])
        self.assertTrue(upload["path_in_repo"].startswith("engineering/"))
        self.assertTrue(all(name.endswith(".json") for name in upload["files_sha256"]))


if __name__ == "__main__":
    unittest.main()
