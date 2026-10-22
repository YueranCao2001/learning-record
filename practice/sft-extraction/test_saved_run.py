"""Check saved evidence consistency, not a fresh GPU or Hub acceptance test."""
import hashlib
import json
import math
from pathlib import Path
import unittest
from contract import load_pilot, score

ROOT = Path(__file__).resolve().parent
RUN = ROOT / "results/2026-10-22-lora-smoke-v1"


def read(name):
    return json.loads((RUN / name).read_text(encoding="utf-8"))


class SavedRunTests(unittest.TestCase):
    def test_provenance(self):
        manifest = read("manifest.json")
        for name, path in (("script_sha256", ROOT / "train_smoke.py"),
                           ("data_sha256", ROOT / "data/pilot-v1.json")):
            self.assertEqual(manifest[name], hashlib.sha256(path.read_bytes()).hexdigest())

    def test_training_evidence(self):
        history = read("training.json")
        self.assertEqual([r["id"] for r in history], ["train-01", "train-02", "train-03"] * 2)
        self.assertTrue(all(math.isfinite(r["loss"]) and math.isfinite(r["grad_norm_before_clip"])
                            for r in history))
        summary = read("summary.json")
        self.assertEqual(summary["frozen_base_sha256_before"], summary["frozen_base_sha256_after"])
        self.assertGreater(summary["changed_adapter_tensors"], 0)

    def test_mask_evidence(self):
        for row in read("token-fixtures.json"):
            prefix = row["prefix_tokens"]
            self.assertEqual(row["labels"][:prefix], [-100] * prefix)
            self.assertEqual(row["labels"][prefix:], row["input_ids"][prefix:])
            self.assertEqual(row["input_ids"][-1], 151645)
            self.assertEqual(row["target_tokens"], len(row["input_ids"]) - prefix)
            self.assertLessEqual(len(row["input_ids"]), 512)

    def test_scores_recompute(self):
        expected = {r["id"]: r["expected"] for r in load_pilot()["rows"] if r["split"] == "dev"}
        summary = read("summary.json")
        for name, count in (("baseline.json", "baseline_exact"), ("after.json", "after_exact")):
            rows = read(name)
            self.assertEqual({r["id"] for r in rows}, set(expected))
            for row in rows:
                self.assertEqual(row["score"], score(row["raw"], expected[row["id"]]))
            self.assertEqual(sum(r["score"]["exact"] for r in rows), summary[count])

    def test_reload_evidence(self):
        after = read("after.json")
        for name in ("reloaded.json", "remote-reloaded.json"):
            self.assertEqual([r["token_ids"] for r in read(name)], [r["token_ids"] for r in after])
        upload = read("upload.json")
        self.assertTrue(upload["private"] and upload["download_hashes_equal"]
                        and upload["remote_model_reload_verified"])
        self.assertEqual(upload["commit"], "def6e17e1d38ae6d618c72dc55ddaaf961a4fb8f")


if __name__ == "__main__":
    unittest.main()
