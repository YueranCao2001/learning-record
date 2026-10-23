"""Offline integrity checks on the completed fixed-budget experiment."""
import hashlib
import json
import math
from pathlib import Path
import unittest
from review_boundary import review

ROOT = Path(__file__).resolve().parent
RUN = ROOT / "results/2026-10-23-boundary-v2"


def read(path):
    return json.loads((RUN / path).read_text(encoding="utf-8"))


class BoundarySavedTests(unittest.TestCase):
    def test_fingerprints(self):
        manifest = read("manifest.json")
        for name, expected in manifest["code_sha256"].items():
            self.assertEqual(hashlib.sha256((ROOT / name).read_bytes()).hexdigest(), expected)
        self.assertEqual(hashlib.sha256((RUN / "data.json").read_bytes()).hexdigest(), manifest["data_sha256"])
        protocol = ROOT / "data/boundary-protocol-v2.json"
        self.assertEqual(hashlib.sha256(protocol.read_bytes()).hexdigest(), manifest["protocol_sha256"])

    def test_budget_and_freezing(self):
        protocol = read("protocol.json")
        for arm in ("control", "candidate"):
            steps = read(arm + "/training.json")
            summary = read(arm + "/summary.json")
            self.assertEqual([s["id"] for s in steps], protocol[arm + "_order"])
            self.assertEqual(len(steps), 6)
            self.assertTrue(all(math.isfinite(s["loss"]) and math.isfinite(s["grad_norm_before_clip"]) for s in steps))
            self.assertEqual(summary["trained_target_tokens"], sum(s["target_tokens"] for s in steps))
            self.assertEqual(summary["frozen_base_sha256_before"], summary["frozen_base_sha256_after"])
            self.assertGreater(summary["changed_adapter_tensors"], 0)
        self.assertEqual(read("control/summary.json")["initial_adapter_sha256"],
                         read("candidate/summary.json")["initial_adapter_sha256"])

    def test_review_recomputes(self):
        actual = review("2026-10-23-boundary-v2")
        self.assertEqual(actual, read("review.json"))
        self.assertTrue(actual["control_old_tokens_reproduce_yesterday"])
        for arm in ("control", "candidate"):
            self.assertEqual(actual["metrics"]["all"][arm]["exact"], read(arm + "/summary.json")["after_exact"])

    def test_masks_and_reload(self):
        for f in read("token-fixtures.json"):
            count = f["prefix_tokens"]
            self.assertEqual(f["labels"][:count], [-100] * count)
            self.assertEqual(f["labels"][count:], f["input_ids"][count:])
            self.assertEqual(f["labels"][-1], 151645)
        for arm in ("control", "candidate"):
            after = [r["token_ids"] for r in read(arm + "/after.json")]
            self.assertEqual(after, [r["token_ids"] for r in read(arm + "/reloaded.json")])
        self.assertEqual([r["token_ids"] for r in read("candidate/after.json")],
                         [r["token_ids"] for r in read("candidate/remote-reloaded.json")])

    def test_hf_branch_preserved_main(self):
        upload = read("candidate/upload.json")
        self.assertEqual(upload["revision"], "boundary-v2-20261023")
        self.assertTrue(upload["private"] and upload["download_hashes_equal"] and upload["remote_model_reload_verified"])
        self.assertEqual(upload["main_before"], upload["main_after"])
        self.assertEqual(upload["main_after"], "def6e17e1d38ae6d618c72dc55ddaaf961a4fb8f")


if __name__ == "__main__":
    unittest.main()
