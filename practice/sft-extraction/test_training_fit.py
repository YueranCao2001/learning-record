"""Next-token alignment, NLL aggregation and saved fitting evidence."""
import hashlib
import math
import unittest
from fit_metrics import first_difference, review, target_statistics
from prompt_diagnostic import ROOT, read


class FitMetricTests(unittest.TestCase):
    def test_mask_shift_includes_eos_and_ignores_prompt_and_last_logits(self):
        import torch
        logits = torch.tensor([[0., 0., 0.], [2., 0., -1.], [-2., 0., 3.], [5., 0., 0.]])
        labels = torch.tensor([-100, -100, 0, 2])
        stats = target_statistics(logits, labels)
        self.assertEqual(stats["positions"], [2, 3])
        self.assertEqual(stats["target_ids"], [0, 2])
        expected = torch.nn.functional.cross_entropy(logits[1:3], torch.tensor([0, 2])).item()
        self.assertAlmostEqual(-sum(stats["target_logps"]) / 2, expected, places=6)
        logits[0] = torch.tensor([99., -99., 99.]); logits[3] = torch.tensor([-99., 99., -99.])
        self.assertEqual(target_statistics(logits, labels), stats)

    def test_empty_supervision_is_rejected(self):
        import torch
        with self.assertRaisesRegex(ValueError, "no supervised"):
            target_statistics(torch.zeros(3, 4), torch.full((3,), -100))

    def test_first_difference_handles_prefix_termination(self):
        self.assertIsNone(first_difference([1, 2], [1, 2]))
        self.assertEqual(first_difference([1, 2], [1, 3]), 1)
        self.assertEqual(first_difference([1, 2], [1]), 1)
        self.assertEqual(first_difference([1], [1, 2]), 1)


class SavedTrainingFitTests(unittest.TestCase):
    folder = ROOT / "results/2026-11-01-training-fit-v1"

    def test_raw_scores_nll_and_differences_recompute(self):
        self.assertEqual(review(self.folder), read(self.folder / "review.json"))

    def test_sources_and_original_training_fixtures_unchanged(self):
        manifest = read(self.folder / "manifest.json")
        for name, sha in manifest["source_sha256"].items():
            self.assertEqual(hashlib.sha256((ROOT / name).read_bytes()).hexdigest(), sha, name)
        prior = {r["id"]: r for r in read(ROOT / "results/2026-10-29-repair-preparation-v1/train-token-fixtures.json")}
        for fixture in read(self.folder / "token-fixtures.json"):
            self.assertEqual(fixture, prior[fixture["id"]])
        self.assertTrue(all(r["split"] == "train" for r in read(self.folder / "cases.json")))

    def test_token_weighted_loss_and_all_three_weights_unchanged(self):
        report = read(self.folder / "review.json")
        self.assertEqual(report["training_steps"], 0)
        self.assertEqual(report["total_generation_calls"], 18)
        self.assertEqual(report["teacher_forced_forwards"], 18)
        for condition, summary in report["summaries"].items():
            self.assertEqual(summary["supervised_tokens"], 142)
            self.assertAlmostEqual(summary["token_weighted_mean_nll"], summary["target_nll_sum"] / 142)
            self.assertTrue(math.isfinite(summary["token_weighted_mean_nll"]))
            weights = read(self.folder / f"{condition}-weights.json")
            self.assertEqual(weights["before"], weights["after"])
            self.assertTrue(weights["unchanged"])

    def test_published_reports_exclude_training_cases_and_weights(self):
        upload = read(self.folder / "hf-audit-upload.json")
        self.assertTrue(upload["private"] and upload["download_hashes_equal"] and upload["adapter_weights_unchanged"])
        self.assertFalse(upload["weight_upload"])
        self.assertIsNone(upload["tested_adapter_revision"])
        self.assertEqual(upload["main_before"], upload["main_after"])
        self.assertEqual(len(upload["files_sha256"]), 13)
        self.assertFalse({"cases.json", "token-fixtures.json"} & upload["files_sha256"].keys())
        for name, sha in upload["files_sha256"].items():
            self.assertEqual(hashlib.sha256((self.folder / name).read_bytes()).hexdigest(), sha, name)


if __name__ == "__main__":
    unittest.main()
