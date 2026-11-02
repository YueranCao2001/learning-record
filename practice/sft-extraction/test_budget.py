"""Fixed mixed-sequence budget and saved endpoint evidence checks."""
import copy
import hashlib
import math
import unittest
from budget_experiment import extended_configuration
from prompt_diagnostic import ROOT, read
from review_budget import review, rows_for_evaluation


class BudgetProtocolTests(unittest.TestCase):
    def test_repeated_schedule_keeps_lr_and_original_unchanged(self):
        source = read(ROOT / "data/repair-protocol-v1.json")
        protocol = read(ROOT / "data/optimization-budget-v1.json")
        before = copy.deepcopy(source)
        extended = extended_configuration(source, protocol)
        self.assertEqual(extended["repair_order"], source["repair_order"] * 3)
        self.assertEqual(extended["planned_steps_per_arm"], 36)
        self.assertEqual(extended["optimizer"], source["optimizer"])
        self.assertEqual(source, before)

    def test_inconsistent_budget_rejected(self):
        source = read(ROOT / "data/repair-protocol-v1.json")
        protocol = read(ROOT / "data/optimization-budget-v1.json")
        protocol["extended_steps"] = 35
        with self.assertRaises(ValueError): extended_configuration(source, protocol)

    def test_evaluation_cohorts_and_source_training_schedule(self):
        data = read(ROOT / "results/2026-10-30-repair-comparison-v1/data.json")
        rows = rows_for_evaluation(data)
        self.assertEqual(len(rows), 63)
        self.assertEqual(len({r["id"] for r in rows}), 63)
        self.assertEqual([r["cohort"] for r in rows[:6]], ["new-train"] * 6)
        source = read(ROOT / "data/repair-protocol-v1.json")
        allowed = {r["id"] for r in data["rows"] if r["split"] == "train"}
        self.assertTrue(set(source["repair_order"]) <= allowed)


class SavedBudgetTests(unittest.TestCase):
    folder = ROOT / "results/2026-11-02-optimization-budget-v1"

    def test_review_scores_nll_and_pairwise_changes_recompute(self):
        self.assertEqual(review(self.folder), read(self.folder / "review.json"))

    def test_sources_data_and_configuration_unchanged(self):
        manifest = read(self.folder / "manifest.json")
        for name, sha in manifest["source_sha256"].items():
            self.assertEqual(hashlib.sha256((ROOT / name).read_bytes()).hexdigest(), sha, name)
        self.assertEqual(hashlib.sha256((self.folder / "data.json").read_bytes()).hexdigest(), manifest["data_sha256"])
        self.assertEqual(read(self.folder / "training-configuration.json"), extended_configuration(
            read(self.folder / "source-protocol.json"), read(self.folder / "protocol.json")))

    def test_finite_thirty_six_updates_and_source_prefix(self):
        history = read(self.folder / "extended/training.json")
        config = read(self.folder / "training-configuration.json")
        self.assertEqual(len(history), 36)
        self.assertEqual([r["id"] for r in history], config["repair_order"])
        self.assertTrue(all(math.isfinite(r["loss"]) and math.isfinite(r["grad_norm_before_clip"]) for r in history))
        self.assertTrue(all(r["lr"] == config["optimizer"]["lr"] for r in history))
        self.assertEqual(sum(r["target_tokens"] for r in history), 834)
        prior = ROOT / "results/2026-10-30-repair-comparison-v1"
        self.assertEqual(history[:12], read(prior / "repair/training.json"))
        summary = read(self.folder / "extended/summary.json")
        self.assertEqual(summary["optimizer_state_entries_before"], 0)
        self.assertEqual(summary["optimizer_final_steps"], [36])
        self.assertEqual(summary["frozen_base_sha256_before"], summary["frozen_base_sha256_after"])
        self.assertEqual(summary["initial_adapter_sha256"], read(prior / "repair/summary.json")["initial_adapter_sha256"])

    def test_replay_reload_and_all_counts(self):
        prior = ROOT / "results/2026-10-30-repair-comparison-v1"
        expected = {o["id"]: o for o in read(prior / "repair/after.json") + read(ROOT / "results/2026-11-01-training-fit-v1/repair-generation.json")}
        for output in read(self.folder / "short/after.json"):
            self.assertEqual(output, expected[output["id"]])
        self.assertEqual(read(self.folder / "extended/after.json"), read(self.folder / "extended/reloaded.json"))
        result = read(self.folder / "review.json")
        self.assertEqual(result["training_steps"], 36)
        self.assertEqual(result["total_generation_calls"], 189)
        self.assertEqual(result["teacher_forced_forwards"], 12)
        self.assertTrue(result["short_tokens_reproduce"] and result["extended_reload_exact"])
        for cohorts in result["summaries"].values():
            self.assertEqual({k: v["count"] for k, v in cohorts.items()}, {"new-train": 6, "new-dev": 12, "historical-regression": 45})

    def test_published_documents_and_original_weights_preserved(self):
        upload = read(self.folder / "hf-audit-upload.json")
        self.assertTrue(upload["private"] and upload["download_hashes_equal"] and upload["adapter_weights_unchanged"])
        self.assertFalse(upload["weight_upload"])
        self.assertIsNone(upload["tested_adapter_revision"])
        self.assertEqual(upload["main_before"], upload["main_after"])
        self.assertEqual(len(upload["files_sha256"]), 16)
        for name, sha in upload["files_sha256"].items():
            self.assertTrue(name.endswith(".json"))
            self.assertEqual(hashlib.sha256((self.folder / name).read_bytes()).hexdigest(), sha, name)


if __name__ == "__main__":
    unittest.main()
