"""Paired regressions and saved training/reload evidence, without GPU/network calls."""
import copy
import hashlib
import json
import math
import unittest
from contract import score
from prompt_diagnostic import ROOT, read
from repair_data import evaluation_rows
from review_repair import paired, review


class PairedRepairTests(unittest.TestCase):
    def test_same_total_can_hide_fixed_and_regressed_rows(self):
        target = {"event": "活动", "time": None, "location": None}
        rows = [{"id": n, "cohort": "new-dev", "expected": target} for n in ("a", "b")]
        make = lambda n, right: {"id": n, "raw": json.dumps(target if right else {**target, "event": "错误"}, ensure_ascii=False)}
        first, second = [make("a", False), make("b", True)], [make("a", True), make("b", False)]
        for outputs in (first, second):
            for row, output in zip(rows, outputs): output["score"] = score(output["raw"], row["expected"])
        self.assertEqual(paired(rows, first, second), {"fixed_ids": ["a"], "regressed_ids": ["b"]})

    def test_wrong_order_and_forged_score_rejected(self):
        rows = [{"id": n, "cohort": "new-dev", "expected": {"event": "课", "time": None, "location": None}} for n in ("a", "b")]
        outputs = [{"id": r["id"], "raw": "{}", "score": score("{}", r["expected"])} for r in rows]
        with self.assertRaises(ValueError): paired(rows, outputs, outputs[::-1])
        forged = copy.deepcopy(outputs)
        forged[0]["score"]["exact"] = True
        with self.assertRaises(ValueError): paired(rows, outputs, forged)


class SavedRepairTrainingTests(unittest.TestCase):
    folder = ROOT / "results/2026-10-30-repair-comparison-v1"

    def test_raw_scores_aggregates_and_pairwise_changes_recompute(self):
        self.assertEqual(review(self.folder), read(self.folder / "review.json"))

    def test_sources_data_and_preparation_unchanged(self):
        manifest = read(self.folder / "manifest.json")
        for name, sha in manifest["source_sha256"].items():
            self.assertEqual(hashlib.sha256((ROOT / name).read_bytes()).hexdigest(), sha, name)
        self.assertEqual(hashlib.sha256((self.folder / "data.json").read_bytes()).hexdigest(), manifest["data_sha256"])
        prior = ROOT / "results" / manifest["source_preparation"]
        self.assertEqual(read(self.folder / "data.json"), read(prior / "data.json"))
        self.assertEqual(read(self.folder / "source-protocol.json"), read(prior / "protocol.json"))
        self.assertEqual(hashlib.sha256((prior / "manifest.json").read_bytes()).hexdigest(), manifest["preparation_manifest_sha256"])

    def test_twenty_four_finite_updates_only_on_train_rows(self):
        data, protocol = read(self.folder / "data.json"), read(self.folder / "source-protocol.json")
        allowed = {r["id"] for r in data["rows"] if r["split"] == "train"}
        count = 0
        for arm in ("control", "repair"):
            history = read(self.folder / arm / "training.json")
            self.assertEqual([r["id"] for r in history], protocol[arm + "_order"])
            self.assertEqual(len(history), 12)
            self.assertTrue(all(r["id"] in allowed and math.isfinite(r["loss"]) and math.isfinite(r["grad_norm_before_clip"]) for r in history))
            self.assertTrue(all(r["lr"] == protocol["optimizer"]["lr"] for r in history))
            expected = read(ROOT / "results/2026-10-29-repair-preparation-v1/token-budget.json")["arms"][arm]["planned_target_tokens"]
            self.assertEqual(sum(r["target_tokens"] for r in history), expected)
            count += len(history)
        self.assertEqual(count, 24)

    def test_initialization_frozen_base_and_reload_evidence(self):
        report = read(self.folder / "review.json")
        for key in ("same_initial_adapter", "same_initial_base", "both_base_unchanged", "both_fresh_optimizers", "both_reload_exact", "both_reload_tokens_recomputed_equal"):
            self.assertTrue(report[key], key)
        self.assertEqual(report["total_generation_calls"], 285)
        for arm in ("control", "repair"):
            summary = read(self.folder / arm / "summary.json")
            self.assertEqual(summary["optimizer_final_steps"], [12])
            self.assertGreater(summary["changed_adapter_tensors"], 0)
            self.assertNotEqual(summary["initial_adapter_sha256"], summary["trained_adapter_sha256"])
            self.assertFalse(summary["exact_training_resume_tested"])

    def test_baseline_replay_and_all_raw_reloaded_outputs(self):
        prior = ROOT / "results/2026-10-29-repair-preparation-v1"
        self.assertEqual(read(self.folder / "baseline.json"), read(prior / "baseline.json"))
        rows = evaluation_rows(read(self.folder / "data.json"))
        for arm in ("control", "repair"):
            after, reloaded = read(self.folder / arm / "after.json"), read(self.folder / arm / "reloaded.json")
            self.assertEqual(after, reloaded)
            self.assertEqual([o["id"] for o in after], [r["id"] for r in rows])
            for row, output in zip(rows, reloaded): self.assertEqual(output["score"], score(output["raw"], row["expected"]))

    def test_published_documents_exclude_trained_weights(self):
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
