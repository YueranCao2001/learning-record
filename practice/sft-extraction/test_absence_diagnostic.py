"""Check surface interventions, unchanged labels and saved read-only evidence."""
import hashlib
import unittest
from absence_diagnostic import compose, prior_outputs, review
from prompt_diagnostic import ROOT, read


class AbsenceDesignTests(unittest.TestCase):
    def test_declared_single_factor_changes_preserve_other_fields(self):
        protocol, _, rows, _ = compose()
        for spec in protocol["anchors"]:
            group = {r["variant"]: r for r in rows if r["source_id"] == spec["source_id"]}
            anchor = group["anchor"]
            old = spec["cue"] + spec["status"]
            expected_clauses = {"cue-only": protocol["canonical_cue"] + spec["status"],
                                "status-only": spec["cue"] + protocol["canonical_status"],
                                "cue-and-status": protocol["canonical_cue"] + protocol["canonical_status"]}
            for variant, clause in expected_clauses.items():
                self.assertEqual(group[variant]["text"], anchor["text"].replace(old, clause))
                self.assertEqual(group[variant]["expected"], anchor["expected"])
            for row in group.values():
                self.assertEqual(row["expected"]["event"], anchor["expected"]["event"])
                self.assertEqual(row["expected"]["time"], anchor["expected"]["time"])

    def test_positive_controls_require_values_and_negative_variants_null(self):
        _, _, rows, queue = compose()
        self.assertEqual(sum(r["expected"]["location"] is None for r in rows), 10)
        for row in rows:
            if row["variant"] == "explicit-value":
                self.assertIsNotNone(row["expected"]["location"])
                self.assertIn(row["expected"]["location"], row["text"])
        self.assertEqual(len(queue), 2)
        self.assertTrue(all(r["human_review"] == "pending" for r in queue))

    def test_known_replays_are_original_inputs_and_targets(self):
        protocol, _, rows, _ = compose()
        source = read(ROOT / "results" / protocol["source_run"] / "data.json")
        lookup = {r["id"]: r for r in source["rows"]}
        known = [r for r in rows if r["replay_source_id"]]
        self.assertEqual(len(known), 4)
        for row in known:
            self.assertEqual(row["text"], lookup[row["replay_source_id"]]["text"])
            self.assertEqual(row["expected"], lookup[row["replay_source_id"]]["expected"])


class SavedAbsenceTests(unittest.TestCase):
    folder = ROOT / "results/2026-11-02-absence-diagnostic-v1"

    def test_raw_scores_matrix_and_all_known_tokens_recompute(self):
        self.assertEqual(review(self.folder), read(self.folder / "review.json"))
        rows = read(self.folder / "cases.json")
        for condition in ("baseline", "short", "extended"):
            old = prior_outputs(condition)
            for row, actual in zip(rows, read(self.folder / f"{condition}-generation.json")):
                if row["replay_source_id"]:
                    self.assertEqual(actual["token_ids"], old[row["replay_source_id"]]["token_ids"])

    def test_sources_cases_and_pending_review_queue_unchanged(self):
        protocol, _, rows, queue = compose()
        self.assertEqual(read(self.folder / "protocol.json"), protocol)
        self.assertEqual(read(self.folder / "cases.json"), rows)
        self.assertEqual(read(self.folder / "label-review.json"), queue)
        for name, sha in read(self.folder / "manifest.json")["source_sha256"].items():
            self.assertEqual(hashlib.sha256((ROOT / name).read_bytes()).hexdigest(), sha, name)

    def test_zero_updates_and_three_unchanged_models(self):
        result = read(self.folder / "review.json")
        self.assertEqual(result["training_steps"], 0)
        self.assertEqual(result["total_generation_calls"], 36)
        self.assertEqual(result["source_groups"], 2)
        self.assertEqual(result["new_independent_sources"], 0)
        self.assertTrue(result["all_known_tokens_reproduce"])
        for condition in ("baseline", "short", "extended"):
            weights = read(self.folder / f"{condition}-weights.json")
            self.assertEqual(weights["before"], weights["after"])
            self.assertTrue(weights["unchanged"])

    def test_published_documents_and_model_scope(self):
        upload = read(self.folder / "hf-audit-upload.json")
        self.assertTrue(upload["private"] and upload["download_hashes_equal"] and upload["adapter_weights_unchanged"])
        self.assertFalse(upload["weight_upload"])
        self.assertIsNone(upload["tested_adapter_revision"])
        self.assertEqual(upload["main_before"], upload["main_after"])
        self.assertEqual(len(upload["files_sha256"]), 12)
        for name, sha in upload["files_sha256"].items():
            self.assertTrue(name.endswith(".json"))
            self.assertEqual(hashlib.sha256((self.folder / name).read_bytes()).hexdigest(), sha, name)


if __name__ == "__main__":
    unittest.main()
