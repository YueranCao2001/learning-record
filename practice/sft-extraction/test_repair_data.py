"""Meaningful leakage negatives and persisted preparation evidence; no GPU/network."""
import copy
import hashlib
import unittest
from collections import Counter
from contract import score
from prompt_diagnostic import ROOT, read
from repair_data import compose, evaluation_rows, historical_rows, rejection_checks, validate


class RepairDataTests(unittest.TestCase):
    def test_inventory_and_group_roles(self):
        data, protocol = compose()
        audit = validate(data, protocol)
        self.assertEqual(audit["counts"], {"old-train": 6, "new-train": 6, "new-dev": 12, "historical-regression": 45})
        self.assertTrue(set(audit["new_train_groups"]).isdisjoint(audit["new_dev_groups"]))
        self.assertFalse(audit["semantic_independence_proven"])
        dev = [r for r in data["rows"] if r["cohort"] == "new-dev"]
        self.assertEqual(Counter(r["state"] for r in dev), {s: 3 for s in protocol["states"]})

    def test_seven_faults_rejected_and_original_untouched(self):
        data, protocol = compose()
        before = copy.deepcopy((data, protocol))
        cases = rejection_checks(data, protocol)
        self.assertEqual(len(cases), 7)
        self.assertTrue(all(c["rejected"] and c["optimizer_updates"] == 0 for c in cases))
        self.assertEqual((data, protocol), before)

    def test_no_dev_or_historical_diagnostic_in_either_schedule(self):
        data, protocol = compose()
        diagnostics = {r["id"] for r in evaluation_rows(data)}
        self.assertEqual(len(diagnostics), 57)
        for arm in ("control", "repair"):
            self.assertFalse(diagnostics & set(protocol[arm + "_order"]))
            self.assertEqual(len(protocol[arm + "_order"]), 12)

    def test_historical_label_mutation_rejected(self):
        data, protocol = compose()
        next(r for r in data["rows"] if r["cohort"] == "historical-regression")["expected"]["event"] = "被修改的活动名"
        with self.assertRaisesRegex(ValueError, "historical-mutation"):
            validate(data, protocol)


class SavedPreparationTests(unittest.TestCase):
    folder = ROOT / "results/2026-10-29-repair-preparation-v1"

    def test_sources_inputs_and_audit_recompute(self):
        data, protocol = compose()
        self.assertEqual(read(self.folder / "data.json"), data)
        self.assertEqual(read(self.folder / "protocol.json"), protocol)
        self.assertEqual(read(self.folder / "split-audit.json"), validate(data, protocol))
        self.assertEqual(read(self.folder / "rejections.json"), rejection_checks(data, protocol))
        manifest = read(self.folder / "manifest.json")
        for name, expected in manifest["source_sha256"].items():
            self.assertEqual(hashlib.sha256((ROOT / name).read_bytes()).hexdigest(), expected, name)
        self.assertEqual(hashlib.sha256((self.folder / "data.json").read_bytes()).hexdigest(), manifest["data_sha256"])

    def test_baseline_scores_and_cohorts_recompute(self):
        rows = evaluation_rows(read(self.folder / "data.json"))
        outputs, review = read(self.folder / "baseline.json"), read(self.folder / "review.json")
        self.assertEqual([r["id"] for r in rows], [o["id"] for o in outputs])
        self.assertEqual(len(outputs), 57)
        for row, output in zip(rows, outputs):
            self.assertEqual(score(output["raw"], row["expected"]), output["score"])
        for cohort, actual in review["cohorts"].items():
            selected = [(r, o) for r, o in zip(rows, outputs) if r["cohort"] == cohort]
            self.assertEqual(actual, {"count": len(selected), "exact": sum(o["score"]["exact"] for _, o in selected),
                "schema_valid": sum(o["score"]["schema_valid"] for _, o in selected),
                "failed_ids": [r["id"] for r, o in selected if not o["score"]["exact"]]})

    def test_token_budgets_and_masks(self):
        protocol = read(self.folder / "protocol.json")
        fixtures = {r["id"]: r for r in read(self.folder / "train-token-fixtures.json")}
        budgets = read(self.folder / "token-budget.json")
        self.assertEqual(len(fixtures), 12)
        for fixture in fixtures.values():
            p = fixture["prefix_tokens"]
            self.assertTrue(all(v == -100 for v in fixture["labels"][:p]))
            self.assertEqual(fixture["labels"][p:], fixture["input_ids"][p:])
            self.assertEqual(fixture["target_tokens"], len(fixture["input_ids"]) - p)
        for arm in ("control", "repair"):
            self.assertEqual(budgets["arms"][arm]["planned_target_tokens"], sum(fixtures[n]["target_tokens"] for n in protocol[arm + "_order"]))
        self.assertEqual(budgets["executed_optimizer_steps"], 0)

    def test_baseline_replay_and_zero_updates(self):
        review = read(self.folder / "review.json")
        self.assertEqual(review["training_steps"], 0)
        self.assertEqual(review["status"], "prepared-not-trained")
        self.assertEqual(len(review["historical_tokens_reproduce"]), 45)
        self.assertTrue(all(review["historical_tokens_reproduce"].values()))
        prior = {}
        for name in ("results/2026-10-23-boundary-v2/candidate/after.json", "results/2026-10-27-prompt-diagnostic-v1/candidate-original.json",
                     "results/2026-10-28-prompt-confirmation-v1/candidate-original.json"):
            prior.update({o["id"]: o for o in read(ROOT / name)})
        for output in read(self.folder / "baseline.json"):
            if output["id"] in prior:
                self.assertEqual(output["token_ids"], prior[output["id"]]["token_ids"])
        weights = read(self.folder / "weights.json")
        self.assertEqual(weights["before"], weights["after"])
        self.assertTrue(weights["unchanged"])

    def test_published_hashes_and_no_data_or_weights_uploaded(self):
        upload = read(self.folder / "hf-audit-upload.json")
        self.assertTrue(upload["private"] and upload["download_hashes_equal"] and upload["adapter_weights_unchanged"])
        self.assertFalse(upload["weight_upload"])
        self.assertEqual(upload["main_before"], upload["main_after"])
        self.assertEqual(len(upload["files_sha256"]), 9)
        self.assertFalse({"data.json", "train-token-fixtures.json"} & upload["files_sha256"].keys())
        for name, expected in upload["files_sha256"].items():
            self.assertEqual(hashlib.sha256((self.folder / name).read_bytes()).hexdigest(), expected, name)


if __name__ == "__main__":
    unittest.main()
