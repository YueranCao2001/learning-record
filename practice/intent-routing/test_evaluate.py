"""Hand-counted synthetic runs for aggregate and paired evaluation."""

import json
import unittest
from dataclasses import replace

from evaluate import Submission, compare_paired, score_submissions, summarize


def reply(intent):
    return json.dumps({"intent": intent})


def example_runs():
    labels = (
        "balance", "bill_balance", "bill_due", "pay_bill", "transfer",
        "transactions", "report_fraud", "report_lost_card", "card_declined",
        "credit_limit", "credit_limit_change",
    )
    baseline_changes = {
        "bill_balance": reply("balance"),
        "card_declined": reply("out_of_scope"),
        "credit_limit_change": "",
    }
    baseline = [
        Submission(intent, intent, baseline_changes.get(intent, reply(intent)))
        for intent in labels
    ]
    baseline.extend((
        Submission("official-1", "out_of_scope", reply("out_of_scope")),
        Submission("official-2", "out_of_scope", reply("balance")),
        Submission("near-1", "out_of_scope", reply("out_of_scope"), same_domain_oos=True),
        Submission("near-2", "out_of_scope", "", same_domain_oos=True),
    ))
    candidate_changes = {
        "balance": "",
        "bill_balance": reply("bill_balance"),
        "near-1": "",
        "near-2": reply("out_of_scope"),
    }
    candidate = [
        replace(row, raw_reply=candidate_changes.get(row.sample_id, row.raw_reply))
        for row in reversed(baseline)
    ]
    return baseline, candidate


class EvaluationTests(unittest.TestCase):
    def assert_rate(self, summary, key, numerator, denominator):
        self.assertEqual(summary[key], {
            "numerator": numerator,
            "denominator": denominator,
            "value": numerator / denominator if denominator else None,
        })

    def test_summary_matches_hand_counted_denominators(self):
        baseline, candidate = example_runs()
        before = summarize(score_submissions(baseline))
        after = summarize(score_submissions(candidate))
        self.assert_rate(before, "task_accuracy", 9, 13)
        self.assert_rate(after, "task_accuracy", 9, 13)
        self.assert_rate(before, "format_valid_rate", 12, 13)
        self.assert_rate(after, "format_valid_rate", 11, 13)
        for summary, combined_failures in ((before, 2), (after, 3)):
            self.assert_rate(summary, "official_oos_recall", 1, 2)
            self.assert_rate(summary, "same_domain_oos_pass_rate", 1, 2)
            self.assert_rate(summary, "in_scope_false_reject_rate", 1, 11)
            self.assert_rate(
                summary, "in_scope_false_reject_or_format_error_rate", combined_failures, 11
            )
            self.assertEqual(summary["in_scope_support"], 11)
            self.assertEqual(len(summary["in_scope_per_intent"]), 11)

        self.assertEqual(before["in_scope_per_intent"]["balance"], {
            "support": 1, "true_positive": 1, "false_positive": 1,
            "false_negative": 0, "f1": 2 / 3,
        })
        for missed in ("card_declined", "credit_limit_change"):
            self.assertEqual(before["in_scope_per_intent"][missed]["false_negative"], 1)
        self.assertAlmostEqual(before["in_scope_macro_f1"], 23 / 33)
        self.assertAlmostEqual(after["in_scope_macro_f1"], 8 / 11)

    def test_pairs_match_by_id_even_when_order_changes(self):
        baseline, candidate = example_runs()
        changes = compare_paired(score_submissions(baseline), score_submissions(candidate))
        self.assertEqual(changes["main"], {
            "both_correct": {"count": 8, "sample_ids": [
                "bill_due", "credit_limit", "official-1", "pay_bill", "report_fraud",
                "report_lost_card", "transactions", "transfer",
            ]},
            "both_wrong": {"count": 3, "sample_ids": [
                "card_declined", "credit_limit_change", "official-2",
            ]},
            "wrong_to_correct": {"count": 1, "sample_ids": ["bill_balance"]},
            "correct_to_wrong": {"count": 1, "sample_ids": ["balance"]},
        })
        self.assertEqual(changes["same_domain_oos"]["wrong_to_correct"], {
            "count": 1, "sample_ids": ["near-2"],
        })
        self.assertEqual(changes["same_domain_oos"]["correct_to_wrong"], {
            "count": 1, "sample_ids": ["near-1"],
        })

    def test_rejects_inconsistent_pairs(self):
        row = Submission("row-1", "out_of_scope", reply("out_of_scope"))
        baseline = score_submissions([row])
        for edited, message in (
            (replace(row, sample_id="row-2"), "different sample IDs"),
            (replace(row, expected_intent="balance"), "reference changed"),
            (replace(row, same_domain_oos=True), "slice changed"),
        ):
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                compare_paired(baseline, score_submissions([edited]))
        with self.assertRaisesRegex(ValueError, "different sample IDs"):
            compare_paired(baseline, {})

    def test_rejects_bad_sample_ids_and_invalid_same_domain_label(self):
        row = Submission("row-1", "balance", reply("balance"))
        with self.assertRaisesRegex(ValueError, "Duplicate sample ID"):
            score_submissions([row, row])
        with self.assertRaisesRegex(ValueError, "Empty sample ID"):
            score_submissions([replace(row, sample_id=" ")])
        with self.assertRaisesRegex(ValueError, "Expected out_of_scope"):
            score_submissions([replace(row, same_domain_oos=True)])

    def test_zero_denominators_and_fixed_macro_labels(self):
        empty = summarize(score_submissions([]))
        self.assert_rate(empty, "format_valid_rate", 0, 0)
        self.assert_rate(empty, "official_oos_recall", 0, 0)
        self.assert_rate(empty, "same_domain_oos_pass_rate", 0, 0)
        self.assert_rate(empty, "in_scope_false_reject_rate", 0, 0)
        self.assertIsNone(empty["in_scope_macro_f1"])

        scoped = summarize(score_submissions([Submission("row-1", "balance", reply("balance"))]))
        self.assertAlmostEqual(scoped["in_scope_macro_f1"], 1 / 11)
        self.assertEqual(scoped["in_scope_per_intent"]["credit_limit"]["support"], 0)

        near_only = summarize(score_submissions([
            Submission("near-1", "out_of_scope", reply("out_of_scope"), same_domain_oos=True)
        ]))
        self.assert_rate(near_only, "official_oos_recall", 0, 0)
        self.assert_rate(near_only, "same_domain_oos_pass_rate", 1, 1)


if __name__ == "__main__":
    unittest.main()
