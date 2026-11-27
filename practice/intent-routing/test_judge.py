"""Independent, synthetic examples for the intent-routing judge."""

import unittest

from judge import grade_reply


class JudgeTests(unittest.TestCase):
    def assert_judgment(self, raw_reply, reference, valid, predicted, correct, reason):
        result = grade_reply(raw_reply, reference)
        self.assertEqual(
            (result.format_valid, result.predicted_intent, result.task_correct, result.format_error),
            (valid, predicted, correct, reason),
        )

    def test_all_contract_labels(self):
        # This list is the test oracle. It is independent of the judge's allowlist.
        for label in (
            "balance", "bill_balance", "bill_due", "pay_bill", "transfer",
            "transactions", "report_fraud", "report_lost_card", "card_declined",
            "credit_limit", "credit_limit_change", "out_of_scope",
        ):
            with self.subTest(label=label):
                self.assert_judgment(f'{{"intent":"{label}"}}', label, True, label, True, None)

    def test_well_formed_replies(self):
        cases = (
            (" \r\n\t{\"intent\":\"balance\"}\t ", "balance", "balance", True),
            (r'{"\u0069ntent":"bal\u0061nce"}', "balance", "balance", True),
            ('{"intent":"bill_balance"}', "balance", "bill_balance", False),
            ('{"intent":"out_of_scope"}', "balance", "out_of_scope", False),
            ('{"intent":"balance"}', "out_of_scope", "balance", False),
        )
        for raw_reply, reference, predicted, correct in cases:
            with self.subTest(raw_reply=raw_reply, reference=reference):
                self.assert_judgment(raw_reply, reference, True, predicted, correct, None)

    def test_invalid_replies(self):
        cases = (
            ("", "empty_reply"),
            (" \t\r\n", "empty_reply"),
            ('```json\n{"intent":"out_of_scope"}\n```', "invalid_json"),
            ('Answer: {"intent":"out_of_scope"}', "invalid_json"),
            ('{"intent":"out_of_scope"} because it is unrelated.', "invalid_json"),
            ('{"intent":"out_of_scope"}{"intent":"out_of_scope"}', "invalid_json"),
            ('{"intent":NaN}', "invalid_json"),
            ('\u00a0{"intent":"out_of_scope"}', "invalid_json"),
            ('{"intent":"out_of_scope",}', "invalid_json"),
            ('[["intent","out_of_scope"]]', "not_object"),
            ('"out_of_scope"', "not_object"),
            ("null", "not_object"),
            ('{"intent":"balance","intent":"out_of_scope"}', "duplicate_key"),
            (r'{"intent":"out_of_scope","\u0069ntent":"out_of_scope"}', "duplicate_key"),
            ('{"intent":"out_of_scope","confidence":1}', "wrong_keys"),
            ("{}", "wrong_keys"),
            ('{"Intent":"out_of_scope"}', "wrong_keys"),
            ('{"intent":false}', "intent_not_string"),
            ('{"intent":{"label":"out_of_scope"}}', "intent_not_string"),
            ('{"intent":"oos"}', "unknown_intent"),
            ('{"intent":"BALANCE"}', "unknown_intent"),
            ('{"intent":"out_of_scope "}', "unknown_intent"),
        )
        # A malformed reply must never be credited as a correct rejection.
        for raw_reply, reason in cases:
            with self.subTest(raw_reply=raw_reply):
                self.assert_judgment(raw_reply, "out_of_scope", False, None, False, reason)

    def test_unknown_reference_fails_the_run(self):
        with self.assertRaisesRegex(ValueError, "Unknown reference intent"):
            grade_reply('{"intent":"out_of_scope"}', "oos")


if __name__ == "__main__":
    unittest.main()
