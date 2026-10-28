"""New-grid design, strict error accounting, and saved confirmation evidence."""
import copy
import hashlib
import json
import unittest
from confirm_prompt import aggregate, compose, review
from prompt_diagnostic import ROOT, read
from contract import score


class ConfirmationContractTests(unittest.TestCase):
    def test_frozen_prompts_and_balanced_four_states(self):
        protocol, prompts, rows = compose()
        prior = read(ROOT / f"results/{protocol['prior_run']}/prompts.json")
        self.assertEqual(prompts, {c: prior[c] for c in protocol["conditions"]})
        for group in protocol["groups"]:
            selected = [r for r in rows if r["group"] == group["group"]]
            self.assertEqual([r["state"] for r in selected], protocol["states"])
            self.assertEqual(len(selected), 4)
            for row in selected:
                self.assertEqual(group["time"] in row["text"], row["expected"]["time"] is not None)
                self.assertEqual(group["location"] in row["text"], row["expected"]["location"] is not None)

    def test_error_directions_are_not_collapsed(self):
        row = compose()[2][4]
        variants = [dict(row["expected"], time=None), dict(row["expected"], time="13:30")]
        for actual, kind in zip(variants, ("value-to-null", "wrong-span")):
            raw = json.dumps(actual, ensure_ascii=False)
            result, _, errors = aggregate([row], [{"id": row["id"], "raw": raw, "score": score(raw, row["expected"])}])
            self.assertEqual(result["error_counts"], {kind: 1})
            self.assertEqual(errors[0]["field"], "time")
        missing = compose()[2][5]
        raw = json.dumps(dict(missing["expected"], time="待定"), ensure_ascii=False)
        self.assertEqual(aggregate([missing], [{"id": missing["id"], "raw": raw, "score": score(raw, missing["expected"])}])[2][0]["kind"], "missing-to-value")

    def test_fenced_correct_content_is_format_failure(self):
        row = compose()[2][4]
        raw = "```json\n" + json.dumps(row["expected"], ensure_ascii=False) + "\n```"
        result, _, errors = aggregate([row], [{"id": row["id"], "raw": raw, "score": score(raw, row["expected"])}])
        self.assertEqual(result["exact"], 0)
        self.assertEqual(result["error_counts"], {"format-or-schema": 1})
        self.assertIsNone(errors[0]["field"])

    def test_reordered_or_forged_outputs_rejected(self):
        rows = compose()[2][4:6]
        outputs = [{"id": r["id"], "raw": "{}", "score": score("{}", r["expected"])} for r in rows]
        with self.assertRaises(ValueError):
            aggregate(rows, outputs[::-1])
        forged = copy.deepcopy(outputs)
        forged[0]["score"]["exact"] = True
        with self.assertRaises(ValueError):
            aggregate(rows, forged)


class SavedConfirmationTests(unittest.TestCase):
    folder = ROOT / "results/2026-10-28-prompt-confirmation-v1"

    def test_results_recompute_and_replay_excluded_from_new_denominator(self):
        actual = review(self.folder)
        self.assertEqual(actual, read(self.folder / "review.json"))
        self.assertEqual(actual["total_generation_calls"], 80)
        self.assertTrue(actual["all_replay_tokens_reproduce"])
        for model in actual["summaries_new_only"].values():
            for summary in model.values():
                self.assertEqual(summary["count"], 16)
                self.assertEqual(sum(x["count"] for x in summary["by_state"].values()), 16)

    def test_sources_and_saved_inputs_unchanged(self):
        protocol, prompts, rows = compose()
        self.assertEqual(read(self.folder / "protocol.json"), protocol)
        self.assertEqual(read(self.folder / "prompts.json"), prompts)
        self.assertEqual(read(self.folder / "cases.json"), rows)
        for name, sha in read(self.folder / "manifest.json")["source_sha256"].items():
            self.assertEqual(hashlib.sha256((ROOT / name).read_bytes()).hexdigest(), sha, name)

    def test_weight_hashes_unchanged_after_each_condition(self):
        for model in ("base", "candidate"):
            weights = read(self.folder / f"{model}-weights.json")
            self.assertTrue(weights["unchanged"])
            self.assertEqual(set(weights["after_each_condition"].values()), {weights["before"]})

    def test_published_scope_and_document_hashes(self):
        upload = read(self.folder / "hf-audit-upload.json")
        self.assertTrue(upload["private"] and upload["download_hashes_equal"] and upload["adapter_weights_unchanged"])
        self.assertFalse(upload["weight_upload"])
        self.assertEqual(upload["main_before"], upload["main_after"])
        self.assertEqual(upload["tested_adapter_revision"], read(self.folder / "protocol.json")["adapter_revision"])
        self.assertEqual(len(upload["files_sha256"]), 11)
        for name, sha in upload["files_sha256"].items():
            self.assertEqual(hashlib.sha256((self.folder / name).read_bytes()).hexdigest(), sha, name)


if __name__ == "__main__":
    unittest.main()
