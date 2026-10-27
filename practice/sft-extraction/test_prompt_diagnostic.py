"""Check diagnostic scoring, paired design and persisted evidence without a GPU."""
import copy
import json
import unittest
from prompt_diagnostic import ROOT, compose, metrics, read, review
from contract import score


class PromptDiagnosticTests(unittest.TestCase):
    def test_pair_only_changes_time_clause_and_label(self):
        protocol, _, rows = compose()
        for pair in protocol["pairs"]:
            a, b = [r for r in rows if r.get("pair") == pair["id"]]
            self.assertEqual(a["text"].replace(pair["missing_clause"], pair["present_clause"]), b["text"])
            self.assertEqual({**a["expected"], "time": pair["expected_time"]}, b["expected"])
            self.assertEqual(a["group"], b["group"])
        self.assertEqual(len({r["group"] for r in rows if r["cohort"] == "new-pairs"}), 4)

    def test_order_is_semantic_invariant_but_emitted_order_is_measured(self):
        row = compose()[2][0]
        raw = json.dumps(dict(reversed(list(row["expected"].items()))), ensure_ascii=False)
        summary, records = metrics([row], [{"id": row["id"], "raw": raw, "score": score(raw, row["expected"])}])
        self.assertTrue(records[0]["exact"])
        self.assertEqual(summary["emitted_key_orders"], {"location,time,event": 1})

    def test_missing_location_and_invalid_json_remain_distinct(self):
        rows = compose()[2][:2]
        values = [{**rows[0]["expected"], "location": None}, None]
        outputs = []
        for row, value in zip(rows, values):
            raw = json.dumps(value, ensure_ascii=False) if value is not None else "```json\n{}\n```"
            outputs.append({"id": row["id"], "raw": raw, "score": score(raw, row["expected"])})
        summary, records = metrics(rows, outputs)
        self.assertTrue(records[0]["location_omitted"])
        self.assertFalse(records[1]["location_omitted"])
        self.assertEqual(summary["json_valid"], 1)
        self.assertEqual(summary["old-audit"]["fields_correct"]["location"], 0)

    def test_reordered_outputs_and_forged_scores_rejected(self):
        rows = compose()[2][:2]
        outputs = [{"id": r["id"], "raw": "{}", "score": score("{}", r["expected"])} for r in rows]
        with self.assertRaises(ValueError):
            metrics(rows, outputs[::-1])
        forged = copy.deepcopy(outputs)
        forged[0]["score"]["exact"] = True
        with self.assertRaises(ValueError):
            metrics(rows, forged)


class SavedPromptEvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.folder = ROOT / "results/2026-10-27-prompt-diagnostic-v1"

    def test_raw_scores_and_aggregates_recompute(self):
        self.assertEqual(review(self.folder), read(self.folder / "review.json"))

    def test_manifest_sources_unchanged(self):
        import hashlib
        manifest = read(self.folder / "manifest.json")
        for name, expected in manifest["source_sha256"].items():
            self.assertEqual(hashlib.sha256((ROOT / name).read_bytes()).hexdigest(), expected, name)

    def test_old_tokens_and_weights_unchanged(self):
        report = read(self.folder / "review.json")
        self.assertEqual(report["total_generation_calls"], 120)
        self.assertTrue(all(report["old_audit_tokens_reproduce"].values()))
        for model in ("base", "candidate"):
            weights = read(self.folder / f"{model}-weights.json")
            self.assertTrue(weights["unchanged"])
            self.assertEqual(set(weights["after_each_condition"].values()), {weights["before"]})

    def test_case_prompt_protocol_snapshots_match_definition(self):
        protocol, prompts, rows = compose()
        self.assertEqual(read(self.folder / "protocol.json"), protocol)
        self.assertEqual(read(self.folder / "prompts.json"), prompts)
        self.assertEqual(read(self.folder / "cases.json"), rows)

    def test_published_report_hashes_and_scope(self):
        import hashlib
        upload = read(self.folder / "hf-audit-upload.json")
        self.assertTrue(upload["private"] and upload["download_hashes_equal"] and upload["adapter_weights_unchanged"])
        self.assertFalse(upload["weight_upload"])
        self.assertEqual(upload["main_before"], upload["main_after"])
        self.assertEqual(upload["tested_adapter_revision"], read(self.folder / "protocol.json")["adapter_revision"])
        self.assertEqual(len(upload["files_sha256"]), 13)
        for name, expected in upload["files_sha256"].items():
            self.assertEqual(hashlib.sha256((self.folder / name).read_bytes()).hexdigest(), expected, name)


if __name__ == "__main__":
    unittest.main()
