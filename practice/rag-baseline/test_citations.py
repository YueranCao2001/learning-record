import copy
import json
import unittest
from dataclasses import replace
from pathlib import Path

from citations import assemble, check_quote, full_quote, load_snapshots, make_evidence
from citation_demo import report
from retrieval import words
from run import validate

HERE = Path(__file__).parent


class CitationTests(unittest.TestCase):
    def setUp(self):
        data = json.loads((HERE / "data/baseline-v1.json").read_text(encoding="utf-8"))
        source = json.loads((HERE / "data/source-snapshots-v1.json").read_text(encoding="utf-8"))
        self.docs = validate(data)
        self.snapshots = load_snapshots(source)
        self.versions = {d.id: "v1" for d in self.docs}
        self.packet = assemble(self.docs, "普通设备退货", "alice", words, "overlap",
                               self.snapshots, self.versions)
        self.allowed = {"rule", "exception", "warranty"}
        self.good = full_quote(self.packet[0])

    def test_demo(self):
        result = report()
        self.assertEqual(result["cases_matched_expectation"], 10)
        self.assertEqual(result["provenance_pass_count"], 2)
        self.assertFalse(result["semantic_support_evaluated"])

    def test_unicode_partial_quote(self):
        quote = dict(self.good, start=2, end=6, quote=self.packet[0].text[2:6])
        self.assertTrue(check_quote(quote, self.packet, self.snapshots, self.allowed))

    def test_boolean_and_empty_offsets(self):
        for quote in (dict(self.good, start=True), dict(self.good, end=False),
                      dict(self.good, end=0), dict(self.good, start=-1)):
            with self.assertRaises(ValueError):
                check_quote(quote, self.packet, self.snapshots, self.allowed)

    def test_missing_snapshot(self):
        with self.assertRaises(ValueError):
            check_quote(self.good, self.packet, {}, self.allowed)

    def test_duplicate_and_modified_evidence(self):
        with self.assertRaises(ValueError):
            check_quote(self.good, self.packet + (self.packet[0],), self.snapshots, self.allowed)
        changed = replace(self.packet[0], text="假的文本")
        with self.assertRaises(ValueError):
            check_quote(self.good, (changed,), self.snapshots, self.allowed)

    def test_assembly_version_binding(self):
        versions = dict(self.versions, rule="v2")
        with self.assertRaises(ValueError):
            assemble(self.docs, "普通设备退货", "alice", words, "overlap", self.snapshots, versions)
        changed = copy.deepcopy(self.snapshots)
        changed[("rule", "v1")] = "新文本"
        with self.assertRaises(ValueError):
            assemble(self.docs, "普通设备退货", "alice", words, "overlap", changed, self.versions)

    def test_empty_packet(self):
        self.assertEqual(assemble(self.docs, "退货", "unknown", words, "overlap",
                                  self.snapshots, self.versions), ())
        with self.assertRaises(ValueError):
            check_quote(self.good, (), self.snapshots, self.allowed)

    def test_quote_within_subchunk(self):
        text = self.snapshots[("rule", "v1")]
        part = make_evidence("rule", "v1", text, 2, 6)
        self.assertTrue(check_quote(full_quote(part), (part,), self.snapshots, self.allowed))
        with self.assertRaises(ValueError):
            check_quote(dict(full_quote(part), start=0, quote=text[:6]),
                        (part,), self.snapshots, self.allowed)

    def test_duplicate_snapshot(self):
        source = {"schema_version": 1, "synthetic": True,
                  "sources": [{"source_id": "x", "version": "v1", "text": "a"}] * 2}
        with self.assertRaises(ValueError):
            load_snapshots(source)


if __name__ == "__main__":
    unittest.main()
