import json
import unittest

from contract import score
from venue_pairs import DIAGNOSTIC, HISTORY, PROTOCOL, ROOT, prepare, read, summarize


class VenuePairsTests(unittest.TestCase):
    def setUp(self):
        self.protocol = read(ROOT / PROTOCOL)
        self.history = read(ROOT / HISTORY)
        self.diagnostic = read(ROOT / DIAGNOSTIC)

    def prepare(self):
        return prepare(self.protocol, self.history, self.diagnostic)["rows"]

    def test_each_pair_keeps_the_same_event_and_time(self):
        rows = self.prepare()
        self.assertEqual(sum(r["split"] == "train" for r in rows), 8)
        self.assertEqual(sum(r["split"] == "dev" for r in rows), 12)
        for missing, explicit in zip(rows[::2], rows[1::2]):
            self.assertEqual(missing["group"], explicit["group"])
            self.assertEqual(missing["split"], explicit["split"])
            self.assertIsNone(missing["expected"]["location"])
            self.assertEqual(missing["expected"], {**explicit["expected"], "location": None})

    def test_rejects_family_overlap_between_splits(self):
        sources = self.protocol["sources"]
        sources[4]["family"] = sources[0]["family"]
        with self.assertRaisesRegex(ValueError, "duplicate source family"):
            self.prepare()

    def test_rejects_historical_group_overlap(self):
        self.protocol["sources"][0]["id"] = self.history["rows"][0]["group"]
        with self.assertRaisesRegex(ValueError, "historical overlap: group"):
            self.prepare()

    def test_rejects_missing_coverage_cell(self):
        self.protocol["sources"].pop(0)
        with self.assertRaisesRegex(ValueError, "incomplete or repeated coverage"):
            self.prepare()

    def test_all_null_fails_the_explicit_half_of_each_pair(self):
        rows = [r for r in self.prepare() if r["split"] == "dev"]
        outputs = []
        for row in rows:
            raw = json.dumps({**row["expected"], "location": None}, ensure_ascii=False)
            outputs.append({"id": row["id"], "raw": raw, "score": score(raw, row["expected"])})
        summary = summarize(rows, outputs)["summary"]
        self.assertEqual(summary["location_by_variant"], {"missing": 6, "explicit": 0})
        self.assertEqual(summary["both_location_correct"], 0)
