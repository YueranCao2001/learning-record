import copy
import json
import unittest
from pathlib import Path

from run import evaluate

HERE = Path(__file__).parent
V1 = json.loads((HERE / "data/baseline-v1.json").read_text(encoding="utf-8"))
V2 = json.loads((HERE / "data/baseline-v2.json").read_text(encoding="utf-8"))


def by_config(data, k):
    return {r["config"]: {row["case_id"]: row for row in r["rows"]}
            for r in evaluate(data, k)}


class DiagnosticTests(unittest.TestCase):
    def test_original_rows_unchanged(self):
        for k in (1, 2):
            old, new = by_config(V1, k), by_config(V2, k)
            for config, rows in old.items():
                for case, row in rows.items():
                    self.assertEqual(row, new[config][case])

    def test_predeclared_rank_directions(self):
        rows = by_config(V2, 1)
        expected = {
            "word_overlap": ("a_freq", "a_long", "a_boundary"),
            "word_bm25": ("z_freq", "z_short", "a_boundary"),
            "bigram_bm25": ("z_freq", "z_short", "z_boundary"),
        }
        for config, ids in expected.items():
            for case, id_ in zip(("frequency", "length", "boundary"), ids):
                self.assertEqual(rows[config][case]["hits"][0]["id"], id_)

    def test_top2_covers_both_candidates(self):
        rows = by_config(V2, 2)
        for config in rows.values():
            for case in ("frequency", "length", "boundary"):
                self.assertTrue(config[case]["complete_evidence"])

    def test_reorder_and_reader_isolation(self):
        shuffled = copy.deepcopy(V2)
        shuffled["documents"].reverse()
        self.assertEqual(evaluate(V2, 1), evaluate(shuffled, 1))
        for result in evaluate(V2, 2):
            for row in result["rows"]:
                visible = {d["id"] for d in V2["documents"] if row["user"] in d["readers"]}
                self.assertTrue({h["id"] for h in row["hits"]} <= visible)


if __name__ == "__main__":
    unittest.main()
