import json
from pathlib import Path
import unittest
from unittest.mock import patch
from location_minimal import compose

ROOT = Path(__file__).resolve().parent


class LocationTests(unittest.TestCase):
    def test_one_source_six_conditions(self):
        rows = compose()
        self.assertEqual(len(rows), 6)
        self.assertEqual({r["group"] for r in rows}, {"notice-18"})
        self.assertEqual({r["parent_id"] for r in rows}, {"audit-03"})

    def test_targets_preserved_except_time_present(self):
        rows = compose()
        self.assertEqual({r["expected"]["location"] for r in rows}, {"潮汐教室"})
        self.assertEqual({r["expected"]["event"] for r in rows}, {"地图绘制体验课"})
        self.assertEqual([r["expected"]["time"] for r in rows], [None, None, "11月20日14:00", None, None, None])

    def test_clause_swap_keeps_other_words(self):
        rows = compose()
        self.assertEqual(rows[0]["text"].replace("场地设在潮汐教室，开课时间另行公布", "开课时间另行公布，场地设在潮汐教室"), rows[3]["text"])

    def test_undeclared_change_rejected(self):
        source = json.loads((ROOT / "data/location-minimal-v1.json").read_text(encoding="utf-8"))
        source["variants"][1]["location_clause"] = "地点是潮汐教室"
        with patch.object(Path, "read_text", return_value=json.dumps(source)):
            with self.assertRaisesRegex(ValueError, "undeclared"):
                compose()


if __name__ == "__main__":
    unittest.main()
