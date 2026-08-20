import tempfile
import unittest
from pathlib import Path

from src.ocr_baseline.data import make_group_split, read_ground_truth


class ReadGroundTruthTest(unittest.TestCase):
    def test_handles_double_quoted_export_rows(self):
        # The first 131 rows of the real export wrap each row in an extra
        # pair of quotes, so a plain csv.reader hands back one cell per row
        # instead of four. read_ground_truth must recover all four columns.
        csv_text = (
            "filename,name,birth_date,address\r\n"
            '"image_001.jpg,HAMZAH BIN KAMMAPU,1965-02-11,""PT 1160, JALAN KENANGA"""\r\n'
            "image_002.jpg,PLAIN ROW,1990-01-01,\r\n"
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ground_truth.csv"
            path.write_text(csv_text, encoding="utf-8")
            records = read_ground_truth(path)

        self.assertEqual(len(records), 2)
        self.assertEqual(records[0].name, "HAMZAH BIN KAMMAPU")
        self.assertEqual(records[0].address, "PT 1160, JALAN KENANGA")
        self.assertEqual(records[1].address, "")


class MakeGroupSplitTest(unittest.TestCase):
    def test_same_identity_never_crosses_splits(self):
        from src.ocr_baseline.data import Record

        records = [
            Record(f"img_{i}.jpg", name, "1990-01-01", "")
            for name in ("A", "B", "C", "D", "E")
            for i in range(4)
        ]
        splits = make_group_split(records, seed=42)
        names_by_split = {
            split: {record.name for record in items} for split, items in splits.items()
        }
        all_names = [name for names in names_by_split.values() for name in names]
        self.assertEqual(len(all_names), len(set(all_names)))  # no identity in two splits

    def test_oversized_identity_excluded_from_eval_splits(self):
        from collections import Counter

        from src.ocr_baseline.data import Record

        records = [Record(f"img_{i}.jpg", "A", "1990-01-01", "") for i in range(10)] + [
            Record(f"img_solo_{i}.jpg", f"P{i}", "1991-01-01", "") for i in range(40)
        ]
        splits = make_group_split(records, seed=42, max_images_per_identity_in_eval=2)
        for split_name in ("val", "test"):
            counts = Counter(record.name for record in splits[split_name])
            self.assertTrue(all(count <= 2 for count in counts.values()), counts)
        self.assertIn("A", {record.name for record in splits["train"]})

    def test_reserves_address_bearing_identities_for_train(self):
        from src.ocr_baseline.data import Record

        # One huge no-address batch alone covers train's ~80% row target, so
        # without the reservation every small address-bearing identity would
        # get pushed to eval by the row-count greedy rule.
        records = [Record(f"img_big_{i}.jpg", "BIG", "1990-01-01", "") for i in range(400)] + [
            Record(f"img_small_{i}.jpg", f"P{i}", "1991-01-01", "123 Main St") for i in range(100)
        ]
        splits = make_group_split(records, seed=42, min_train_address_identities=15)
        train_address_rows = sum(1 for r in splits["train"] if r.address)
        self.assertGreaterEqual(train_address_rows, 15)


if __name__ == "__main__":
    unittest.main()
