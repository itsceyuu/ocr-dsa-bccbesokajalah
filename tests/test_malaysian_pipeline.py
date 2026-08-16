import unittest

from src.ocr_baseline.engines import OCRBlock
from src.ocr_baseline.malaysian_pipeline import extract_malaysian_fields, group_into_lines


def block(text: str, y: float, confidence: float = 0.9) -> OCRBlock:
    return OCRBlock(
        text=text,
        bbox=((10.0, y), (300.0, y), (300.0, y + 16.0), (10.0, y + 16.0)),
        confidence=confidence,
    )


class MalaysianPipelineTest(unittest.TestCase):
    def test_boxes_are_grouped_into_reading_order_lines(self):
        blocks = [block("ALI", 130), block("BIN", 130), block("001230-11-0470", 100)]
        lines = group_into_lines(blocks)
        self.assertEqual([line.text for line in lines], ["001230-11-0470", "ALI BIN"])

    def test_spatial_candidates_extract_malaysian_fields(self):
        blocks = [
            block("001230-11-0470", 100),
            block("ALI BIN TEST", 130),
            block("NO 1", 160),
            block("JALAN TEST 50000 KUALA LUMPUR ISLAM WARGANEGARA", 190),
            block("WARGANEGARA", 280),
        ]
        predicted, diagnostics = extract_malaysian_fields(blocks, {"full": ""})
        self.assertEqual(predicted["name"], "ALI BIN TEST")
        self.assertEqual(predicted["birth_date"], "2000-12-30")
        self.assertEqual(
            predicted["address"],
            "NO 1 JALAN TEST 50000 KUALA LUMPUR",
        )
        self.assertEqual(diagnostics["id_anchor"]["line_index"], 0)
