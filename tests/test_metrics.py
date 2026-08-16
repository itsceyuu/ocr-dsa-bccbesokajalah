import unittest

from src.ocr_baseline.evaluate import character_error_rate, word_error_rate


class OCRMetricTest(unittest.TestCase):
    def test_exact_text_has_zero_error(self):
        self.assertEqual(character_error_rate("Ali Bin Abu", "ALI BIN ABU"), 0.0)
        self.assertEqual(word_error_rate("Ali Bin Abu", "ALI BIN ABU"), 0.0)

    def test_character_and_word_errors_are_normalized(self):
        self.assertAlmostEqual(character_error_rate("ABCD", "ABXD"), 0.25)
        self.assertAlmostEqual(word_error_rate("NO 12 JALAN", "NO 13 JALAN"), 1 / 3)

    def test_empty_reference_counts_insertions(self):
        self.assertEqual(character_error_rate("", "OCR"), 1.0)
        self.assertEqual(word_error_rate("", "OCR TEXT"), 1.0)
