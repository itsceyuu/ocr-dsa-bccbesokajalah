import unittest

from src.ocr_baseline.mykad import extract_birth_date, parse_mykad


class MyKadParserTest(unittest.TestCase):
    def test_clean_printed_number(self):
        candidate = parse_mykad("001230-11-0470")
        self.assertIsNotNone(candidate)
        self.assertEqual(candidate.number, "001230110470")
        self.assertEqual(candidate.birth_date, "2000-12-30")
        self.assertTrue(candidate.full_number)

    def test_common_ocr_confusions_are_repaired_inside_candidate(self):
        candidate = parse_mykad("O01230-I1-O47O")
        self.assertIsNotNone(candidate)
        self.assertEqual(candidate.number, "001230110470")

    def test_invalid_calendar_date_is_rejected(self):
        self.assertIsNone(parse_mykad("991332-01-1234"))

    def test_date_only_fallback(self):
        candidate = parse_mykad("DOB: 801025")
        self.assertIsNotNone(candidate)
        self.assertEqual(candidate.birth_date, "1980-10-25")
        self.assertFalse(candidate.full_number)

    def test_full_number_outranks_date_only_candidate(self):
        candidate = parse_mykad("DOB 801025; ID 801025-14-5127")
        self.assertIsNotNone(candidate)
        self.assertEqual(candidate.number, "801025145127")

    def test_birth_date_helper(self):
        self.assertEqual(extract_birth_date("MyKad 801025-14-5127"), "1980-10-25")


if __name__ == "__main__":
    unittest.main()

