import unittest

from src.ocr_baseline.fields import extract_address, extract_date, extract_name


class ExtractDateTest(unittest.TestCase):
    def test_iso_format(self):
        self.assertEqual(extract_date("DATE OF BIRTH\n1965-02-11\n"), "1965-02-11")

    def test_dotted_day_month_year(self):
        self.assertEqual(extract_date("04 DATUM KAROZENI\n22.06.1969"), "1969-06-22")

    def test_text_month(self):
        self.assertEqual(extract_date("BORN 22 JUN 1969"), "1969-06-22")

    def test_rejects_impossible_date(self):
        self.assertEqual(extract_date("991332"), "")

    def test_no_date_returns_empty(self):
        self.assertEqual(extract_date("NO DATE HERE"), "")


class ExtractNameAddressTest(unittest.TestCase):
    def test_name_picks_longest_letters_only_line(self):
        text = "MALAYSIA\nHAMZAH BIN KAMMAPU\nISLAM"
        self.assertEqual(extract_name(text), "HAMZAH BIN KAMMAPU")

    def test_address_needs_a_street_or_postcode_cue(self):
        text = "HAMZAH BIN KAMMAPU\nPT 1160 P, JALAN KENANGA, 20400 KUALA TERENGGANU"
        self.assertIn("JALAN KENANGA", extract_address(text))

    def test_address_empty_when_no_cue(self):
        self.assertEqual(extract_address("HAMZAH BIN KAMMAPU\nMALAYSIA"), "")


if __name__ == "__main__":
    unittest.main()
