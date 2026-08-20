import json
import tempfile
import unittest
from pathlib import Path

from src.ocr_baseline.line_features import (
    build_char_ngram_model,
    cluster_from_seed,
    compute_boilerplate_frequency,
    ends_with_name_particle,
    find_name_continuation,
    in_cluster,
    main_cluster_bbox,
    name_char_ngram_score,
)


def _block(x0, y0, x1, y1, text="x"):
    return {"text": text, "bbox": [(x0, y0), (x1, y0), (x1, y1), (x0, y1)], "confidence": 1.0}


class MainClusterBboxTest(unittest.TestCase):
    def test_outlier_block_rejected_by_in_cluster(self):
        # 5 blocks clustered together, 1 far outlier (background clutter)
        blocks = [_block(10, 10, 20, 20), _block(12, 30, 22, 40), _block(11, 50, 21, 60),
                  _block(13, 70, 23, 80), _block(10, 90, 20, 100), _block(900, 900, 950, 950)]
        bbox = main_cluster_bbox(blocks)
        in_flags = [in_cluster(b, bbox) for b in blocks]
        self.assertTrue(all(in_flags[:5]))
        self.assertFalse(in_flags[5])


class ClusterFromSeedTest(unittest.TestCase):
    def test_absorbs_close_high_scoring_neighbors_stops_at_gap(self):
        blocks = [_block(0, 0, 50, 10), _block(0, 12, 50, 22), _block(0, 24, 50, 34), _block(0, 200, 50, 210)]
        scores = [0.9, 0.8, 0.7, 0.9]  # last one scores high but is far away
        result = cluster_from_seed(blocks, scores, seed_index=0, image_height=300)
        self.assertEqual(result, [0, 1, 2])


class ComputeBoilerplateFrequencyTest(unittest.TestCase):
    def test_identity_deduped_not_image_deduped(self):
        # Same identity repeated 3x (a retake batch) shouldn't inflate their
        # own name's frequency; a boilerplate word shared across 2 different
        # identities should score higher than a name unique to 1 identity.
        report = {
            "predictions": [
                {"expected": {"name": "SAME PERSON", "birth_date": "2000-01-01"},
                 "raw": {"blocks": [{"text": "MALAYSIA"}, {"text": "SAME PERSON"}]}},
                {"expected": {"name": "SAME PERSON", "birth_date": "2000-01-01"},
                 "raw": {"blocks": [{"text": "MALAYSIA"}, {"text": "SAME PERSON"}]}},
                {"expected": {"name": "SAME PERSON", "birth_date": "2000-01-01"},
                 "raw": {"blocks": [{"text": "MALAYSIA"}, {"text": "SAME PERSON"}]}},
                {"expected": {"name": "OTHER PERSON", "birth_date": "1990-05-05"},
                 "raw": {"blocks": [{"text": "MALAYSIA"}]}},
            ]
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "report.json"
            path.write_text(json.dumps(report))
            freq = compute_boilerplate_frequency(path)
        # 2 identities total; "MALAYSIA" appears for both -> 1.0
        self.assertAlmostEqual(freq["MALAYSIA"], 1.0)
        # "SAME PERSON" appears for only 1 of 2 identities, despite 3 images
        self.assertAlmostEqual(freq["SAMEPERSON"], 0.5)


class FindNameContinuationTest(unittest.TestCase):
    def test_absorbs_adjacent_short_surname_not_distant_address_line(self):
        blocks = [
            _block(0, 0, 100, 10, "MOHD ASWARDIBIN"),   # seed
            _block(0, 12, 100, 22, "AHMAD"),             # true surname, adjacent, short
            _block(0, 24, 100, 34, "KAMPUNG PULAUTEMBUN"),
            _block(0, 200, 100, 210, "SUNGALLIMAUDALAM"),  # far away, would score high but wrong
        ]
        cont = find_name_continuation(blocks, seed_index=0, image_height=300)
        self.assertEqual(cont, 1)

    def test_rejects_long_next_block(self):
        blocks = [
            _block(0, 0, 100, 10, "SEED NAME"),
            _block(0, 12, 100, 22, "THIS BLOCK HAS WAY TOO MANY TOKENS TO BE A SURNAME"),
        ]
        cont = find_name_continuation(blocks, seed_index=0, image_height=300)
        self.assertIsNone(cont)


class EndsWithNameParticleTest(unittest.TestCase):
    def test_true_positives(self):
        for text in ("MOHD ASWARDIBIN", "AHMAD HAZIM BIN", "THEVAN SILEN A/L", "NOR FARAH EMI BINT"):
            self.assertTrue(ends_with_name_particle(text), text)

    def test_true_negatives_glued_or_complete(self):
        # "BINTINAYAN" glues BINTI+NAYAN with no space -- a surname is
        # already appended, this must NOT look like a dangling particle.
        for text in ("TAN CHEE BOON", "ROSLINDA BINTINAYAN", "AFIZAN BIN ZAINAL ABIDIN"):
            self.assertFalse(ends_with_name_particle(text), text)


class CharNgramModelTest(unittest.TestCase):
    def test_scores_name_like_text_above_boilerplate(self):
        # Several distinct "name" blocks (different letters each time) vs a
        # single repeated boilerplate word -- the model should learn the
        # character-pattern difference, not just memorize one string.
        names = ["JOHN SMITH", "MARIA GARCIA", "AHMAD BIN ALI", "WEI CHEN", "PRIYA SHARMA"]
        report = {
            "predictions": [
                {"expected": {"name": name, "birth_date": "2000-01-01"},
                 "raw": {"blocks": [{"text": name}, {"text": "WARGANEGARA"}, {"text": "WARGANEGARA"}]}}
                for name in names
            ]
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "report.json"
            path.write_text(json.dumps(report))
            model = build_char_ngram_model(path)
        self.assertGreater(
            name_char_ngram_score("ROBERT JONES", model),
            name_char_ngram_score("WARGANEGARA", model),
        )

    def test_unseen_ngrams_and_empty_model_are_neutral(self):
        self.assertEqual(name_char_ngram_score("ANYTHING", None), 0.0)
        self.assertEqual(name_char_ngram_score("", {"abc": 5.0}), 0.0)


if __name__ == "__main__":
    unittest.main()
