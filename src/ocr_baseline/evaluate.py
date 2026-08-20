from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from PIL import Image

from .data import FIELDS, Record, compact_text, levenshtein, normalize_text, normalized_equal
from .engines import BaseEngine
from .fields import parse_fields


def character_error_rate(expected: str, predicted: str) -> float:
    """Compute CER after the same case/whitespace/punctuation normalization."""

    reference = normalize_text(expected)
    hypothesis = normalize_text(predicted)
    if not reference:
        return float(bool(hypothesis))
    return levenshtein(reference, hypothesis) / max(len(reference), 1)


def word_error_rate(expected: str, predicted: str) -> float:
    """Compute token-level WER after normalizing and splitting on whitespace."""

    reference = normalize_text(expected).split()
    hypothesis = normalize_text(predicted).split()
    if not reference:
        return float(bool(hypothesis))
    previous = list(range(len(hypothesis) + 1))
    for index, reference_word in enumerate(reference, start=1):
        current = [index]
        for hypothesis_index, hypothesis_word in enumerate(hypothesis, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[hypothesis_index] + 1,
                    previous[hypothesis_index - 1] + (reference_word != hypothesis_word),
                )
            )
        previous = current
    return previous[-1] / max(len(reference), 1)


@dataclass
class Prediction:
    filename: str
    expected: dict[str, str]
    predicted: dict[str, str]
    raw: dict[str, object]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def predict_record(
    engine: BaseEngine, image_root: Path, record: Record, preprocess: str = "none"
) -> Prediction:
    image = Image.open(image_root / record.filename).convert("RGB")
    if preprocess != "none":
        from . import preprocess as pp

        transform = {
            "grayscale": pp.grayscale_image,
            "perspective": pp.perspective_correct_image,
            "perspective-hough": pp.perspective_correct_image_hough,
            "perspective-ldrnet": pp.perspective_correct_image_ldrnet,
        }.get(preprocess, pp.preprocess_image)
        image = transform(image).convert("RGB")
    ocr_result = engine.predict(image)
    raw_result = dict(ocr_result.raw)
    predicted = parse_fields(str(raw_result.get("full", "")))
    return Prediction(
        filename=record.filename,
        expected={field: getattr(record, field) for field in FIELDS},
        predicted=predicted,
        raw=raw_result,
    )


def score_predictions(predictions: list[Prediction]) -> dict[str, object]:
    field_scores: dict[str, dict[str, float | int]] = {}
    for field in FIELDS:
        exact = 0
        nonempty_exact = 0
        nonempty_count = 0
        empty_exact = 0
        empty_count = 0
        distances = []
        character_errors = []
        word_errors = []
        for prediction in predictions:
            expected = prediction.expected[field]
            predicted = prediction.predicted[field]
            is_exact = normalized_equal(expected, predicted)
            exact += is_exact
            if expected:
                nonempty_count += 1
                nonempty_exact += is_exact
            else:
                empty_count += 1
                empty_exact += is_exact
            expected_compact = compact_text(expected)
            predicted_compact = compact_text(predicted)
            if not expected_compact:
                distances.append(float(bool(predicted_compact)))
            else:
                distances.append(levenshtein(expected_compact, predicted_compact) / len(expected_compact))
            character_errors.append(character_error_rate(expected, predicted))
            word_errors.append(word_error_rate(expected, predicted))
        field_scores[field] = {
            "exact_accuracy": exact / max(len(predictions), 1),
            "nonempty_expected_exact_accuracy": nonempty_exact / max(nonempty_count, 1),
            "empty_expected_exact_accuracy": empty_exact / max(empty_count, 1),
            "n_nonempty_expected": nonempty_count,
            "mean_normalized_edit_distance": sum(distances) / max(len(distances), 1),
            "character_error_rate": sum(character_errors) / max(len(character_errors), 1),
            "word_error_rate": sum(word_errors) / max(len(word_errors), 1),
            "n": len(predictions),
        }

    exact_all = sum(
        all(normalized_equal(p.expected[field], p.predicted[field]) for field in FIELDS)
        for p in predictions
    )
    return {
        "n": len(predictions),
        "all_fields_exact_accuracy": exact_all / max(len(predictions), 1),
        "fields": field_scores,
    }


def run_evaluation(
    engine: BaseEngine,
    records: list[Record],
    image_root: Path,
    output_path: Path,
    preprocess: str = "none",
) -> dict[str, object]:
    predictions = []
    for index, record in enumerate(records, start=1):
        predictions.append(predict_record(engine, image_root, record, preprocess=preprocess))
        if index == 1 or index % 25 == 0 or index == len(records):
            print(f"processed {index}/{len(records)}", flush=True)

    report = {
        "engine": engine.name,
        "metrics": score_predictions(predictions),
        "predictions": [prediction.to_dict() for prediction in predictions],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report
