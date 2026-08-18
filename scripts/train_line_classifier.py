#!/usr/bin/env python3
"""Baseline 1: fit a per-line classifier on train, evaluate on val/test.

Reuses the OCR already saved by run_baseline.py (reports/<engine>_<split>.json)
instead of re-running OCR. The classifier finds *which line* holds a field;
fields.extract_date still does the format normalization for birth_date.
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression

from ocr_baseline.data import normalize_text
from ocr_baseline.evaluate import Prediction, score_predictions
from ocr_baseline.fields import extract_date
from ocr_baseline.line_features import build_table, featurize_image_blocks, label_block


def fit(train_report: Path, image_root: Path) -> tuple[DictVectorizer, LogisticRegression]:
    features, labels, _ = build_table(train_report, image_root)
    vectorizer = DictVectorizer(sparse=False)
    X = vectorizer.fit_transform(features)
    model = LogisticRegression(max_iter=2000, class_weight="balanced")
    model.fit(X, labels)
    return vectorizer, model


def predict_fields(
    blocks: list[dict],
    image_size: tuple[int, int],
    vectorizer: DictVectorizer,
    model: LogisticRegression,
    address_threshold: float = 0.5,
) -> dict[str, str]:
    if not blocks:
        return {"name": "", "birth_date": "", "address": ""}

    feats = featurize_image_blocks(blocks, image_size)
    X = vectorizer.transform(feats)
    proba = model.predict_proba(X)
    classes = list(model.classes_)

    def class_proba(row: int, label: str) -> float:
        return proba[row][classes.index(label)] if label in classes else 0.0

    best_name, best_name_score = "", -1.0
    best_date_text, best_date_score = "", -1.0
    address_lines: list[tuple[int, str]] = []
    for i, block in enumerate(blocks):
        name_score = class_proba(i, "name")
        if name_score > best_name_score:
            best_name, best_name_score = block["text"], name_score
        date_score = class_proba(i, "birth_date")
        if date_score > best_date_score:
            best_date_text, best_date_score = block["text"], date_score
        if class_proba(i, "address") >= address_threshold:
            address_lines.append((i, block["text"]))

    address_lines.sort(key=lambda item: item[0])
    return {
        "name": normalize_text(best_name) if best_name_score > 0 else "",
        "birth_date": extract_date(best_date_text) if best_date_score > 0 else "",
        "address": normalize_text(", ".join(text for _, text in address_lines)),
    }


def evaluate_report(
    report_path: Path, image_root: Path, vectorizer: DictVectorizer, model: LogisticRegression
) -> dict:
    from PIL import Image

    data = json.loads(report_path.read_text(encoding="utf-8"))
    predictions = []
    for record in data["predictions"]:
        blocks = record["raw"].get("blocks") or []
        with Image.open(image_root / record["filename"]) as image:
            size = image.size
        predicted = predict_fields(blocks, size, vectorizer, model)
        predictions.append(
            Prediction(filename=record["filename"], expected=record["expected"], predicted=predicted, raw={})
        )
    return score_predictions(predictions)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train + evaluate the Baseline 1 line classifier")
    parser.add_argument("--engine", default="paddleocr", help="Which saved OCR run to build features from")
    parser.add_argument("--reports-dir", default="reports")
    parser.add_argument("--image-root", default="data/raw/images")
    parser.add_argument("--model-out", default="reports/line_classifier.pkl")
    args = parser.parse_args()

    reports_dir = Path(args.reports_dir)
    image_root = Path(args.image_root)

    train_report = reports_dir / f"{args.engine}_train.json"
    vectorizer, model = fit(train_report, image_root)

    Path(args.model_out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.model_out, "wb") as handle:
        pickle.dump({"vectorizer": vectorizer, "model": model}, handle)

    # Which features push a line toward each class -- cheap interpretability
    # for the report, not just a black box.
    print("Top positive coefficients per class:")
    for class_index, class_name in enumerate(model.classes_):
        coefs = model.coef_[class_index] if len(model.classes_) > 2 else model.coef_[0]
        ranked = sorted(zip(vectorizer.feature_names_, coefs), key=lambda item: -item[1])[:4]
        print(f"  {class_name:12s} {ranked}")
    print()

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from run_baseline import log_experiment

    for split in ("val", "test"):
        report_path = reports_dir / f"{args.engine}_{split}.json"
        if not report_path.exists():
            continue
        metrics = evaluate_report(report_path, image_root, vectorizer, model)
        log_experiment(reports_dir, args.engine, split, "baseline1-line-classifier", metrics)
        print(f"=== {args.engine} / {split} ===")
        print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
