#!/usr/bin/env python3
"""Rescore Baseline 0 (fields.py) against already-saved OCR text.

Useful after a fields.py fix: reruns the (cheap) regex/heuristic parsing
without redoing the (slow) OCR pass, so the comparison to Baseline 1 stays
fair without a 20+ minute re-run.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from ocr_baseline.evaluate import Prediction, score_predictions
from ocr_baseline.fields import parse_fields
from run_baseline import log_experiment


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", required=True, choices=["tesseract", "easyocr", "paddleocr"])
    parser.add_argument("--split", required=True, choices=["train", "val", "test"])
    parser.add_argument("--reports-dir", default="reports")
    parser.add_argument("--method", default="baseline0-regex-v2")
    args = parser.parse_args()

    reports_dir = Path(args.reports_dir)
    report_path = reports_dir / f"{args.engine}_{args.split}.json"
    data = json.loads(report_path.read_text(encoding="utf-8"))

    predictions = [
        Prediction(
            filename=record["filename"],
            expected=record["expected"],
            predicted=parse_fields(str(record["raw"].get("full", ""))),
            raw={},
        )
        for record in data["predictions"]
    ]
    metrics = score_predictions(predictions)
    log_experiment(reports_dir, args.engine, args.split, args.method, metrics)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
