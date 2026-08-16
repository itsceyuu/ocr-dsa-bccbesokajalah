#!/usr/bin/env python3
"""Recompute metrics from a saved OCR report without rerunning the OCR engine."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ocr_baseline.evaluate import Prediction, score_predictions


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("reports", nargs="+", type=Path)
    args = parser.parse_args()

    for report_path in args.reports:
        report = json.loads(report_path.read_text(encoding="utf-8"))
        predictions = [Prediction(**item) for item in report["predictions"]]
        report["metrics"] = score_predictions(predictions)
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"report": str(report_path), "metrics": report["metrics"]}, indent=2))


if __name__ == "__main__":
    main()
