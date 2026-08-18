#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ocr_baseline.data import Record
from ocr_baseline.engines import build_engine
from ocr_baseline.evaluate import run_evaluation


def read_split(path: Path) -> list[Record]:
    with path.open(newline="", encoding="utf-8") as handle:
        return [Record(**row) for row in csv.DictReader(handle)]


def log_experiment(output_dir: Path, engine: str, split: str, method: str, metrics: dict) -> None:
    """Append one line to the run history. No predictions/PII here -- just
    the aggregate numbers -- so unlike the rest of reports/ this file is
    safe to commit and diff across experiments."""

    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "engine": engine,
        "split": split,
        "method": method,
        "metrics": metrics,
    }
    with (output_dir / "experiments.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a zero-shot OCR engine over a split and score it")
    parser.add_argument("--engine", required=True, choices=["tesseract", "easyocr", "paddleocr"])
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--split-dir", default="data/splits")
    parser.add_argument("--image-root", default="data/raw/images")
    parser.add_argument("--output-dir", default="reports")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--method", default="baseline0-regex", help="Tag identifying the extraction approach")
    args = parser.parse_args()

    records = read_split(Path(args.split_dir) / f"{args.split}.csv")
    engine = build_engine(args.engine, device=args.device)
    output_dir = Path(args.output_dir)
    output_path = output_dir / f"{args.engine}_{args.split}.json"
    report = run_evaluation(engine, records, Path(args.image_root), output_path)
    log_experiment(output_dir, args.engine, args.split, args.method, report["metrics"])
    print(json.dumps(report["metrics"], indent=2))


if __name__ == "__main__":
    main()
