#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ocr_baseline.data import read_ground_truth
from ocr_baseline.engines import build_engine
from ocr_baseline.evaluate import run_evaluation


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a tiny OCR + regex baseline")
    parser.add_argument("--engine", choices=["tesseract", "easyocr", "trocr-small-printed"], required=True)
    parser.add_argument("--split", choices=["train", "val", "test", "all"], default="test")
    parser.add_argument("--split-dir", default="data/splits")
    parser.add_argument("--ground-truth", default="data/raw/ground_truth.csv")
    parser.add_argument("--image-root", default="data/raw/images")
    parser.add_argument("--output-dir", default="reports")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--trocr-model", default="microsoft/trocr-small-printed")
    parser.add_argument(
        "--field-parser",
        choices=["baseline", "malaysia-structured"],
        default="baseline",
        help="Use OCR boxes and Malaysian spatial candidates when set to malaysia-structured",
    )
    args = parser.parse_args()

    if args.split == "all":
        records = read_ground_truth(args.ground_truth)
    else:
        split_path = Path(args.split_dir) / f"{args.split}.csv"
        if not split_path.exists():
            raise SystemExit(f"Missing {split_path}; run scripts/make_split.py first")
        records = read_ground_truth(split_path)

    engine = build_engine(args.engine, device=args.device, trocr_model=args.trocr_model)
    parser_suffix = "" if args.field_parser == "baseline" else f"_{args.field_parser.replace('-', '_')}"
    output_path = Path(args.output_dir) / f"{args.engine}{parser_suffix}_{args.split}.json"
    report = run_evaluation(
        engine,
        records,
        Path(args.image_root),
        output_path,
        field_parser=args.field_parser,
    )
    print(json.dumps({"engine": args.engine, "split": args.split, **report["metrics"]}, indent=2))


if __name__ == "__main__":
    main()
