#!/usr/bin/env python3
"""PII-specialized pretrained NER (mDeBERTa-v3-base, iiiorg/piiranha) as a
field-extraction method, run on already-saved OCR text. Discriminative
token-classifier, not a generative LLM/VLM -- same category as the OCR
engines themselves, per project guideline.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from ocr_baseline.data import normalize_text
from ocr_baseline.evaluate import Prediction, score_predictions
from ocr_baseline.fields import extract_date
from run_baseline import log_experiment

MODEL_NAME = "iiiorg/piiranha-v1-detect-personal-information"

_NAME_LABELS = {"GIVENNAME", "SURNAME"}
_ADDRESS_LABELS = {"STREET", "BUILDINGNUM", "CITY", "ZIPCODE"}


def _load_pipeline():
    from transformers import AutoModelForTokenClassification, AutoTokenizer, pipeline

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForTokenClassification.from_pretrained(MODEL_NAME)
    return pipeline("token-classification", model=model, tokenizer=tokenizer, aggregation_strategy="simple")


def extract_fields_piiranha(text: str, ner) -> dict[str, str]:
    if not text.strip():
        return {"name": "", "birth_date": "", "address": ""}

    # The model was trained on natural mixed-case text; our OCR output is
    # ALL CAPS, which is badly out-of-distribution for it (confirmed: 0
    # entities found on raw caps text vs 99%+ confidence after title-casing
    # the exact same content). mDeBERTa context is 256 tokens -- truncate
    # rather than crash on longer text.
    entities = ner(text.title()[:1000])

    name_parts, address_parts, date_text = [], [], ""
    for ent in entities:
        # aggregation_strategy="simple" already strips the B-/I- prefix.
        label = ent["entity_group"]
        word = ent["word"].strip()
        if not word:
            continue
        if label in _NAME_LABELS:
            name_parts.append(word)
        elif label in _ADDRESS_LABELS:
            address_parts.append(word)
        elif label == "DATEOFBIRTH" and not date_text:
            date_text = word

    # Birth date is a narrow numeric pattern (esp. MyKad's embedded
    # YYMMDD-PB-#### id) that our own regex already handles near-optimally
    # (see fields.extract_date) -- fall back to it when the generic NER
    # doesn't surface a DATEOFBIRTH entity, rather than leaving it empty.
    birth_date = extract_date(date_text) or extract_date(text)

    return {
        "name": normalize_text(" ".join(name_parts)),
        "birth_date": birth_date,
        "address": normalize_text(", ".join(address_parts)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", default="paddleocr", choices=["tesseract", "easyocr", "paddleocr"])
    parser.add_argument("--split", required=True, choices=["train", "val", "test"])
    parser.add_argument("--reports-dir", default="reports")
    parser.add_argument("--method", default="piiranha-ner")
    args = parser.parse_args()

    reports_dir = Path(args.reports_dir)
    report_path = reports_dir / f"{args.engine}_{args.split}.json"
    data = json.loads(report_path.read_text(encoding="utf-8"))

    ner = _load_pipeline()
    predictions = []
    for i, record in enumerate(data["predictions"], start=1):
        text = str(record["raw"].get("full", ""))
        predicted = extract_fields_piiranha(text, ner)
        predictions.append(
            Prediction(filename=record["filename"], expected=record["expected"], predicted=predicted, raw={})
        )
        if i == 1 or i % 25 == 0 or i == len(data["predictions"]):
            print(f"processed {i}/{len(data['predictions'])}", flush=True)

    metrics = score_predictions(predictions)
    log_experiment(reports_dir, args.engine, args.split, args.method, metrics)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
