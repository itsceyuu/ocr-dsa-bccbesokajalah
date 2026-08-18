"""Baseline 1: turn OCR-detected lines into a table for a line classifier.

Each OCR detection ("block": text + bbox + confidence) becomes one row.
Features are hand-engineered from its geometry, text shape, and simple
pattern cues -- no LLM/VLM, no AutoML. Labels are derived automatically by
fuzzy-matching each block's text against ground_truth.csv, never by manual
annotation.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from PIL import Image

from .data import normalize_text
from .fields import _ADDRESS_KEYWORDS, _NAME_STOPWORDS, extract_date

FEATURE_NAMES = (
    "y_frac", "x_frac", "height_frac", "width_frac",
    "rank_from_top_frac", "rank_from_bottom_frac",
    "char_count", "token_count", "digit_ratio", "alpha_ratio", "upper_ratio",
    "confidence", "has_date_regex", "has_address_keyword", "has_postcode_pattern",
    "has_header_stopword", "length_rank_on_card",
)


def featurize_image_blocks(blocks: list[dict], image_size: tuple[int, int]) -> list[dict[str, float]]:
    """One feature row per block. Rank/length features are relative to the
    other blocks on the same card, so this always operates on a whole image
    at once, not a single block in isolation."""

    width, height = image_size
    n = len(blocks)
    if n == 0:
        return []

    order_by_y = sorted(range(n), key=lambda i: min(point[1] for point in blocks[i]["bbox"]))
    rank_from_top = {index: rank for rank, index in enumerate(order_by_y)}

    char_counts = [len(block["text"]) for block in blocks]
    length_order = sorted(range(n), key=lambda i: char_counts[i])
    length_rank = {index: rank / max(n - 1, 1) for rank, index in enumerate(length_order)}

    rows = []
    for i, block in enumerate(blocks):
        text = block["text"]
        xs = [point[0] for point in block["bbox"]]
        ys = [point[1] for point in block["bbox"]]
        x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
        letters = sum(char.isalpha() for char in text)
        digits = sum(char.isdigit() for char in text)
        upper = sum(char.isupper() for char in text if char.isalpha())
        length = max(len(text), 1)
        rows.append({
            "y_frac": y0 / max(height, 1),
            "x_frac": x0 / max(width, 1),
            "height_frac": (y1 - y0) / max(height, 1),
            "width_frac": (x1 - x0) / max(width, 1),
            "rank_from_top_frac": rank_from_top[i] / max(n - 1, 1),
            "rank_from_bottom_frac": 1 - rank_from_top[i] / max(n - 1, 1),
            "char_count": float(len(text)),
            "token_count": float(len(text.split())),
            "digit_ratio": digits / length,
            "alpha_ratio": letters / length,
            "upper_ratio": upper / max(letters, 1),
            "confidence": float(block.get("confidence", 0.0)),
            "has_date_regex": float(bool(extract_date(text))),
            "has_address_keyword": float(any(k in text.upper() for k in _ADDRESS_KEYWORDS)),
            "has_postcode_pattern": float(bool(re.search(r"\b\d{4,5}\b", text))),
            "has_header_stopword": float(normalize_text(text) in _NAME_STOPWORDS),
            "length_rank_on_card": length_rank[i],
        })
    return rows


def _tokens(text: str) -> set[str]:
    return set(normalize_text(text).split())


def label_block(text: str, expected: dict[str, str], min_overlap: float = 0.6) -> str:
    """Auto-label one block from ground truth -- no human looked at this.

    birth_date: the block's own regex-extracted date must equal the ISO
    ground-truth date. name/address: containment from the block's side (so
    one block that's only part of a multi-line address can still be
    labeled), scored as the fraction of the block's own tokens found in the
    expected string.
    """

    if not text.strip():
        return "other"

    candidate_date = extract_date(text)
    if candidate_date and candidate_date == expected.get("birth_date", ""):
        return "birth_date"

    block_tokens = _tokens(text)
    # A single generic token (e.g. "FREDEZ" matching one word of "FREDEZ
    # FIDAL") trivially hits 100% containment and floods the label with
    # boilerplate-shaped false positives -- require at least two tokens so
    # only a real name/address fragment, not a lone word, gets labeled.
    if len(block_tokens) < 2:
        return "other"

    best_label, best_score = "other", 0.0
    for field in ("name", "address"):
        expected_tokens = _tokens(expected.get(field, ""))
        if not expected_tokens:
            continue
        overlap = len(block_tokens & expected_tokens) / len(block_tokens)
        if overlap >= min_overlap and overlap > best_score:
            best_label, best_score = field, overlap
    return best_label


def build_table(report_path: str | Path, image_root: str | Path) -> tuple[list[dict], list[str], list[dict]]:
    """One row per OCR block across every image in a saved run_baseline.py
    report. Reuses the already-computed OCR output -- no re-running OCR."""

    data = json.loads(Path(report_path).read_text(encoding="utf-8"))
    image_root = Path(image_root)
    features: list[dict] = []
    labels: list[str] = []
    meta: list[dict] = []
    for prediction in data["predictions"]:
        blocks = prediction["raw"].get("blocks") or []
        if not blocks:
            continue
        with Image.open(image_root / prediction["filename"]) as image:
            size = image.size
        rows = featurize_image_blocks(blocks, size)
        for block, feat in zip(blocks, rows):
            features.append(feat)
            labels.append(label_block(block["text"], prediction["expected"]))
            meta.append({"filename": prediction["filename"], "text": block["text"]})
    return features, labels, meta
