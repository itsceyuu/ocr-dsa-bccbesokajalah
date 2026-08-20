"""Baseline 1: turn OCR-detected lines into a table for a line classifier.

Each OCR detection ("block": text + bbox + confidence) becomes one row.
Features are hand-engineered from its geometry, text shape, and simple
pattern cues -- no LLM/VLM, no AutoML. Labels are derived automatically by
fuzzy-matching each block's text against ground_truth.csv, never by manual
annotation.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

from PIL import Image

from .data import compact_text, normalize_text
from .fields import _ADDRESS_KEYWORDS, _NAME_STOPWORDS, extract_date

FEATURE_NAMES = (
    "y_frac", "x_frac", "height_frac", "width_frac",
    "rank_from_top_frac", "rank_from_bottom_frac",
    "char_count", "token_count", "digit_ratio", "alpha_ratio", "upper_ratio",
    "confidence", "has_date_regex", "has_address_keyword", "has_postcode_pattern",
    "has_header_stopword", "length_rank_on_card", "boilerplate_freq", "name_char_ngram_score",
    "dist_to_face_frac", "date_anchor_rank_dist",
)

_NGRAM_N = 3


def _char_ngrams(text: str, n: int = _NGRAM_N) -> list[str]:
    s = compact_text(text)
    if len(s) < n:
        return [s] if s else []
    return [s[i:i + n] for i in range(len(s) - n + 1)]


def build_char_ngram_model(report_path: str | Path, n: int = _NGRAM_N) -> dict[str, float]:
    """Character n-gram log-likelihood-ratio model: does this text's letter
    pattern look more like a "name" block or a generic "other" block, in ANY
    language? Denser signal than a word-level gazetteer (every character of
    every training block contributes n-grams, vs. 0-or-1 gazetteer hits per
    block) and, unlike a word list, doesn't need to know any language's
    vocabulary -- it learns the character statistics directly from this
    dataset's own weak-labeled blocks (see label_block), so it's as
    multilingual as the training data itself.

    Returns {ngram: log(P(ngram|name) / P(ngram|other))}; unseen n-grams
    score 0 (neutral) at lookup time.
    """

    from collections import Counter

    data = json.loads(Path(report_path).read_text(encoding="utf-8"))
    name_counts: Counter = Counter()
    other_counts: Counter = Counter()
    for prediction in data["predictions"]:
        for block in prediction["raw"].get("blocks") or []:
            label = label_block(block["text"], prediction["expected"])
            bucket = name_counts if label == "name" else (other_counts if label == "other" else None)
            if bucket is not None:
                bucket.update(_char_ngrams(block["text"], n))

    vocab = set(name_counts) | set(other_counts)
    name_total, other_total = sum(name_counts.values()), sum(other_counts.values())
    alpha = 1.0  # Laplace smoothing -- avoids -inf on an n-gram seen in only one class
    return {
        ngram: math.log((name_counts[ngram] + alpha) / (name_total + alpha * len(vocab)))
        - math.log((other_counts[ngram] + alpha) / (other_total + alpha * len(vocab)))
        for ngram in vocab
    }


def name_char_ngram_score(text: str, ngram_model: dict[str, float] | None, n: int = _NGRAM_N) -> float:
    if not ngram_model:
        return 0.0
    grams = _char_ngrams(text, n)
    if not grams:
        return 0.0
    return sum(ngram_model.get(g, 0.0) for g in grams) / len(grams)


def featurize_image_blocks(
    blocks: list[dict], image_size: tuple[int, int], freq_lookup: dict[str, float] | None = None,
    ngram_model: dict[str, float] | None = None, face_center: tuple[float, float] | None = None,
) -> list[dict[str, float]]:
    """One feature row per block. Rank/length features are relative to the
    other blocks on the same card, so this always operates on a whole image
    at once, not a single block in isolation. `freq_lookup` (optional) maps
    compact block text -> fraction of train identities whose card contains
    that exact text anywhere -- a corpus-frequency signal for "this is
    boilerplate/template text", learned from data instead of a per-language
    stopword list (see compute_boilerplate_frequency). `face_center`
    (optional, from preprocess.detect_face_center) is the detected portrait
    photo's center -- distance to it is a layout-agnostic proxy for "personal
    data lives near the photo", true across virtually every ID document
    worldwide, not just one country's template."""

    width, height = image_size
    n = len(blocks)
    if n == 0:
        return []
    freq_lookup = freq_lookup or {}
    diagonal = (width ** 2 + height ** 2) ** 0.5 or 1.0

    order_by_y = sorted(range(n), key=lambda i: min(point[1] for point in blocks[i]["bbox"]))
    rank_from_top = {index: rank for rank, index in enumerate(order_by_y)}

    char_counts = [len(block["text"]) for block in blocks]
    length_order = sorted(range(n), key=lambda i: char_counts[i])
    length_rank = {index: rank / max(n - 1, 1) for rank, index in enumerate(length_order)}

    # Reading-order rank-distance to the nearest date-shaped line -- learned,
    # not a hardcoded rule: extract_date already handles many formats/
    # languages (ISO, MyKad's embedded ID, DD.MM.YYYY, text-month), so this
    # doesn't encode any one country's layout, just "how far (in reading
    # order) is this line from a date," which the model is free to weight
    # however it turns out to matter for a given layout.
    date_ranks = [rank_from_top[i] for i, b in enumerate(blocks) if extract_date(b["text"])]

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
            "boilerplate_freq": freq_lookup.get(compact_text(text), 0.0),
            "name_char_ngram_score": name_char_ngram_score(text, ngram_model),
            "date_anchor_rank_dist": (
                min(abs(rank_from_top[i] - dr) for dr in date_ranks) / max(n - 1, 1) if date_ranks else 1.0
            ),
            "dist_to_face_frac": (
                (((x0 + x1) / 2 - face_center[0]) ** 2 + ((y0 + y1) / 2 - face_center[1]) ** 2) ** 0.5 / diagonal
                if face_center is not None else 1.0  # no face detected -- neutral "far/unknown"
            ),
        })
    return rows


def compute_boilerplate_frequency(report_path: str | Path) -> dict[str, float]:
    """Fraction of distinct *identities* (not images -- a 40-photo retake
    batch of one person shouldn't make their own name look "common") in this
    report whose card has a block matching this exact compact text anywhere.
    High frequency = template/boilerplate ("MALAYSIA", "WARGANEGARA", a
    country name); low frequency = likely personal data (a name, address,
    ID number). Learned straight from the corpus -- no per-language keyword
    list, so it doesn't need to know about any specific layout or language."""

    data = json.loads(Path(report_path).read_text(encoding="utf-8"))
    identity_texts: dict[str, set[str]] = {}
    for prediction in data["predictions"]:
        expected = prediction["expected"]
        identity = f"{expected.get('name', '')}\x1f{expected.get('birth_date', '')}"
        blocks = prediction["raw"].get("blocks") or []
        bucket = identity_texts.setdefault(identity, set())
        bucket.update(compact_text(block["text"]) for block in blocks if block["text"].strip())

    n_identities = max(len(identity_texts), 1)
    counts: dict[str, int] = {}
    for texts in identity_texts.values():
        for text in texts:
            counts[text] = counts.get(text, 0) + 1
    return {text: count / n_identities for text, count in counts.items()}


def _block_center(block: dict) -> tuple[float, float]:
    xs = [p[0] for p in block["bbox"]]
    ys = [p[1] for p in block["bbox"]]
    return (sum(xs) / len(xs), sum(ys) / len(ys))


def _median(values: list[float]) -> float:
    s = sorted(values)
    n = len(s)
    return s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2


def main_cluster_bbox(blocks: list[dict]) -> tuple[float, float, float, float, float, float]:
    """Median center + median absolute deviation of block centers, per axis.
    Robust-statistics outlier detection (works regardless of block count,
    unlike a fixed percentile trim) -- flags a block far from where most of
    this card's own blocks sit, without assuming *where* that is (different
    layouts put content anywhere)."""

    centers = [_block_center(b) for b in blocks]
    mx, my = _median([c[0] for c in centers]), _median([c[1] for c in centers])
    madx = _median([abs(c[0] - mx) for c in centers]) or 1.0
    mady = _median([abs(c[1] - my) for c in centers]) or 1.0
    return (mx, my, madx, mady, 0.0, 0.0)


def in_cluster(block: dict, cluster_stats: tuple[float, float, float, float, float, float], k: float = 4.0) -> bool:
    """Is this block within k robust-deviations of the card's median block
    center on both axes? k=4 is deliberately loose -- only rejects clear
    outliers (e.g. a poster in the background), not genuine spread-out
    document content (name at top, address at bottom is normal)."""

    mx, my, madx, mady, _, _ = cluster_stats
    cx, cy = _block_center(block)
    return abs(cx - mx) <= k * madx and abs(cy - my) <= k * mady


# Suffixes (not prefixes: OCR often glues "ASWARDI"+"BIN" into one token
# with no space) that mark a name block as visibly incomplete -- the join
# below only fires when this matches, so an already-complete name never
# gets a stray next block appended (confirmed necessary: without this gate,
# joining regressed far more already-correct predictions than it fixed).
_NAME_CONTINUATION_SUFFIXES = ("A/L", "A/P", "BINTI", "BINTE", "BINT", "BIN")


def ends_with_name_particle(text: str) -> bool:
    if not text.split():
        return False
    last_token = text.split()[-1].upper()
    return any(last_token.endswith(suffix) for suffix in _NAME_CONTINUATION_SUFFIXES)


def starts_with_name_particle(text: str) -> bool:
    """The surname-continuation half of the same pattern: a block starting
    with "BIN"/"BINTI"/"A/L"/... (e.g. "BIN IDRIS", "BINAZLAN" glued) is a
    name fragment, essentially never a genuine address line -- confirmed
    directly, these leak into address predictions when their own `name`
    score is too low to trip the address-side name-leak guard (short
    fragments score lower than full name lines, same reason cluster_from_seed
    can't rely on score alone -- see find_name_continuation's docstring)."""

    if not text.split():
        return False
    first_token = text.split()[0].upper()
    return any(first_token.startswith(suffix) for suffix in _NAME_CONTINUATION_SUFFIXES)


def find_name_continuation(
    blocks: list[dict], seed_index: int, image_height: float,
    max_gap_frac: float = 0.05, max_tokens: int = 3,
) -> int | None:
    """The single nearest not-yet-included block to the name seed, accepted
    by adjacency + shape (short, like a surname) -- NOT by the classifier's
    own class score for that block. A real surname block often scores lower
    for "name" than an unrelated nearby address line does (confirmed: e.g.
    "AHMAD" scored 0.195 while a stray address line scored 0.664 on the same
    card), so score-based absorption (what cluster_from_seed does for the
    multi-line address case) would be unsafe here. Reading-order adjacency
    is more reliable: in every observed case the true surname was the very
    next block, while unrelated same-scoring lines were several blocks
    further away. Malay names are at most given-name + surname (2 blocks),
    so this only ever looks one step, unlike address's open-ended walk."""

    if not blocks or len(blocks) < 2:
        return None
    y_ranges = [(min(p[1] for p in b["bbox"]), max(p[1] for p in b["bbox"])) for b in blocks]
    max_gap = max_gap_frac * max(image_height, 1.0)
    lo, hi = y_ranges[seed_index]

    best, best_gap = None, max_gap
    for i, (y0, y1) in enumerate(y_ranges):
        if i == seed_index:
            continue
        gap = max(0.0, y0 - hi, lo - y1)
        if gap <= best_gap:
            best, best_gap = i, gap
    if best is None:
        return None
    if len(blocks[best]["text"].split()) > max_tokens:
        return None
    return best


def cluster_from_seed(
    blocks: list[dict],
    scores: list[float],
    seed_index: int,
    image_height: float,
    soft_threshold: float = 0.15,
    max_gap_frac: float = 0.05,
) -> list[int]:
    """Grow outward from the highest-scoring block by actual spatial gap, not
    list-index adjacency -- reading-order sort isn't perfect when blocks
    visually overlap in y (e.g. an unrelated header block sitting between
    two real address lines), so index-adjacency alone stops the walk one
    step too early. Finds a multi-line field (e.g. a 4-line address) by
    mutual proximity, not by assuming which part of the card it lives on."""

    if not blocks:
        return []
    max_gap = max_gap_frac * max(image_height, 1.0)
    y_ranges = [(min(p[1] for p in b["bbox"]), max(p[1] for p in b["bbox"])) for b in blocks]

    included = {seed_index}
    while True:
        lo = min(y_ranges[i][0] for i in included)
        hi = max(y_ranges[i][1] for i in included)
        best, best_gap = None, max_gap
        for i, (y0, y1) in enumerate(y_ranges):
            if i in included or scores[i] < soft_threshold:
                continue
            gap = max(0.0, y0 - hi, lo - y1)
            if gap <= best_gap:
                best, best_gap = i, gap
        if best is None:
            break
        included.add(best)
    return sorted(included)


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


def build_table(
    report_path: str | Path, image_root: str | Path, freq_lookup: dict[str, float] | None = None,
    ngram_model: dict[str, float] | None = None, use_face: bool = False,
) -> tuple[list[dict], list[str], list[dict]]:
    """One row per OCR block across every image in a saved run_baseline.py
    report. Reuses the already-computed OCR output -- no re-running OCR.
    `freq_lookup`/`ngram_model` should come from compute_boilerplate_frequency
    / build_char_ngram_model on the train report -- pass the same ones in for
    both train and val/test tables so the features mean the same thing in
    both (train-only, no leakage). `use_face=True` runs face detection per
    image (slower -- opt-in, not free like the other two precomputed
    lookups)."""

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
            face_center = None
            if use_face:
                from .preprocess import detect_face_center

                face_center = detect_face_center(image)
        rows = featurize_image_blocks(blocks, size, freq_lookup, ngram_model, face_center)
        for block, feat in zip(blocks, rows):
            features.append(feat)
            labels.append(label_block(block["text"], prediction["expected"]))
            meta.append({"filename": prediction["filename"], "text": block["text"]})
    return features, labels, meta
