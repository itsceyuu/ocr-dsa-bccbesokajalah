"""Structure-aware field extraction for Malaysian identity cards.

This module consumes EasyOCR-style detections instead of treating the OCR
transcript as an undifferentiated string. It is intentionally conservative:
the existing fixed-crop extractor remains a fallback when OCR boxes are
missing or the card layout is too noisy.
"""

from __future__ import annotations

from dataclasses import dataclass
import re

from .data import normalize_text
from .engines import OCRBlock
from .mykad import OCR_DIGIT_MAP, parse_mykad


STOP_LINES = {
    "MALAYSIA",
    "MYKAD",
    "MYKID",
    "IDENTITY",
    "WARGANEGARA",
    "WARGANEOARA",
    "ISLAM",
    "LELAKI",
    "PEREMPUAN",
}

NAME_ANCHORS = {"NAMA", "NAME"}
ADDRESS_HINTS = {
    "ALAMAT",
    "ADDRESS",
    "JALAN",
    "JLN",
    "LORONG",
    "TAMAN",
    "KAMPUNG",
    "KG",
    "NO",
    "PT",
    "PERSIARAN",
    "PANGSAPURI",
    "BLOK",
    "TINGKAT",
    "UNIT",
    "LOT",
    "BANDAR",
    "JAYA",
    "FELDA",
    "PPR",
}


@dataclass(frozen=True)
class TextLine:
    text: str
    blocks: tuple[OCRBlock, ...]
    left: float
    top: float
    right: float
    bottom: float
    confidence: float

    def to_dict(self, index: int) -> dict[str, object]:
        return {
            "index": index,
            "text": self.text,
            "bbox": [self.left, self.top, self.right, self.bottom],
            "confidence": self.confidence,
        }


def _block_bbox(block: OCRBlock) -> tuple[float, float, float, float]:
    xs = [point[0] for point in block.bbox]
    ys = [point[1] for point in block.bbox]
    return min(xs), min(ys), max(xs), max(ys)


def _make_line(blocks: list[OCRBlock]) -> TextLine:
    blocks = sorted(blocks, key=lambda block: _block_bbox(block)[0])
    boxes = [_block_bbox(block) for block in blocks]
    left = min(box[0] for box in boxes)
    top = min(box[1] for box in boxes)
    right = max(box[2] for box in boxes)
    bottom = max(box[3] for box in boxes)
    return TextLine(
        text=" ".join(block.text for block in blocks),
        blocks=tuple(blocks),
        left=left,
        top=top,
        right=right,
        bottom=bottom,
        confidence=sum(block.confidence for block in blocks) / len(blocks),
    )


def group_into_lines(blocks: list[OCRBlock]) -> list[TextLine]:
    """Group nearby OCR boxes into reading-order text lines."""

    groups: list[list[OCRBlock]] = []
    for block in sorted(blocks, key=lambda item: (_block_bbox(item)[1], _block_bbox(item)[0])):
        _, top, _, bottom = _block_bbox(block)
        center = (top + bottom) / 2
        height = max(bottom - top, 1.0)
        best_group: list[OCRBlock] | None = None
        best_distance = float("inf")
        for group in groups:
            group_box = _make_line(group)
            group_center = (group_box.top + group_box.bottom) / 2
            group_height = max(group_box.bottom - group_box.top, 1.0)
            threshold = 0.55 * max(height, group_height)
            if abs(center - group_center) <= threshold and abs(center - group_center) < best_distance:
                best_group = group
                best_distance = abs(center - group_center)
        if best_group is None:
            groups.append([block])
        else:
            best_group.append(block)

    lines = [_make_line(group) for group in groups]
    return sorted(lines, key=lambda line: (line.top, line.left))


def _postcode(text: str) -> str:
    for token in re.findall(r"[A-Z0-9]+", text.upper()):
        digits = "".join(
            OCR_DIGIT_MAP.get(char, char)
            for char in token
            if char.isdigit() or char in OCR_DIGIT_MAP
        )
        if len(digits) == 5 and digits.isdigit():
            return digits
    return ""


def _is_stop(text: str) -> bool:
    return normalize_text(text) in STOP_LINES


def _before_stop(text: str) -> str:
    """Remove card metadata that EasyOCR sometimes appends to an address line."""

    tokens = normalize_text(text).split()
    for index, token in enumerate(tokens):
        if token in STOP_LINES:
            return " ".join(tokens[:index])
    return " ".join(tokens)


def _looks_like_name(text: str) -> bool:
    normalized = normalize_text(text)
    tokens = normalized.split()
    alpha_tokens = [token for token in tokens if re.fullmatch(r"[A-Z]+", token)]
    return bool(tokens) and len(alpha_tokens) >= 2 and len(alpha_tokens) / len(tokens) >= 0.5


def _looks_like_address(text: str) -> bool:
    normalized = normalize_text(text)
    tokens = set(normalized.split())
    return bool(tokens & ADDRESS_HINTS) or bool(_postcode(text))


def _fallback_name(lines: list[str]) -> str:
    id_index = next((index for index, line in enumerate(lines) if parse_mykad(line)), None)
    if id_index is None:
        return ""
    for line in lines[id_index + 1 : id_index + 5]:
        if _is_stop(line) or any(char.isdigit() for char in line):
            continue
        if _looks_like_name(line) and not _looks_like_address(line):
            return normalize_text(line)
    return ""


def _choose_name(lines: list[TextLine], id_index: int | None) -> tuple[str, int | None, dict[str, object]]:
    if id_index is None:
        return "", None, {"reason": "no_id_anchor"}

    name_anchor_index = next(
        (
            index
            for index, line in enumerate(lines)
            if set(normalize_text(line.text).split()) & NAME_ANCHORS
        ),
        None,
    )
    candidates: list[tuple[float, int, TextLine]] = []
    for index in range(id_index + 1, min(id_index + 6, len(lines))):
        line = lines[index]
        if _is_stop(line.text):
            continue
        if any(char.isdigit() for char in line.text) or not _looks_like_name(line.text):
            continue
        distance_score = max(0.0, 1.0 - (index - id_index - 1) / 5.0)
        name_hint = 1.0 if {"BIN", "BINTI"} & set(normalize_text(line.text).split()) else 0.0
        anchor_hint = 1.0 if name_anchor_index is not None and index == name_anchor_index + 1 else 0.0
        score = 0.30 * line.confidence + 0.20 * distance_score + 0.40 * name_hint + 0.10 * anchor_hint
        candidates.append((score, index, line))

    if not candidates:
        return "", None, {"reason": "no_name_candidate"}
    score, index, line = max(candidates, key=lambda item: item[0])
    return normalize_text(line.text), index, {
        "line_index": index,
        "score": score,
        "confidence": line.confidence,
        "text": normalize_text(line.text),
        "name_anchor_index": name_anchor_index,
    }


def _choose_address(
    lines: list[TextLine],
    name_index: int | None,
    id_index: int | None,
) -> tuple[str, dict[str, object]]:
    start = (name_index + 1) if name_index is not None else ((id_index + 1) if id_index is not None else 0)
    scan_end = min(len(lines), start + 12)
    address_lines = lines[start:scan_end]
    postcode_index = next(
        (index for index, line in enumerate(address_lines) if _postcode(line.text)),
        None,
    )

    selected: list[tuple[int, TextLine]] = []
    if postcode_index is not None:
        for index, line in enumerate(address_lines):
            relative_distance = index - postcode_index
            if relative_distance < -4 or relative_distance > 2:
                continue
            if _is_stop(line.text) or parse_mykad(line.text):
                continue
            if (
                any(char.isdigit() for char in line.text)
                and not _postcode(line.text)
                and not _looks_like_address(line.text)
            ):
                continue
            if _looks_like_name(line.text) or _looks_like_address(line.text) or relative_distance >= 0:
                selected.append((index, line))
    else:
        for index, line in enumerate(address_lines):
            if _is_stop(line.text) or parse_mykad(line.text):
                continue
            if _looks_like_address(line.text):
                selected.append((index, line))

    text = normalize_text(" ".join(_before_stop(line.text) for _, line in selected))
    confidence_score = (
        sum(line.confidence for _, line in selected) / len(selected) if selected else 0.0
    )
    address_cue_score = (
        sum(bool(_looks_like_address(line.text)) for _, line in selected) / len(selected)
        if selected
        else 0.0
    )
    diagnostics = {
        "line_indices": [start + index for index, _ in selected],
        "postcode": _postcode(" ".join(line.text for _, line in selected)),
        "score": 0.50 * confidence_score + 0.30 * bool(_postcode(text)) + 0.20 * address_cue_score,
        "text": text,
    }
    return text, diagnostics


def extract_malaysian_fields(
    blocks: list[OCRBlock],
    raw: dict[str, object],
) -> tuple[dict[str, str], dict[str, object]]:
    """Extract Malaysian fields from OCR blocks with a text fallback."""

    lines = group_into_lines(blocks)
    line_texts = [line.text for line in lines]
    id_index: int | None = None
    id_candidate = None
    for index, line in enumerate(lines):
        candidate = parse_mykad(line.text)
        if candidate is not None:
            id_index = index
            id_candidate = candidate
            if candidate.full_number:
                break

    full_text = str(raw.get("full", ""))
    if id_candidate is None:
        id_candidate = parse_mykad(full_text)
    name, name_index, name_diagnostics = _choose_name(lines, id_index)
    if not name:
        name = _fallback_name(line_texts or full_text.splitlines())
    address, address_diagnostics = _choose_address(lines, name_index, id_index)
    if not address:
        address = normalize_text(str(raw.get("address", "")))

    predicted = {
        "name": name,
        "birth_date": id_candidate.birth_date if id_candidate else "",
        "address": address,
    }
    diagnostics = {
        "lines": [line.to_dict(index) for index, line in enumerate(lines)],
        "id_anchor": {
            "line_index": id_index,
            "number": id_candidate.number if id_candidate else "",
            "full_number": bool(id_candidate and id_candidate.full_number),
        },
        "name_candidate": name_diagnostics,
        "address_candidate": address_diagnostics,
    }
    return predicted, diagnostics
