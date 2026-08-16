"""OCR-tolerant parsing for Malaysian MyKad/MyKid numbers.

The Malaysian NRIC format is twelve digits displayed as ``YYMMDD-PB-####``.
OCR often confuses a small set of glyphs (for example ``O`` and ``0``), so
correction is applied only inside a candidate that already matches the MyKad
shape. This avoids globally replacing letters in names and addresses.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from datetime import date


# These substitutions are deliberately conservative and are only used inside
# the candidate regex below, never on the entire OCR transcript.
OCR_DIGIT_MAP = {
    "O": "0",
    "Q": "0",
    "D": "0",
    "I": "1",
    "L": "1",
    "T": "1",
    "Z": "2",
    "S": "5",
    "G": "6",
    "B": "8",
}

OCR_DIGIT = r"[0-9OQDILTZSGB]"
OCR_SEPARATOR = r"[\s._:/\\|,–—-]*"


def _digit_block(length: int) -> str:
    return OCR_DIGIT + "".join(f"{OCR_SEPARATOR}{OCR_DIGIT}" for _ in range(length - 1))


# Supports both the printed 6-2-4 form and OCR output such as 79074 9-14-5657.
MYKAD_RE = re.compile(
    rf"(?<![A-Z0-9])"
    rf"(?P<birth>{_digit_block(6)}){OCR_SEPARATOR}"
    rf"(?P<place>{_digit_block(2)}){OCR_SEPARATOR}"
    rf"(?P<serial>{_digit_block(4)})"
    rf"(?![A-Z0-9])",
    re.IGNORECASE,
)

# Fallback for a crop that contains only the first six digits.
MYKAD_DATE_RE = re.compile(
    rf"(?<![A-Z0-9])(?P<birth>{_digit_block(6)})(?![A-Z0-9])",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class MyKadCandidate:
    raw: str
    number: str
    birth_date: str
    place_code: str
    serial: str
    score: int
    start: int
    end: int
    full_number: bool


def _normalize_digits(value: str) -> str:
    value = unicodedata.normalize("NFKC", value).upper()
    return "".join(OCR_DIGIT_MAP.get(char, char) for char in value if char.isdigit() or char in OCR_DIGIT_MAP)


def _parse_date(value: str) -> str:
    if len(value) != 6 or not value.isdigit():
        return ""
    yy, month, day = int(value[:2]), int(value[2:4]), int(value[4:6])
    # Malaysian NRIC numbers do not encode the century. Use the current-year
    # pivot so current and recent birth years are interpreted naturally.
    year = 2000 + yy if yy <= date.today().year % 100 else 1900 + yy
    try:
        return date(year, month, day).isoformat()
    except ValueError:
        return ""


def parse_mykad_candidates(text: str) -> list[MyKadCandidate]:
    """Return valid MyKad candidates ordered from strongest to weakest."""

    text = unicodedata.normalize("NFKC", text).upper()
    candidates: list[MyKadCandidate] = []

    for match in MYKAD_RE.finditer(text):
        birth = _normalize_digits(match.group("birth"))
        place = _normalize_digits(match.group("place"))
        serial = _normalize_digits(match.group("serial"))
        birth_date = _parse_date(birth)
        if not birth_date or len(place) != 2 or len(serial) != 4:
            continue
        # A full 12-digit candidate outranks a date-only fallback. A two-digit
        # place code in 01-16 receives a small bonus for the common state codes;
        # extended JPN place codes remain accepted.
        score = 100 + (2 if 1 <= int(place) <= 16 else 0)
        candidates.append(
            MyKadCandidate(
                raw=match.group(0),
                number=birth + place + serial,
                birth_date=birth_date,
                place_code=place,
                serial=serial,
                score=score,
                start=match.start(),
                end=match.end(),
                full_number=True,
            )
        )

    for match in MYKAD_DATE_RE.finditer(text):
        birth = _normalize_digits(match.group("birth"))
        birth_date = _parse_date(birth)
        if not birth_date:
            continue
        candidates.append(
            MyKadCandidate(
                raw=match.group(0),
                number=birth,
                birth_date=birth_date,
                place_code="",
                serial="",
                score=50,
                start=match.start(),
                end=match.end(),
                full_number=False,
            )
        )

    return sorted(candidates, key=lambda candidate: (-candidate.score, candidate.start))


def parse_mykad(text: str) -> MyKadCandidate | None:
    return next(iter(parse_mykad_candidates(text)), None)


def extract_birth_date(text: str) -> str:
    candidate = parse_mykad(text)
    return candidate.birth_date if candidate else ""

