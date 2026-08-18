"""Turn raw OCR text into the three target fields: name, birth_date, address.

The dataset spans dozens of document types and countries (MyKad, EU driving
licences, passports, national ID specimens, ...) with completely different
layouts, so there is no fixed region to crop -- field selection has to work
off the OCR text/line structure itself. This is a first, deliberately simple
baseline for that; the real modelling work is improving these heuristics
(or replacing them with a trained line classifier) against held-out data.
"""

from __future__ import annotations

import re
from datetime import date

from .data import normalize_text

_MONTHS = {
    "JAN": 1, "FEB": 2, "MAR": 3, "APR": 4, "MAY": 5, "JUN": 6,
    "JUL": 7, "AUG": 8, "SEP": 9, "OCT": 10, "NOV": 11, "DEC": 12,
}

# YYYY-MM-DD / YYYY.MM.DD
_ISO_DATE_RE = re.compile(r"\b(?P<y>\d{4})[.\-/](?P<m>\d{2})[.\-/](?P<d>\d{2})\b")
# DD MON YYYY, e.g. "22 JUN 1969"
_TEXT_MONTH_RE = re.compile(r"\b(?P<d>\d{1,2})[.\- ]+(?P<mon>[A-Z]{3,9})\.?[.\- ]+(?P<y>\d{4})\b")
# DD.MM.YYYY / DD/MM/YYYY / DD-MM-YY etc (most European/MyKad documents)
_NUMERIC_DATE_RE = re.compile(r"\b(?P<a>\d{1,2})[.\-/](?P<b>\d{1,2})[.\-/](?P<c>\d{2,4})\b")

_ADDRESS_KEYWORDS = (
    "JALAN", "KAMPUNG", "TAMAN", "STREET", "STRASSE", "AVENUE", "ROAD", "RUA", "VIA",
)
_NAME_STOPWORDS = {
    "MALAYSIA", "MYKAD", "MYKID", "IDENTITY", "WARGANEGARA", "ISLAM", "LELAKI",
    "PEREMPUAN", "PASSPORT", "REPUBLIC", "LICENCE", "LICENSE", "SPECIMEN",
}


def _valid_iso_date(year: int, month: int, day: int) -> str:
    try:
        return date(year, month, day).isoformat()
    except ValueError:
        return ""


def extract_date(text: str) -> str:
    """Find the first plausible calendar date in OCR text, many formats."""

    text = text.upper()

    match = _ISO_DATE_RE.search(text)
    if match:
        result = _valid_iso_date(int(match["y"]), int(match["m"]), int(match["d"]))
        if result:
            return result

    match = _TEXT_MONTH_RE.search(text)
    if match and match["mon"][:3] in _MONTHS:
        result = _valid_iso_date(int(match["y"]), _MONTHS[match["mon"][:3]], int(match["d"]))
        if result:
            return result

    for match in _NUMERIC_DATE_RE.finditer(text):
        a, b, c = int(match["a"]), int(match["b"]), int(match["c"])
        year = c if c > 31 else (2000 + c if c < 50 else 1900 + c)
        # Most represented documents write DD.MM.YYYY; fall back to
        # MM.DD.YYYY only if the day-first reading is impossible.
        result = _valid_iso_date(year, b, a) or _valid_iso_date(year, a, b)
        if result:
            return result
    return ""


def extract_name(text: str) -> str:
    """ponytail: naive heuristic (longest all-letters line), ceiling is low on
    numbered-field documents (e.g. EU licences: "1. Mustermann"). Upgrade by
    detecting field labels or training a line classifier once there's a
    labeled sample of layouts to learn from."""

    best = ""
    for raw_line in text.splitlines():
        line = normalize_text(raw_line)
        if not line or line in _NAME_STOPWORDS or any(char.isdigit() for char in line):
            continue
        if len(line) > len(best):
            best = line
    return best


def extract_address(text: str) -> str:
    """ponytail: keyword + digit-density heuristic, only fires when the line
    looks like a street/postcode line. Most documents in this dataset have no
    address field at all, so returning "" is the correct answer far more
    often than any guess."""

    lines = [normalize_text(raw_line) for raw_line in text.splitlines()]
    candidates = [
        line
        for line in lines
        if line and (any(keyword in line for keyword in _ADDRESS_KEYWORDS) or re.search(r"\b\d{4,5}\b", line))
    ]
    return max(candidates, key=len, default="")


def parse_fields(text: str) -> dict[str, str]:
    return {
        "name": extract_name(text),
        "birth_date": extract_date(text),
        "address": extract_address(text),
    }
