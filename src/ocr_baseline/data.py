from __future__ import annotations

import csv
import random
import re
import unicodedata
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


FIELDS = ("name", "birth_date", "address")
SPLITS = ("train", "val", "test")


@dataclass(frozen=True)
class Record:
    filename: str
    name: str
    birth_date: str
    address: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


def is_malaysian_record(record: Record) -> bool:
    """Select the Malaysian subset in this export.

    The supplied annotations contain addresses for all 112 Malaysian
    MyKad/MyKid rows (image_001 through image_112) and blank addresses for the
    international specimen-card rows. This is an annotation-level selector,
    not a visual guess from the image.
    """

    return bool(record.address)


def _parse_csv_row(row: list[str]) -> list[str]:
    """Handle both normal rows and the first 131 rows in the supplied CSV.

    The first portion of the provided export is wrapped in one extra pair of
    quotes, so ``csv.reader`` returns one cell for those rows.  Parsing that
    cell again recovers the same four columns as the rest of the file.
    """

    if len(row) == 1 and "," in row[0]:
        row = next(csv.reader([row[0]]))
    return row


def read_ground_truth(path: str | Path) -> list[Record]:
    path = Path(path)
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        expected = ["filename", "name", "birth_date", "address"]
        if header[:4] != expected:
            raise ValueError(f"Unexpected ground-truth header: {header}")

        records: list[Record] = []
        for line_number, raw_row in enumerate(reader, start=2):
            row = _parse_csv_row(raw_row)
            if len(row) < 4:
                raise ValueError(f"Malformed row {line_number}: expected 4 columns, got {row}")
            records.append(Record(*(cell.strip() for cell in row[:4])))
    return records


def identity_key(record: Record) -> str:
    """Use the stable label pair to prevent the same person crossing splits."""

    return f"{record.name}\x1f{record.birth_date}"


def make_group_split(
    records: Iterable[Record],
    seed: int = 42,
    ratios: tuple[float, float, float] = (0.8, 0.1, 0.1),
) -> dict[str, list[Record]]:
    """Make a deterministic approximate 80/10/10 split at identity level."""

    records = list(records)
    if not records:
        raise ValueError("Cannot split an empty record set")
    if abs(sum(ratios) - 1.0) > 1e-8 or any(r <= 0 for r in ratios):
        raise ValueError(f"Ratios must be positive and sum to 1: {ratios}")

    groups: dict[str, list[Record]] = defaultdict(list)
    for record in records:
        groups[identity_key(record)].append(record)

    rng = random.Random(seed)
    group_items = list(groups.items())
    rng.shuffle(group_items)
    group_items.sort(key=lambda item: len(item[1]), reverse=True)

    split_names = list(SPLITS)
    target_sizes = [len(records) * ratio for ratio in ratios]
    result: dict[str, list[Record]] = {name: [] for name in split_names}
    for key, group in group_items:
        del key  # The key is only needed for grouping; records carry the labels.
        split_index = min(
            range(len(split_names)),
            key=lambda index: len(result[split_names[index]]) - target_sizes[index],
        )
        result[split_names[split_index]].extend(group)

    # Stable order makes generated manifests easy to diff and audit.
    for split in split_names:
        result[split].sort(key=lambda record: record.filename)
    return result


def normalize_text(value: str) -> str:
    value = unicodedata.normalize("NFKC", value).upper()
    value = value.replace("\n", " ").replace("\r", " ")
    value = re.sub(r"[^\w\s]", " ", value, flags=re.UNICODE)
    return re.sub(r"\s+", " ", value).strip()


def compact_text(value: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", normalize_text(value))


def levenshtein(left: str, right: str) -> int:
    if len(left) < len(right):
        left, right = right, left
    previous = list(range(len(right) + 1))
    for i, left_char in enumerate(left, 1):
        current = [i]
        for j, right_char in enumerate(right, 1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[j] + 1,
                    previous[j - 1] + (left_char != right_char),
                )
            )
        previous = current
    return previous[-1]


def normalized_equal(expected: str, predicted: str) -> bool:
    return normalize_text(expected) == normalize_text(predicted)
