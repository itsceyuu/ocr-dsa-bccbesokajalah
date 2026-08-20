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


def _parse_csv_row(row: list[str]) -> list[str]:
    """Handle both normal rows and the first 131 rows in the supplied CSV.

    The first portion of the provided export is wrapped in one extra pair of
    quotes, so ``csv.reader`` returns one cell for those rows. Parsing that
    cell again recovers the same four columns as the rest of the file.
    """

    if len(row) == 1 and "," in row[0]:
        row = next(csv.reader([row[0]]))
    return row


def read_ground_truth(path: str | Path) -> list[Record]:
    """Read every row of the dataset's ground truth. No filtering, no subset."""

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
    max_images_per_identity_in_eval: int = 2,
    min_train_address_identities: int = 15,
) -> dict[str, list[Record]]:
    """Make a deterministic approximate 80/10/10 split at identity level.

    Some identities in this dataset are the same physical card retaken 20-40
    times under different lighting/angle (see the row counts in the ground
    truth). Grouping by identity already keeps those retakes out of more than
    one split, but without a cap a single retaken identity can still dominate
    val/test by row count. ``max_images_per_identity_in_eval`` keeps any
    identity bigger than that out of val/test entirely (it lands in train,
    where the extra retakes are useful as lighting/angle variety); only
    small, close-to-single-shot identities are eligible for val/test.

    Only a minority of identities (the large international specimen/passport
    batches) are big enough to fill train's row-count target on their own --
    if they do, every small identity (which is where almost all non-empty
    `address` values live, one MyKad per person) gets pushed to eval by the
    row-count greedy rule below, leaving train with zero address training
    signal purely by chance. `min_train_address_identities` reserves that
    many address-bearing identities for train up front so this can't happen.
    """

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

    address_bearing_keys = [key for key, group in group_items if any(r.address for r in group)]
    reserved_for_train = set(address_bearing_keys[:min_train_address_identities])

    group_items.sort(key=lambda item: len(item[1]), reverse=True)

    split_names = list(SPLITS)
    eval_splits = {"val", "test"}
    target_sizes = [len(records) * ratio for ratio in ratios]
    result: dict[str, list[Record]] = {name: [] for name in split_names}
    for key, group in group_items:
        if key in reserved_for_train:
            result["train"].extend(group)
            continue
        oversized = len(group) > max_images_per_identity_in_eval
        eligible = [
            index
            for index, name in enumerate(split_names)
            if not (oversized and name in eval_splits)
        ]
        split_index = min(
            eligible,
            key=lambda index: len(result[split_names[index]]) - target_sizes[index],
        )
        result[split_names[split_index]].extend(group)

    # Stable order makes generated manifests easy to diff and audit.
    for split in split_names:
        result[split].sort(key=lambda record: record.filename)
    return result


def list_unlabeled_images(image_root: str | Path, records: Iterable[Record]) -> list[str]:
    """Filenames present under ``image_root`` with no ground-truth row.

    This is the held-out set: no labels exist, so it can never be part of
    train/val/test scoring. It's meant for a final, small hand-labeled
    generalization check plus qualitative demo predictions.
    """

    labeled = {record.filename for record in records}
    all_images = {path.name for path in Path(image_root).iterdir() if path.is_file()}
    return sorted(all_images - labeled)


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
