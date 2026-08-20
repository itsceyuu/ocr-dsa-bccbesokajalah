#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ocr_baseline.data import list_unlabeled_images, make_group_split, read_ground_truth


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a deterministic identity-level split of the full dataset")
    parser.add_argument("--ground-truth", default="data/raw/ground_truth.csv")
    parser.add_argument("--image-root", default="data/raw/images")
    parser.add_argument("--output-dir", default="data/splits")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-images-per-identity-in-eval", type=int, default=2)
    parser.add_argument("--min-train-address-identities", type=int, default=15)
    args = parser.parse_args()

    # No subset filtering: the guideline requires using the whole dataset.
    records = read_ground_truth(args.ground_truth)
    splits = make_group_split(
        records, seed=args.seed, max_images_per_identity_in_eval=args.max_images_per_identity_in_eval,
        min_train_address_identities=args.min_train_address_identities,
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for split_name, split_records in splits.items():
        output_path = output_dir / f"{split_name}.csv"
        with output_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["filename", "name", "birth_date", "address"])
            writer.writeheader()
            writer.writerows(record.to_dict() for record in split_records)

    # No ground truth exists for these -- they can't be scored, only used for
    # a final hand-labeled generalization check and qualitative predictions.
    held_out = list_unlabeled_images(args.image_root, records)
    with (output_dir / "held_out.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["filename"])
        writer.writerows([filename] for filename in held_out)

    summary = {
        "seed": args.seed,
        "max_images_per_identity_in_eval": args.max_images_per_identity_in_eval,
        "min_train_address_identities": args.min_train_address_identities,
        "total_labeled": len(records),
        "unique_identities": len({f"{r.name}\x1f{r.birth_date}" for r in records}),
        "held_out_unlabeled": len(held_out),
        "splits": {
            name: {
                "rows": len(items),
                "unique_identities": len({f"{r.name}\x1f{r.birth_date}" for r in items}),
                "max_identity_rows": max(
                    (Counter(f"{r.name}\x1f{r.birth_date}" for r in items).values()), default=0
                ),
                # Sanity check for the address-scarcity bug this flag fixes --
                # should never be 0 for train.
                "address_bearing_rows": sum(1 for r in items if r.address),
            }
            for name, items in splits.items()
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
