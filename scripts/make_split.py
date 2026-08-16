#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ocr_baseline.data import is_malaysian_record, make_group_split, read_ground_truth


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a deterministic identity-level split")
    parser.add_argument("--ground-truth", default="data/raw/ground_truth.csv")
    parser.add_argument("--output-dir", default="data/splits")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--subset", choices=["all", "malaysia"], default="all")
    args = parser.parse_args()

    records = read_ground_truth(args.ground_truth)
    if args.subset == "malaysia":
        records = [record for record in records if is_malaysian_record(record)]
    splits = make_group_split(records, seed=args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for split_name, split_records in splits.items():
        output_path = output_dir / f"{split_name}.csv"
        with output_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["filename", "name", "birth_date", "address"])
            writer.writeheader()
            writer.writerows(record.to_dict() for record in split_records)

    summary = {
        "seed": args.seed,
        "subset": args.subset,
        "total": len(records),
        "unique_identities": len({f"{r.name}\x1f{r.birth_date}" for r in records}),
        "splits": {
            name: {
                "rows": len(items),
                "unique_identities": len({f"{r.name}\x1f{r.birth_date}" for r in items}),
            }
            for name, items in splits.items()
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
