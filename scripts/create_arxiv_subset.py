"""
Create a scalable subset from the arXiv metadata JSONL file.

Example:
    python scripts/create_arxiv_subset.py \
        --input data/raw/arxiv-metadata-oai-snapshot.json \
        --output data/raw/arxiv_cs_subset.csv \
        --max-papers 10000 \
        --categories cs.AI cs.LG cs.CL cs.IR \
        --seed 42
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
from typing import Iterable, List


DEFAULT_CATEGORIES = ["cs.AI", "cs.LG", "cs.CL", "cs.IR"]
FIELDS = ["id", "title", "abstract", "authors", "categories", "update_date"]


def split_categories(value: str) -> List[str]:
    """Split arXiv category strings such as 'cs.CL cs.AI'."""
    if not value:
        return []
    return [category.strip() for category in value.split() if category.strip()]


def matches_categories(
    paper_categories: Iterable[str],
    wanted_categories: set[str],
) -> bool:
    """Return True if a paper has at least one wanted category."""
    return bool(set(paper_categories).intersection(wanted_categories))


def clean_text_field(value) -> str:
    """Normalize multiline metadata fields for CSV/JSONL output."""
    if value is None:
        return ""
    return str(value).replace("\n", " ").strip()


def normalize_record(record: dict) -> dict:
    """Keep only the fields required by the project."""
    categories = split_categories(record.get("categories", ""))

    return {
        "id": clean_text_field(record.get("id", "")),
        "title": clean_text_field(record.get("title", "")),
        "abstract": clean_text_field(record.get("abstract", "")),
        "authors": clean_text_field(record.get("authors", "")),
        "categories": " ".join(categories),
        "update_date": clean_text_field(record.get("update_date", "")),
    }


def write_csv(rows: List[dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def write_jsonl(rows: List[dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def create_arxiv_subset(
    input_path: Path,
    output_path: Path,
    max_papers: int,
    categories: List[str],
    seed: int = 42,
) -> None:
    """
    Stream the arXiv JSONL file and save a filtered subset.

    The input file is processed line by line, so the full dataset is never
    loaded into memory.
    """
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    if max_papers <= 0:
        raise ValueError("max_papers must be a positive integer")

    random.seed(seed)

    wanted_categories = set(categories)
    rows: List[dict] = []

    total_seen = 0
    total_matched = 0

    with input_path.open("r", encoding="utf-8") as file:
        for line in file:
            total_seen += 1
            line = line.strip()

            if not line:
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue

            paper_categories = split_categories(record.get("categories", ""))

            if not matches_categories(paper_categories, wanted_categories):
                continue

            rows.append(normalize_record(record))
            total_matched += 1

            if total_matched >= max_papers:
                break

    suffix = output_path.suffix.lower()

    if suffix == ".csv":
        write_csv(rows, output_path)
    elif suffix in {".jsonl", ".json"}:
        write_jsonl(rows, output_path)
    else:
        raise ValueError("Output file must end with .csv, .json, or .jsonl")

    print("arXiv subset created successfully.")
    print(f"Input file: {input_path}")
    print(f"Output file: {output_path}")
    print(f"Total records scanned: {total_seen}")
    print(f"Total matching papers saved: {len(rows)}")
    print(f"Categories: {', '.join(categories)}")
    print(f"Seed: {seed}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a filtered arXiv subset from a large JSONL file."
    )

    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help="Path to the full arXiv metadata JSON/JSONL file.",
    )

    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Output file path ending in .csv, .json, or .jsonl.",
    )

    parser.add_argument(
        "--max-papers",
        type=int,
        default=10000,
        help="Maximum number of matching papers to save.",
    )

    parser.add_argument(
        "--categories",
        nargs="+",
        default=DEFAULT_CATEGORIES,
        help="arXiv categories to keep.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed argument for reproducible experiment configuration.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    create_arxiv_subset(
        input_path=args.input,
        output_path=args.output,
        max_papers=args.max_papers,
        categories=args.categories,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()