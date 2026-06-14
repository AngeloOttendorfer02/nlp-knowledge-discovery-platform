import json
from pathlib import Path

from src.preprocessing.document_loader import load_arxiv_jsonl


def write_jsonl(path: Path, records):
    with path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record) + "\n")


def test_load_arxiv_jsonl_filters_by_category(tmp_path):
    path = tmp_path / "sample.jsonl"

    write_jsonl(
        path,
        [
            {
                "id": "001",
                "title": "AI Paper",
                "abstract": "Artificial intelligence paper.",
                "authors": "Alice Smith",
                "categories": "cs.AI cs.CL",
                "update_date": "2026-01-01",
            },
            {
                "id": "002",
                "title": "Math Paper",
                "abstract": "Mathematics paper.",
                "authors": "Bob Miller",
                "categories": "math.CO",
                "update_date": "2026-01-02",
            },
        ],
    )

    df = load_arxiv_jsonl(
        str(path),
        categories=["cs.AI"],
    )

    assert len(df) == 1
    assert df.iloc[0]["doc_id"] == "001"
    assert df.iloc[0]["categories"] == ["cs.AI", "cs.CL"]


def test_load_arxiv_jsonl_respects_sample_size(tmp_path):
    path = tmp_path / "sample.jsonl"

    write_jsonl(
        path,
        [
            {
                "id": "001",
                "title": "Paper 1",
                "abstract": "First paper.",
                "authors": "Alice",
                "categories": "cs.AI",
                "update_date": "2026-01-01",
            },
            {
                "id": "002",
                "title": "Paper 2",
                "abstract": "Second paper.",
                "authors": "Bob",
                "categories": "cs.AI",
                "update_date": "2026-01-02",
            },
        ],
    )

    df = load_arxiv_jsonl(
        str(path),
        sample_size=1,
        categories=["cs.AI"],
    )

    assert len(df) == 1
    assert df.iloc[0]["doc_id"] == "001"


def test_load_arxiv_jsonl_has_standard_columns(tmp_path):
    path = tmp_path / "sample.jsonl"

    write_jsonl(
        path,
        [
            {
                "id": "001",
                "title": "Paper",
                "abstract": "Abstract.",
                "authors": "Alice",
                "categories": "cs.CL",
                "update_date": "2026-01-01",
            },
        ],
    )

    df = load_arxiv_jsonl(str(path))

    assert list(df.columns) == [
        "doc_id",
        "title",
        "abstract",
        "authors",
        "categories",
        "date",
    ]