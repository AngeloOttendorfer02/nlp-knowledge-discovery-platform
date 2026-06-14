import json
from pathlib import Path

import pandas as pd

from src.experiments.run_bm25_baseline import (
    load_relevance_queries,
    run_bm25_baseline,
)


def test_load_relevance_queries_requires_relevant_documents(tmp_path):
    path = tmp_path / "queries.json"
    path.write_text(
        json.dumps(
            [
                {
                    "query_id": "q1",
                    "query": "graph retrieval",
                    "relevant_doc_ids": [],
                }
            ]
        ),
        encoding="utf-8",
    )

    try:
        load_relevance_queries(path)
    except ValueError as exc:
        assert "at least one relevant document" in str(exc)
    else:
        raise AssertionError("Expected ValueError for empty relevant_doc_ids")


def test_bm25_baseline_experiment_writes_results_and_metrics(tmp_path):
    documents_path = tmp_path / "processed_documents.csv"
    queries_path = tmp_path / "queries.json"
    results_path = tmp_path / "bm25_results.csv"
    metrics_path = tmp_path / "bm25_metrics.csv"

    pd.DataFrame(
        [
            {
                "doc_id": "1",
                "text": "Graph neural networks for scientific document retrieval.",
            },
            {
                "doc_id": "2",
                "text": "Transformer models for named entity recognition and relation extraction.",
            },
        ]
    ).to_csv(documents_path, index=False)

    queries_path.write_text(
        json.dumps(
            [
                {
                    "query_id": "q1",
                    "query": "graph neural networks document retrieval",
                    "relevant_doc_ids": ["1"],
                },
                {
                    "query_id": "q2",
                    "query": "transformer entity recognition",
                    "relevant_doc_ids": ["2"],
                },
            ]
        ),
        encoding="utf-8",
    )

    metrics = run_bm25_baseline(
        documents_path=documents_path,
        queries_path=queries_path,
        results_path=results_path,
        metrics_path=metrics_path,
        top_k=2,
    )

    assert results_path.exists()
    assert metrics_path.exists()

    results_df = pd.read_csv(results_path)
    metrics_df = pd.read_csv(metrics_path)

    assert {
        "query_id",
        "query",
        "rank",
        "doc_id",
        "score",
        "is_relevant",
        "text",
    }.issubset(results_df.columns)

    assert {"method", "precision@2", "recall@2", "mrr"}.issubset(metrics_df.columns)

    assert metrics["recall@2"] == 1.0
    assert metrics_df.iloc[0]["method"] == "BM25"