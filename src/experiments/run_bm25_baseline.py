"""
Run a reproducible BM25 baseline retrieval experiment.

Example:
    python -m src.experiments.run_bm25_baseline
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from src.evaluation.retrieval_metrics import evaluate_retrieval_run
from src.retrieval.bm25_retriever import BM25Retriever


def load_processed_documents(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Processed documents not found: {path}\n"
            "Run first:\n"
            "    python -m src.pipeline.run_pipeline --skip-embeddings"
        )

    df = pd.read_csv(path, dtype={"doc_id": "string"})

    required_columns = {"doc_id", "text"}
    missing = required_columns.difference(df.columns)

    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df["doc_id"] = df["doc_id"].fillna("").astype(str).str.strip()
    df["text"] = df["text"].fillna("").astype(str)

    return df


def load_relevance_queries(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Relevance query file not found: {path}")

    with path.open("r", encoding="utf-8") as file:
        queries = json.load(file)

    if not isinstance(queries, list):
        raise ValueError("Relevance query file must contain a list.")

    for query in queries:
        if not query.get("query_id"):
            raise ValueError("Every query must have a query_id.")
        if not query.get("query"):
            raise ValueError("Every query must have a query text.")
        if not query.get("relevant_doc_ids"):
            raise ValueError("Every query must have at least one relevant document.")

        query["relevant_doc_ids"] = [
            str(doc_id).strip()
            for doc_id in query["relevant_doc_ids"]
        ]

    return queries


def run_bm25_baseline(
    documents_path: Path,
    queries_path: Path,
    results_path: Path,
    metrics_path: Path,
    top_k: int = 5,
) -> Dict[str, float]:
    df = load_processed_documents(documents_path)
    queries = load_relevance_queries(queries_path)

    doc_ids = df["doc_id"].tolist()
    texts = df["text"].tolist()

    retriever = BM25Retriever()
    retriever.index(doc_ids, texts)

    results_by_query: Dict[str, List[str]] = {}
    relevance_sets: Dict[str, List[str]] = {}
    result_rows: List[Dict[str, Any]] = []

    for query_item in queries:
        query_id = str(query_item["query_id"])
        query_text = str(query_item["query"])
        relevant_doc_ids = [str(doc_id) for doc_id in query_item["relevant_doc_ids"]]
        relevant_set = set(relevant_doc_ids)

        hits = retriever.search(query_text, top_k=top_k)
        retrieved_ids = [str(hit.doc_id) for hit in hits]

        results_by_query[query_id] = retrieved_ids
        relevance_sets[query_id] = relevant_doc_ids

        for hit in hits:
            result_rows.append(
                {
                    "query_id": query_id,
                    "query": query_text,
                    "rank": hit.rank,
                    "doc_id": str(hit.doc_id),
                    "score": hit.score,
                    "is_relevant": str(hit.doc_id) in relevant_set,
                    "text": hit.text,
                }
            )

    metrics = evaluate_retrieval_run(
        results=results_by_query,
        relevance_sets=relevance_sets,
        k=top_k,
    )

    results_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(result_rows).to_csv(results_path, index=False)
    pd.DataFrame([{"method": "BM25", **metrics}]).to_csv(metrics_path, index=False)

    print("BM25 baseline experiment completed.")
    print(f"Results saved to: {results_path}")
    print(f"Metrics saved to: {metrics_path}")
    print(metrics)

    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run BM25 baseline experiment.")

    parser.add_argument(
        "--documents",
        type=Path,
        default=Path("data/processed/processed_documents.csv"),
    )
    parser.add_argument(
        "--queries",
        type=Path,
        default=Path("data/evaluation/retrieval_queries.example.json"),
    )
    parser.add_argument(
        "--results",
        type=Path,
        default=Path("reports/tables/bm25_results.csv"),
    )
    parser.add_argument(
        "--metrics",
        type=Path,
        default=Path("reports/tables/bm25_metrics.csv"),
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    run_bm25_baseline(
        documents_path=args.documents,
        queries_path=args.queries,
        results_path=args.results,
        metrics_path=args.metrics,
        top_k=args.top_k,
    )


if __name__ == "__main__":
    main()