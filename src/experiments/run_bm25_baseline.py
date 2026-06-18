
"""Run a reproducible BM25 baseline retrieval experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Sequence

import pandas as pd
import yaml

from src.evaluation.retrieval_metrics import evaluate_retrieval_run
from src.experiments.common import (
    evaluate_rankings,
    load_documents,
    load_evaluation_queries,
    normalize_top_ks,
    validate_relevance_against_corpus,
    write_experiment_outputs,
)
from src.retrieval.bm25_retriever import BM25Retriever


def load_relevance_queries(path: Path) -> List[Dict[str, Any]]:
    """
    Backward-compatible relevance query loader.
    """
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
            raise ValueError(
                "Every query must have at least one relevant document."
            )

        query["relevant_doc_ids"] = [
            str(doc_id).strip()
            for doc_id in query["relevant_doc_ids"]
        ]

    return queries


def run_bm25_baseline(
    *,
    documents_path: str | Path,
    queries_path: str | Path,
    output_dir: str | Path = "reports/tables",
    results_path: str | Path | None = None,
    metrics_path: str | Path | None = None,
    bm25_k1: float = 1.5,
    bm25_b: float = 0.75,
    top_ks: Sequence[int] = (5, 10),
    top_k: int | None = None,
):
    """
    Execute BM25 retrieval and save ranked results plus summary metrics.

    Supports both the new API:

        run_bm25_baseline(output_dir=..., top_ks=(5, 10))

    and the older test-compatible API:

        run_bm25_baseline(results_path=..., metrics_path=..., top_k=2)
    """
    if top_k is not None:
        top_ks = normalize_top_ks([top_k])
    else:
        top_ks = normalize_top_ks(top_ks)

    documents = load_documents(documents_path)
    queries = load_evaluation_queries(queries_path)
    validate_relevance_against_corpus(queries, documents["doc_id"])

    doc_ids = documents["doc_id"].astype(str).tolist()
    texts = documents["text"].fillna("").astype(str).tolist()

    retriever = BM25Retriever(
        k1=bm25_k1,
        b=bm25_b,
    )
    retriever.index(doc_ids, texts)

    retrieval_depth = max(top_ks)

    rankings = {
        query.query_id: retriever.search(query.query, top_k=retrieval_depth)
        for query in queries
    }

    results, metrics = evaluate_rankings(
        method="bm25",
        queries=queries,
        rankings=rankings,
        top_ks=top_ks,
    )

    # Backward-compatible result columns expected by existing tests/notebooks.
    if "relevant" in results.columns and "is_relevant" not in results.columns:
        results["is_relevant"] = results["relevant"]

    if "text" not in results.columns:
        hit_text_lookup = {}

        for hits in rankings.values():
            for hit in hits:
                hit_text_lookup[str(hit.doc_id)] = getattr(hit, "text", "")

        results["text"] = (
            results["doc_id"]
            .astype(str)
            .map(hit_text_lookup)
            .fillna("")
        )

    if results_path is not None and metrics_path is not None:
        result_path = Path(results_path)
        metric_path = Path(metrics_path)

        result_path.parent.mkdir(parents=True, exist_ok=True)
        metric_path.parent.mkdir(parents=True, exist_ok=True)

        results.to_csv(result_path, index=False)

        # Legacy-compatible metric columns expected by older tests.
        legacy_metrics = metrics.copy()
        legacy_metrics["method"] = "BM25"

        if len(top_ks) == 1:
            k = top_ks[0]
            legacy_metrics = legacy_metrics.rename(
                columns={
                    "precision_at_k": f"precision@{k}",
                    "recall_at_k": f"recall@{k}",
                }
            )

        legacy_metrics.to_csv(metric_path, index=False)

        if top_k is not None and len(legacy_metrics) == 1:
            row = legacy_metrics.iloc[0].to_dict()

            return {
                key: value
                for key, value in row.items()
                if key not in {"method", "k", "num_queries"}
            }

    else:
        result_path, metric_path = write_experiment_outputs(
            results,
            metrics,
            output_dir=output_dir,
            result_filename="bm25_results.csv",
            metric_filename="bm25_metrics.csv",
        )

    return results, metrics, result_path, metric_path


def run_bm25_baseline_legacy(
    *,
    documents_path: str | Path,
    queries_path: str | Path,
    results_path: str | Path,
    metrics_path: str | Path,
    top_k: int = 5,
) -> Dict[str, float]:
    """
    Legacy BM25 runner that returns a metrics dictionary.

    This mirrors the original implementation and is useful if older notebooks or
    scripts expected a dictionary return value.
    """
    documents = pd.read_csv(documents_path, dtype={"doc_id": "string"})

    if "doc_id" not in documents.columns or "text" not in documents.columns:
        raise ValueError("Documents must contain 'doc_id' and 'text' columns.")

    documents["doc_id"] = documents["doc_id"].fillna("").astype(str).str.strip()
    documents["text"] = documents["text"].fillna("").astype(str)

    queries = load_relevance_queries(Path(queries_path))

    doc_ids = documents["doc_id"].tolist()
    texts = documents["text"].tolist()

    retriever = BM25Retriever()
    retriever.index(doc_ids, texts)

    results_by_query: Dict[str, List[str]] = {}
    relevance_sets: Dict[str, List[str]] = {}
    result_rows: List[Dict[str, Any]] = []

    for query_item in queries:
        query_id = str(query_item["query_id"])
        query_text = str(query_item["query"])
        relevant_doc_ids = [
            str(doc_id).strip()
            for doc_id in query_item["relevant_doc_ids"]
        ]
        relevant_set = set(relevant_doc_ids)

        hits = retriever.search(query_text, top_k=top_k)
        retrieved_ids = [str(hit.doc_id).strip() for hit in hits]

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
                    "is_relevant": str(hit.doc_id).strip() in relevant_set,
                    "text": hit.text,
                }
            )

    metrics = evaluate_retrieval_run(
        results=results_by_query,
        relevance_sets=relevance_sets,
        k=top_k,
    )

    result_path = Path(results_path)
    metric_path = Path(metrics_path)

    result_path.parent.mkdir(parents=True, exist_ok=True)
    metric_path.parent.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(result_rows).to_csv(result_path, index=False)
    pd.DataFrame([{"method": "BM25", **metrics}]).to_csv(
        metric_path,
        index=False,
    )

    return metrics


def _config_defaults(config_path: str | Path) -> dict:
    defaults = {
        "top_k": 10,
        "bm25_k1": 1.5,
        "bm25_b": 0.75,
    }

    path = Path(config_path)

    if not path.exists():
        return defaults

    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}

    retrieval = config.get("retrieval", {})

    defaults.update(
        {
            "top_k": retrieval.get("top_k", defaults["top_k"]),
            "bm25_k1": retrieval.get("bm25_k1", defaults["bm25_k1"]),
            "bm25_b": retrieval.get("bm25_b", defaults["bm25_b"]),
        }
    )

    return defaults


def parse_args() -> argparse.Namespace:
    defaults = _config_defaults("config.yaml")

    parser = argparse.ArgumentParser(
        description="Evaluate BM25 keyword retrieval."
    )

    parser.add_argument(
        "--documents",
        default="data/processed/processed_documents.csv",
        help="Processed document CSV generated by the pipeline.",
    )

    parser.add_argument(
        "--queries",
        default="data/evaluation/retrieval_queries.example.json",
        help="Evaluation JSON containing query and relevant_doc_ids entries.",
    )

    parser.add_argument(
        "--output-dir",
        default="reports/tables",
    )

    parser.add_argument(
        "--bm25-k1",
        type=float,
        default=defaults["bm25_k1"],
    )

    parser.add_argument(
        "--bm25-b",
        type=float,
        default=defaults["bm25_b"],
    )

    parser.add_argument(
        "--top-k",
        type=int,
        nargs="+",
        default=sorted({5, int(defaults["top_k"])}),
        help="One or more cutoffs, for example: --top-k 5 10",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    _, metrics, result_path, metric_path = run_bm25_baseline(
        documents_path=args.documents,
        queries_path=args.queries,
        output_dir=args.output_dir,
        bm25_k1=args.bm25_k1,
        bm25_b=args.bm25_b,
        top_ks=args.top_k,
    )

    print(f"BM25 ranked results: {result_path}")
    print(f"BM25 summary metrics: {metric_path}")
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()