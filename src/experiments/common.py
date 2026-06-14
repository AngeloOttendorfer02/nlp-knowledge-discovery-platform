"""Shared loading, validation, evaluation, and output helpers for experiments."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence

import pandas as pd


@dataclass(frozen=True)
class EvaluationQuery:
    """One retrieval query and its manually judged relevant documents."""

    query_id: str
    query: str
    relevant_doc_ids: frozenset[str]


def load_documents(path: str | Path) -> pd.DataFrame:
    """Load and validate processed documents for retrieval experiments."""
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(
            f"Processed document file not found: {source}. "
            "Run the NLP pipeline first."
        )

    # Keep arXiv identifiers byte-for-byte (for example ``0704.0001``).
    # Without an explicit string dtype, pandas may parse numeric-looking IDs as
    # floats, remove leading zeros, or truncate trailing zeros.
    frame = pd.read_csv(source, dtype={"doc_id": "string"})
    if "doc_id" not in frame.columns:
        raise ValueError(f"{source} must contain a 'doc_id' column")

    frame = frame.copy()
    frame["doc_id"] = frame["doc_id"].fillna("").astype(str).str.strip()
    if (frame["doc_id"] == "").any():
        raise ValueError(f"{source} contains empty document IDs")
    if frame["doc_id"].duplicated().any():
        duplicates = frame.loc[frame["doc_id"].duplicated(), "doc_id"].tolist()
        raise ValueError(f"Duplicate document IDs are not allowed: {duplicates[:5]}")

    if "text" not in frame.columns:
        if not {"title", "abstract"}.issubset(frame.columns):
            raise ValueError(
                f"{source} must contain either 'text' or both 'title' and 'abstract'"
            )
        frame["text"] = (
            frame["title"].fillna("").astype(str).str.strip()
            + ". "
            + frame["abstract"].fillna("").astype(str).str.strip()
        ).str.strip(". ")
    else:
        frame["text"] = frame["text"].fillna("").astype(str)

    if frame.empty:
        raise ValueError(f"{source} contains no documents")
    if not frame["text"].str.strip().any():
        raise ValueError(f"{source} contains no non-empty document text")

    return frame


def load_evaluation_queries(path: str | Path) -> List[EvaluationQuery]:
    """Load the roadmap-compatible relevance JSON format."""
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(
            f"Evaluation query file not found: {source}. "
            "Create it before running retrieval experiments."
        )

    with source.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    if isinstance(payload, dict):
        payload = payload.get("queries")
    if not isinstance(payload, list) or not payload:
        raise ValueError("Evaluation JSON must be a non-empty list or {'queries': [...]} object")

    queries: List[EvaluationQuery] = []
    seen_ids = set()

    for index, item in enumerate(payload, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"Evaluation query #{index} must be an object")

        query = str(item.get("query", "")).strip()
        query_id = str(item.get("query_id", f"q{index:03d}")).strip()
        relevant = item.get("relevant_doc_ids", item.get("relevant_ids"))

        if not query:
            raise ValueError(f"Evaluation query #{index} has an empty 'query'")
        if not query_id:
            raise ValueError(f"Evaluation query #{index} has an empty 'query_id'")
        if query_id in seen_ids:
            raise ValueError(f"Duplicate query_id: {query_id}")
        if not isinstance(relevant, list) or not relevant:
            raise ValueError(
                f"Evaluation query '{query_id}' must contain at least one relevant_doc_id"
            )

        relevant_ids = frozenset(
            str(doc_id).strip() for doc_id in relevant if str(doc_id).strip()
        )
        if not relevant_ids:
            raise ValueError(
                f"Evaluation query '{query_id}' must contain non-empty relevant_doc_ids"
            )

        seen_ids.add(query_id)
        queries.append(
            EvaluationQuery(
                query_id=query_id,
                query=query,
                relevant_doc_ids=relevant_ids,
            )
        )

    return queries


def validate_relevance_against_corpus(
    queries: Sequence[EvaluationQuery],
    corpus_doc_ids: Iterable[str],
) -> None:
    """Fail early when any judged relevant document is absent from the corpus."""
    corpus = {str(doc_id).strip() for doc_id in corpus_doc_ids}
    missing_by_query = {
        query.query_id: sorted(query.relevant_doc_ids.difference(corpus))
        for query in queries
        if query.relevant_doc_ids.difference(corpus)
    }
    if missing_by_query:
        details = "; ".join(
            f"{query_id}: {', '.join(missing_ids[:5])}"
            for query_id, missing_ids in missing_by_query.items()
        )
        raise ValueError(
            "Relevant document IDs missing from the corpus: " + details
        )


def normalize_top_ks(values: Sequence[int]) -> List[int]:
    """Validate and deduplicate top-k values while preserving sorted order."""
    if not values:
        raise ValueError("At least one top-k value is required")
    top_ks = sorted({int(value) for value in values})
    if top_ks[0] <= 0:
        raise ValueError("All top-k values must be positive")
    return top_ks


def evaluate_rankings(
    *,
    method: str,
    queries: Sequence[EvaluationQuery],
    rankings: Mapping[str, Sequence[object]],
    top_ks: Sequence[int],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build ranked-result and summary-metric tables.

    The canonical functions from ``src.evaluation.retrieval_metrics`` are used
    when Duong's evaluation PR is present. A mathematically equivalent local
    fallback keeps this PR independently testable before that merge.
    """
    precision_at_k, recall_at_k, mean_reciprocal_rank = _metric_functions()
    top_ks = normalize_top_ks(top_ks)

    result_rows: List[Dict[str, object]] = []
    retrieved_per_query: List[List[str]] = []
    relevance_sets: List[set[str]] = []

    for query in queries:
        hits = list(rankings.get(query.query_id, []))
        retrieved_ids = [str(hit.doc_id) for hit in hits]
        retrieved_per_query.append(retrieved_ids)
        relevance_sets.append(set(query.relevant_doc_ids))

        for hit in hits:
            row: Dict[str, object] = {
                "method": method,
                "query_id": query.query_id,
                "query": query.query,
                "rank": int(hit.rank),
                "doc_id": str(hit.doc_id),
                "score": float(hit.score),
                "relevant": str(hit.doc_id) in query.relevant_doc_ids,
            }

            for attribute in (
                "base_score",
                "normalized_base_score",
                "graph_score",
                "normalized_graph_score",
            ):
                if hasattr(hit, attribute):
                    row[attribute] = float(getattr(hit, attribute))

            if hasattr(hit, "evidence"):
                row["graph_evidence"] = json.dumps(
                    list(getattr(hit, "evidence")),
                    ensure_ascii=False,
                )

            result_rows.append(row)

    mrr = float(mean_reciprocal_rank(retrieved_per_query, relevance_sets))
    metric_rows: List[Dict[str, object]] = []

    for k in top_ks:
        per_query_precision = []
        per_query_recall = []
        for retrieved_ids, relevant_ids in zip(retrieved_per_query, relevance_sets):
            per_query_precision.append(
                float(precision_at_k(retrieved_ids, relevant_ids, k))
            )
            per_query_recall.append(float(recall_at_k(retrieved_ids, relevant_ids, k)))

        metric_rows.append(
            {
                "method": method,
                "k": k,
                "precision_at_k": sum(per_query_precision) / len(per_query_precision),
                "recall_at_k": sum(per_query_recall) / len(per_query_recall),
                "mrr": mrr,
                "num_queries": len(queries),
            }
        )

    return pd.DataFrame(result_rows), pd.DataFrame(metric_rows)


def write_experiment_outputs(
    results: pd.DataFrame,
    metrics: pd.DataFrame,
    *,
    output_dir: str | Path,
    result_filename: str,
    metric_filename: str,
) -> tuple[Path, Path]:
    """Write deterministic CSV outputs and return their paths."""
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    result_path = target / result_filename
    metric_path = target / metric_filename
    results.to_csv(result_path, index=False)
    metrics.to_csv(metric_path, index=False)
    return result_path, metric_path


def _metric_functions():
    try:
        from src.evaluation.retrieval_metrics import (  # type: ignore
            mean_reciprocal_rank,
            precision_at_k,
            recall_at_k,
        )

        return precision_at_k, recall_at_k, mean_reciprocal_rank
    except ModuleNotFoundError as exc:
        if exc.name not in {"src.evaluation", "src.evaluation.retrieval_metrics"}:
            raise
        return _precision_at_k, _recall_at_k, _mean_reciprocal_rank


def _precision_at_k(retrieved_ids, relevant_ids, k):
    if k <= 0:
        raise ValueError("k must be positive")
    retrieved = list(retrieved_ids)[:k]
    relevant = set(relevant_ids)
    return sum(doc_id in relevant for doc_id in retrieved) / k


def _recall_at_k(retrieved_ids, relevant_ids, k):
    if k <= 0:
        raise ValueError("k must be positive")
    relevant = set(relevant_ids)
    if not relevant:
        return 0.0
    retrieved = set(list(retrieved_ids)[:k])
    return len(retrieved.intersection(relevant)) / len(relevant)


def _mean_reciprocal_rank(results, relevance_sets):
    reciprocal_ranks = []
    for retrieved_ids, relevant_ids in zip(results, relevance_sets):
        relevant = set(relevant_ids)
        reciprocal_rank = 0.0
        for rank, doc_id in enumerate(retrieved_ids, start=1):
            if doc_id in relevant:
                reciprocal_rank = 1.0 / rank
                break
        reciprocal_ranks.append(reciprocal_rank)
    return sum(reciprocal_ranks) / len(reciprocal_ranks) if reciprocal_ranks else 0.0
