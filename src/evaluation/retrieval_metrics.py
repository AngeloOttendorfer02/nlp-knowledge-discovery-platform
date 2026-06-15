"""
retrieval_metrics.py — Evaluation metrics for retrieval experiments.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Sequence


@dataclass(frozen=True)
class RetrievalMetrics:
    precision_at_k: float
    recall_at_k: float
    reciprocal_rank: float


def _to_set(ids: Iterable[str]) -> set[str]:
    return {str(doc_id) for doc_id in ids}


def precision_at_k(
    retrieved_ids: Sequence[str],
    relevant_ids: Iterable[str],
    k: int,
) -> float:
    if k <= 0:
        raise ValueError("k must be a positive integer")

    retrieved_top_k = [str(doc_id) for doc_id in retrieved_ids[:k]]

    if not retrieved_top_k:
        return 0.0

    relevant = _to_set(relevant_ids)
    hits = sum(1 for doc_id in retrieved_top_k if doc_id in relevant)

    return hits / len(retrieved_top_k)


def recall_at_k(
    retrieved_ids: Sequence[str],
    relevant_ids: Iterable[str],
    k: int,
) -> float:
    if k <= 0:
        raise ValueError("k must be a positive integer")

    relevant = _to_set(relevant_ids)

    if not relevant:
        return 0.0

    retrieved_top_k = [str(doc_id) for doc_id in retrieved_ids[:k]]
    hits = sum(1 for doc_id in retrieved_top_k if doc_id in relevant)

    return hits / len(relevant)


def reciprocal_rank(
    retrieved_ids: Sequence[str],
    relevant_ids: Iterable[str],
) -> float:
    relevant = _to_set(relevant_ids)

    if not relevant:
        return 0.0

    for rank, doc_id in enumerate(retrieved_ids, start=1):
        if str(doc_id) in relevant:
            return 1.0 / rank

    return 0.0


def evaluate_single_query(
    retrieved_ids: Sequence[str],
    relevant_ids: Iterable[str],
    k: int = 10,
) -> RetrievalMetrics:
    return RetrievalMetrics(
        precision_at_k=precision_at_k(retrieved_ids, relevant_ids, k),
        recall_at_k=recall_at_k(retrieved_ids, relevant_ids, k),
        reciprocal_rank=reciprocal_rank(retrieved_ids, relevant_ids),
    )


def mean_reciprocal_rank(
    results: Mapping[str, Sequence[str]],
    relevance_sets: Mapping[str, Iterable[str]],
) -> float:
    if not results:
        return 0.0

    scores = [
        reciprocal_rank(retrieved_ids, relevance_sets.get(query_id, []))
        for query_id, retrieved_ids in results.items()
    ]

    return sum(scores) / len(scores)


def evaluate_retrieval_run(
    results: Mapping[str, Sequence[str]],
    relevance_sets: Mapping[str, Iterable[str]],
    k: int = 10,
) -> Dict[str, float]:
    if k <= 0:
        raise ValueError("k must be a positive integer")

    if not results:
        return {
            f"precision@{k}": 0.0,
            f"recall@{k}": 0.0,
            "mrr": 0.0,
        }

    precision_scores: List[float] = []
    recall_scores: List[float] = []
    reciprocal_rank_scores: List[float] = []

    for query_id, retrieved_ids in results.items():
        relevant_ids = relevance_sets.get(query_id, [])

        precision_scores.append(
            precision_at_k(retrieved_ids, relevant_ids, k)
        )
        recall_scores.append(
            recall_at_k(retrieved_ids, relevant_ids, k)
        )
        reciprocal_rank_scores.append(
            reciprocal_rank(retrieved_ids, relevant_ids)
        )

    return {
        f"precision@{k}": sum(precision_scores) / len(precision_scores),
        f"recall@{k}": sum(recall_scores) / len(recall_scores),
        "mrr": sum(reciprocal_rank_scores) / len(reciprocal_rank_scores),
    }