from src.evaluation.retrieval_metrics import (
    evaluate_retrieval_run,
    mean_reciprocal_rank,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)


def test_precision_at_k_partial_match():
    retrieved = ["A", "B", "C", "D"]
    relevant = ["B", "D"]

    assert precision_at_k(retrieved, relevant, k=4) == 0.5


def test_recall_at_k_full_match():
    retrieved = ["A", "B", "C", "D"]
    relevant = ["B", "D"]

    assert recall_at_k(retrieved, relevant, k=4) == 1.0


def test_reciprocal_rank_first_relevant_at_rank_two():
    retrieved = ["A", "B", "C", "D"]
    relevant = ["B", "D"]

    assert reciprocal_rank(retrieved, relevant) == 0.5


def test_reciprocal_rank_no_match():
    retrieved = ["A", "B", "C"]
    relevant = ["X", "Y"]

    assert reciprocal_rank(retrieved, relevant) == 0.0


def test_mean_reciprocal_rank():
    results = {
        "q1": ["A", "B", "C"],
        "q2": ["X", "Y", "Z"],
    }

    relevance_sets = {
        "q1": ["B"],
        "q2": ["X"],
    }

    assert mean_reciprocal_rank(results, relevance_sets) == 0.75


def test_evaluate_retrieval_run():
    results = {
        "q1": ["A", "B", "C"],
        "q2": ["X", "Y", "Z"],
    }

    relevance_sets = {
        "q1": ["B"],
        "q2": ["X"],
    }

    metrics = evaluate_retrieval_run(results, relevance_sets, k=2)

    assert metrics["precision@2"] == 0.5
    assert metrics["recall@2"] == 1.0
    assert metrics["mrr"] == 0.75


def test_empty_results():
    metrics = evaluate_retrieval_run({}, {}, k=5)

    assert metrics["precision@5"] == 0.0
    assert metrics["recall@5"] == 0.0
    assert metrics["mrr"] == 0.0