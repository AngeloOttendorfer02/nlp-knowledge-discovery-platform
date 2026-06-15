import json

import numpy as np
import pandas as pd

from src.experiments import run_semantic_retrieval as semantic_module


class FakeEmbeddingRetriever:
    encode_call_count = 0

    def __init__(self, model_name="fake-model"):
        self.model_name = model_name
        self._store = None
        self._doc_texts = {}

    def encode(self, texts):
        FakeEmbeddingRetriever.encode_call_count += 1
        vectors = []

        for text in texts:
            text = text.lower()

            if "graph" in text or "retrieval" in text:
                vectors.append([1.0, 0.0])
            elif "transformer" in text or "entity" in text:
                vectors.append([0.0, 1.0])
            else:
                vectors.append([0.5, 0.5])

        return np.array(vectors, dtype="float32")

    def search(self, query, top_k=5):
        from src.retrieval.bm25_retriever import RetrievalResult

        query = query.lower()

        if "graph" in query or "retrieval" in query:
            ranked = ["1", "2"]
        else:
            ranked = ["2", "1"]

        return [
            RetrievalResult(
                doc_id=doc_id,
                score=1.0 / rank,
                rank=rank,
                text=self._doc_texts.get(doc_id, ""),
            )
            for rank, doc_id in enumerate(ranked[:top_k], start=1)
        ]


def test_semantic_retrieval_experiment_writes_results_and_metrics(tmp_path, monkeypatch):
    FakeEmbeddingRetriever.encode_call_count = 0
    monkeypatch.setattr(
        semantic_module,
        "EmbeddingRetriever",
        FakeEmbeddingRetriever,
    )

    documents_path = tmp_path / "processed_documents.csv"
    queries_path = tmp_path / "queries.json"
    results_path = tmp_path / "semantic_results.csv"
    metrics_path = tmp_path / "semantic_metrics.csv"
    cache_dir = tmp_path / "embedding_cache"

    pd.DataFrame(
        [
            {
                "doc_id": "1",
                "text": "Graph neural networks for scientific document retrieval.",
            },
            {
                "doc_id": "2",
                "text": "Transformer models for named entity recognition.",
            },
        ]
    ).to_csv(documents_path, index=False)

    queries_path.write_text(
        json.dumps(
            [
                {
                    "query_id": "q1",
                    "query": "graph retrieval",
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

    metrics = semantic_module.run_semantic_retrieval(
        documents_path=documents_path,
        queries_path=queries_path,
        results_path=results_path,
        metrics_path=metrics_path,
        cache_dir=cache_dir,
        top_k=2,
        model_name="fake-model",
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

    assert {"method", "model", "precision@2", "recall@2", "mrr"}.issubset(
        metrics_df.columns
    )

    assert metrics["recall@2"] == 1.0
    assert metrics_df.iloc[0]["method"] == "Semantic"


def test_semantic_retrieval_uses_cached_embeddings(tmp_path, monkeypatch):
    FakeEmbeddingRetriever.encode_call_count = 0
    monkeypatch.setattr(
        semantic_module,
        "EmbeddingRetriever",
        FakeEmbeddingRetriever,
    )

    documents_path = tmp_path / "processed_documents.csv"
    queries_path = tmp_path / "queries.json"
    results_path = tmp_path / "semantic_results.csv"
    metrics_path = tmp_path / "semantic_metrics.csv"
    cache_dir = tmp_path / "embedding_cache"

    pd.DataFrame(
        [
            {"doc_id": "1", "text": "Graph document retrieval."},
            {"doc_id": "2", "text": "Transformer entity recognition."},
        ]
    ).to_csv(documents_path, index=False)

    queries_path.write_text(
        json.dumps(
            [
                {
                    "query_id": "q1",
                    "query": "graph retrieval",
                    "relevant_doc_ids": ["1"],
                }
            ]
        ),
        encoding="utf-8",
    )

    semantic_module.run_semantic_retrieval(
        documents_path=documents_path,
        queries_path=queries_path,
        results_path=results_path,
        metrics_path=metrics_path,
        cache_dir=cache_dir,
        top_k=2,
        model_name="fake-model",
    )

    semantic_module.run_semantic_retrieval(
        documents_path=documents_path,
        queries_path=queries_path,
        results_path=results_path,
        metrics_path=metrics_path,
        cache_dir=cache_dir,
        top_k=2,
        model_name="fake-model",
    )

    assert FakeEmbeddingRetriever.encode_call_count == 1