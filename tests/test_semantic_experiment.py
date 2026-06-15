from __future__ import annotations

import json

import numpy as np
import pandas as pd

import src.retrieval.embedding_retriever as embedding_module
from src.experiments.common import (
    EvaluationQuery,
    load_documents,
    validate_relevance_against_corpus,
)
from src.experiments.run_semantic_retrieval import run_semantic_experiment
from src.retrieval.embedding_retriever import EmbeddingRetriever


class FakeVectorHit:
    def __init__(self, doc_id, score, rank):
        self.doc_id = doc_id
        self.score = score
        self.rank = rank


class FakeVectorStore:
    def __init__(self, dim):
        self.dim = dim
        self.doc_ids = []
        self.embeddings = None

    def add(self, doc_ids, embeddings):
        vectors = np.asarray(embeddings, dtype="float32")
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self.embeddings = vectors / norms
        self.doc_ids = list(doc_ids)

    def search(self, query_embedding, top_k=10):
        query = np.asarray(query_embedding, dtype="float32")
        norm = np.linalg.norm(query) or 1.0
        query = query / norm
        scores = self.embeddings @ query
        indices = np.argsort(scores)[::-1][:top_k]
        return [
            FakeVectorHit(self.doc_ids[index], float(scores[index]), rank)
            for rank, index in enumerate(indices, start=1)
        ]


class FakeEmbeddingModel:
    def __init__(self):
        self.calls = 0

    def encode(self, texts, **kwargs):
        self.calls += 1
        vectors = []
        for text in texts:
            lowered = text.lower()
            vectors.append(
                [
                    float("graph" in lowered),
                    float("language" in lowered),
                    float("vision" in lowered),
                ]
            )
        return np.asarray(vectors, dtype="float32")


def _write_fixture_files(tmp_path):
    documents_path = tmp_path / "processed_documents.csv"
    pd.DataFrame(
        [
            {"doc_id": "d1", "text": "knowledge graph retrieval"},
            {"doc_id": "d2", "text": "language model embeddings"},
            {"doc_id": "d3", "text": "computer vision segmentation"},
        ]
    ).to_csv(documents_path, index=False)

    queries_path = tmp_path / "queries.json"
    queries_path.write_text(
        json.dumps(
            [
                {
                    "query_id": "q1",
                    "query": "graph search",
                    "relevant_doc_ids": ["d1"],
                },
                {
                    "query_id": "q2",
                    "query": "language embeddings",
                    "relevant_doc_ids": ["d2"],
                },
            ]
        ),
        encoding="utf-8",
    )
    return documents_path, queries_path


def test_embedding_cache_is_reused(tmp_path, monkeypatch):
    monkeypatch.setattr(embedding_module, "VectorStore", FakeVectorStore)
    model = FakeEmbeddingModel()
    cache_path = tmp_path / "embeddings.npz"
    retriever = EmbeddingRetriever(model_name="fake", model=model)

    retriever.index(["d1", "d2"], ["graph", "language"], cache_path=cache_path)
    assert cache_path.exists()
    assert retriever.last_cache_hit is False
    first_call_count = model.calls

    retriever.index(["d1", "d2"], ["graph", "language"], cache_path=cache_path)
    assert retriever.last_cache_hit is True
    assert model.calls == first_call_count


def test_semantic_experiment_writes_rankings_and_metrics(tmp_path, monkeypatch):
    monkeypatch.setattr(embedding_module, "VectorStore", FakeVectorStore)
    documents_path, queries_path = _write_fixture_files(tmp_path)
    retriever = EmbeddingRetriever(model_name="fake", model=FakeEmbeddingModel())

    results, metrics, result_path, metric_path = run_semantic_experiment(
        documents_path=documents_path,
        queries_path=queries_path,
        output_dir=tmp_path / "reports",
        top_ks=(1, 2),
        cache_path=tmp_path / "cache.npz",
        retriever=retriever,
    )

    assert result_path.exists()
    assert metric_path.exists()
    assert set(results.columns) >= {
        "query_id",
        "query",
        "rank",
        "doc_id",
        "score",
        "relevant",
    }
    assert metrics["method"].unique().tolist() == ["semantic"]
    assert metrics["k"].tolist() == [1, 2]
    assert metrics.loc[metrics["k"] == 1, "precision_at_k"].iloc[0] == 1.0


def test_document_loader_preserves_arxiv_ids(tmp_path):
    documents_path = tmp_path / "processed_documents.csv"
    documents_path.write_text(
        "doc_id,title,abstract\n"
        "0704.0001,First paper,First abstract\n"
        "0704.0010,Second paper,Second abstract\n",
        encoding="utf-8",
    )

    documents = load_documents(documents_path)

    assert documents["doc_id"].tolist() == ["0704.0001", "0704.0010"]


def test_relevance_validation_rejects_any_missing_document():
    queries = [
        EvaluationQuery(
            query_id="q1",
            query="graph retrieval",
            relevant_doc_ids=frozenset({"d1", "missing"}),
        )
    ]

    try:
        validate_relevance_against_corpus(queries, ["d1", "d2"])
    except ValueError as exc:
        assert "missing" in str(exc)
    else:
        raise AssertionError("Expected missing relevant IDs to raise ValueError")
