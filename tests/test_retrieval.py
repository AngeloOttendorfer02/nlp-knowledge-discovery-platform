from __future__ import annotations

import numpy as np
import pytest

from src.retrieval.bm25_retriever import BM25Retriever, RetrievalResult
from src.retrieval.embedding_retriever import EmbeddingRetriever
from src.retrieval.vector_store import VectorStore


class _FakeEmbeddingModel:
    def __init__(self) -> None:
        self.calls = 0

    def encode(self, texts, **kwargs):
        self.calls += 1
        vectors = []
        for text in texts:
            low = text.lower()
            if "graph" in low or "retrieval" in low:
                vectors.append([1.0, 0.0, 0.0])
            elif "transformer" in low or "entity" in low:
                vectors.append([0.0, 1.0, 0.0])
            else:
                vectors.append([0.0, 0.0, 1.0])
        return np.asarray(vectors, dtype="float32")


def _tokenize(text: str):
    return [token.lower().strip(".,") for token in text.split()]


def test_bm25_retriever_indexes_and_ranks_matching_documents_first():
    retriever = BM25Retriever(tokenizer=_tokenize)
    retriever.index(
        ["d1", "d2", "d3"],
        [
            "graph retrieval for scientific papers",
            "transformer entity extraction for documents",
            "topic modeling with latent dirichlet allocation",
        ],
    )

    results = retriever.search("graph retrieval", top_k=2)

    assert [result.doc_id for result in results] == ["d1", "d3"] or [result.doc_id for result in results] == ["d1", "d2"]
    assert results[0].rank == 1
    assert results[0].score > results[1].score
    assert "graph retrieval" in results[0].text


def test_bm25_retriever_validates_index_and_search_inputs():
    with pytest.raises(ValueError, match="k1"):
        BM25Retriever(k1=0)

    with pytest.raises(ValueError, match="between 0 and 1"):
        BM25Retriever(b=1.5)

    retriever = BM25Retriever(tokenizer=_tokenize)

    with pytest.raises(RuntimeError, match="Call index"):
        retriever.search("graph")

    with pytest.raises(ValueError, match="same length"):
        retriever.index(["d1"], ["text", "extra"])

    with pytest.raises(ValueError, match="empty"):
        retriever.index([], [])

    retriever.index(["d1"], ["graph retrieval"])
    with pytest.raises(ValueError, match="positive"):
        retriever.search("graph", top_k=0)


def test_embedding_retriever_indexes_precomputed_embeddings_and_searches():
    retriever = EmbeddingRetriever(model=_FakeEmbeddingModel())
    retriever.index_embeddings(
        ["d1", "d2"],
        ["graph retrieval", "transformer entity extraction"],
        np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype="float32"),
    )

    results = retriever.search("graph retrieval", top_k=1)

    assert results == [RetrievalResult(doc_id="d1", score=1.0, rank=1, text="graph retrieval")]


def test_embedding_retriever_uses_cache_for_unchanged_corpus(tmp_path):
    cache_path = tmp_path / "embeddings.npz"
    model = _FakeEmbeddingModel()
    first = EmbeddingRetriever(model_name="fake-model", model=model)
    first.index(["d1", "d2"], ["graph retrieval", "transformer entity extraction"], cache_path=cache_path)

    second = EmbeddingRetriever(model_name="fake-model", model=model)
    second.index(["d1", "d2"], ["graph retrieval", "transformer entity extraction"], cache_path=cache_path)

    assert cache_path.exists()
    assert first.last_cache_hit is False
    assert second.last_cache_hit is True
    # One call encodes the first corpus. The second indexing run reuses the cache.
    assert model.calls == 1


def test_embedding_retriever_recomputes_stale_cache_when_corpus_changes(tmp_path):
    cache_path = tmp_path / "embeddings.npz"
    model = _FakeEmbeddingModel()
    retriever = EmbeddingRetriever(model_name="fake-model", model=model)
    retriever.index(["d1"], ["graph retrieval"], cache_path=cache_path)

    retriever.index(["d1"], ["unrelated topic modeling"], cache_path=cache_path)

    assert retriever.last_cache_hit is False
    assert model.calls == 2


def test_embedding_retriever_validates_inputs_before_indexing_and_searching():
    retriever = EmbeddingRetriever(model=_FakeEmbeddingModel())

    with pytest.raises(ValueError, match="batch_size"):
        EmbeddingRetriever(batch_size=0, model=_FakeEmbeddingModel())

    with pytest.raises(ValueError, match="same length"):
        retriever.index(["d1"], ["text", "extra"])

    with pytest.raises(ValueError, match="empty"):
        retriever.index([], [])

    with pytest.raises(ValueError, match="two-dimensional"):
        retriever.index_embeddings(["d1"], ["text"], np.array([1.0, 2.0]))

    with pytest.raises(ValueError, match="matching lengths"):
        retriever.index_embeddings(["d1"], ["text"], np.array([[1.0], [2.0]]))

    retriever.index_embeddings(["d1"], ["text"], np.array([[1.0, 0.0]], dtype="float32"))

    with pytest.raises(ValueError, match="positive"):
        retriever.search("text", top_k=0)

    with pytest.raises(ValueError, match="non-empty"):
        retriever.search("   ")


def test_vector_store_returns_empty_for_invalid_top_k_or_empty_index():
    store = VectorStore(dim=2, use_faiss=False)

    assert store.search(np.array([1.0, 0.0], dtype="float32"), top_k=1) == []

    store.add(["d1"], np.array([[1.0, 0.0]], dtype="float32"))

    assert store.search(np.array([1.0, 0.0], dtype="float32"), top_k=0) == []
