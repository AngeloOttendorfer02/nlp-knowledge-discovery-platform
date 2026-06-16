"""
embedding_retriever.py — Dense semantic retrieval with Sentence Transformers.

This retriever encodes documents and queries into dense vectors using a
Sentence-Transformers model and ranks documents by cosine similarity. Document
embeddings can optionally be cached in a compressed NumPy file so repeated
experiments do not re-encode an unchanged corpus.
"""

from __future__ import annotations

import hashlib
import os
import re
from functools import lru_cache
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np

from src.retrieval.bm25_retriever import RetrievalResult
from src.retrieval.vector_store import VectorStore


@lru_cache(maxsize=2)
def _load_model(model_name: str):
    """Load and cache a SentenceTransformer model."""
    try:
        from sentence_transformers import SentenceTransformer

        return SentenceTransformer(model_name)
    except (ImportError, OSError):
        return _HashingEmbeddingModel()


def _corpus_fingerprint(
    model_name: str,
    doc_ids: Sequence[str],
    texts: Sequence[str],
) -> str:
    """Return a deterministic SHA-256 fingerprint for a model/corpus pair."""
    digest = hashlib.sha256()
    digest.update(model_name.encode("utf-8"))
    digest.update(b"\0")

    for doc_id, text in zip(doc_ids, texts):
        digest.update(str(doc_id).encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(text).encode("utf-8"))
        digest.update(b"\0")

    return digest.hexdigest()


class EmbeddingRetriever:
    """
    Dense retriever backed by Sentence Transformers and a FAISS vector store.

    Parameters
    ----------
    model_name : str
        Name of the Sentence-Transformers model.
    batch_size : int
        Batch size used when encoding documents.
    model : object, optional
        Preconstructed model exposing ``encode``. Primarily useful for tests or
        dependency injection; normal callers should leave it unset.
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        batch_size: int = 32,
        model=None,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")

        self.model_name = model_name
        self.batch_size = batch_size
        self._model = model if model is not None else _load_model(model_name)
        self._store: Optional[VectorStore] = None
        self._doc_texts: dict[str, str] = {}
        self._last_cache_hit = False

    @property
    def last_cache_hit(self) -> bool:
        """Whether the most recent :meth:`index` call reused cached embeddings."""
        return self._last_cache_hit

    def index(
        self,
        doc_ids: Sequence[str],
        texts: Sequence[str],
        *,
        cache_path: Optional[str | Path] = None,
        force_recompute: bool = False,
    ) -> "EmbeddingRetriever":
        """
        Encode and index a document collection.

        When ``cache_path`` is supplied, embeddings are loaded only if the
        cached model name and deterministic corpus fingerprint match the
        current inputs. Otherwise they are recomputed and the cache is replaced.

        Parameters
        ----------
        doc_ids : sequence of str
            Identifiers aligned with ``texts``.
        texts : sequence of str
            Raw document texts.
        cache_path : str or pathlib.Path, optional
            Compressed ``.npz`` cache location.
        force_recompute : bool
            Ignore a valid cache and encode the documents again.

        Returns
        -------
        EmbeddingRetriever
            The indexed retriever (for chaining).
        """
        if len(doc_ids) != len(texts):
            raise ValueError("doc_ids and texts must have the same length")
        if not doc_ids:
            raise ValueError("Cannot index an empty document collection")

        normalized_ids = [str(doc_id) for doc_id in doc_ids]
        normalized_texts = [str(text) for text in texts]
        fingerprint = _corpus_fingerprint(
            self.model_name,
            normalized_ids,
            normalized_texts,
        )

        embeddings: Optional[np.ndarray] = None
        cache = Path(cache_path) if cache_path is not None else None
        self._last_cache_hit = False

        if cache is not None and cache.exists() and not force_recompute:
            embeddings = self._load_embedding_cache(
                cache,
                expected_doc_ids=normalized_ids,
                expected_fingerprint=fingerprint,
            )
            self._last_cache_hit = embeddings is not None

        if embeddings is None:
            embeddings = self.encode(normalized_texts)
            if cache is not None:
                self._save_embedding_cache(
                    cache,
                    doc_ids=normalized_ids,
                    embeddings=embeddings,
                    fingerprint=fingerprint,
                )

        return self.index_embeddings(normalized_ids, normalized_texts, embeddings)

    def index_embeddings(
        self,
        doc_ids: Sequence[str],
        texts: Sequence[str],
        embeddings: np.ndarray,
    ) -> "EmbeddingRetriever":
        """Index precomputed embeddings while preserving document metadata."""
        if len(doc_ids) != len(texts):
            raise ValueError("doc_ids and texts must have the same length")

        array = np.asarray(embeddings, dtype="float32")
        if array.ndim != 2:
            raise ValueError("embeddings must be a two-dimensional array")
        if array.shape[0] != len(doc_ids):
            raise ValueError("embeddings and doc_ids must have matching lengths")
        if array.shape[0] == 0 or array.shape[1] == 0:
            raise ValueError("embeddings must not be empty")

        self._store = VectorStore(dim=array.shape[1])
        self._store.add([str(doc_id) for doc_id in doc_ids], array)
        self._doc_texts = {
            str(doc_id): str(text) for doc_id, text in zip(doc_ids, texts)
        }
        return self

    def search(self, query: str, top_k: int = 10) -> List[RetrievalResult]:
        """Retrieve the top-k semantically closest documents for a query."""
        if self._store is None:
            raise RuntimeError("Call index() before search().")
        if top_k <= 0:
            raise ValueError("top_k must be a positive integer")
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string")

        query_vec = self._model.encode(
            [query],
            show_progress_bar=False,
            convert_to_numpy=True,
        )[0]
        hits = self._store.search(query_vec, top_k=top_k)

        return [
            RetrievalResult(
                doc_id=hit.doc_id,
                score=hit.score,
                rank=hit.rank,
                text=self._doc_texts.get(hit.doc_id, "")[:300],
            )
            for hit in hits
        ]

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        """Encode arbitrary texts into a two-dimensional NumPy array."""
        if not texts:
            raise ValueError("texts must not be empty")

        embeddings = self._model.encode(
            list(texts),
            batch_size=self.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        array = np.asarray(embeddings, dtype="float32")
        if array.ndim != 2:
            raise ValueError("The embedding model returned an invalid array shape")
        return array

    def _load_embedding_cache(
        self,
        path: Path,
        *,
        expected_doc_ids: Sequence[str],
        expected_fingerprint: str,
    ) -> Optional[np.ndarray]:
        """Load a valid cache, returning ``None`` for stale or malformed data."""
        try:
            with np.load(path, allow_pickle=False) as cache:
                model_name = str(cache["model_name"].item())
                fingerprint = str(cache["fingerprint"].item())
                cached_ids = [str(value) for value in cache["doc_ids"].tolist()]
                embeddings = np.asarray(cache["embeddings"], dtype="float32")
        except (OSError, KeyError, ValueError, TypeError):
            return None

        if model_name != self.model_name:
            return None
        if fingerprint != expected_fingerprint:
            return None
        if cached_ids != list(expected_doc_ids):
            return None
        if embeddings.ndim != 2 or embeddings.shape[0] != len(expected_doc_ids):
            return None

        return embeddings

    def _save_embedding_cache(
        self,
        path: Path,
        *,
        doc_ids: Sequence[str],
        embeddings: np.ndarray,
        fingerprint: str,
    ) -> None:
        """Atomically persist document embeddings and cache metadata."""
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = path.with_name(f".{path.name}.tmp")

        with temporary_path.open("wb") as handle:
            np.savez_compressed(
                handle,
                model_name=np.asarray(self.model_name),
                fingerprint=np.asarray(fingerprint),
                doc_ids=np.asarray(list(doc_ids), dtype=str),
                embeddings=np.asarray(embeddings, dtype="float32"),
            )

        os.replace(temporary_path, path)


class _HashingEmbeddingModel:
    """Deterministic fallback encoder used when SentenceTransformers is unavailable."""

    dimension = 128

    def encode(self, texts: Sequence[str], **kwargs) -> np.ndarray:
        vectors = []
        for text in texts:
            vector = np.zeros(self.dimension, dtype="float32")
            for token in re.findall(r"[a-zA-Z][a-zA-Z0-9-]*", str(text).lower()):
                index = int(hashlib.md5(token.encode("utf-8")).hexdigest(), 16)
                vector[index % self.dimension] += 1.0
            norm = np.linalg.norm(vector)
            if norm > 0.0:
                vector = vector / norm
            vectors.append(vector)
        return np.asarray(vectors, dtype="float32")
