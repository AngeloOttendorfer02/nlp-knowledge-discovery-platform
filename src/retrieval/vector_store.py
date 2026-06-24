"""
vector_store.py — Lightweight vector store with optional FAISS support.

The store keeps L2-normalised document embeddings and supports cosine-similarity
search. It works without FAISS by using a NumPy backend, which keeps the project
portable for tests and small experiments. If FAISS is installed and requested,
FAISS is used transparently. A NumPy copy is always persisted as a fallback so a
FAISS index can still be searched on systems where FAISS is not installed.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import List, Sequence

import numpy as np


@dataclass(frozen=True)
class VectorHit:
    """A single nearest-neighbour search result."""

    doc_id: str
    score: float
    rank: int


class VectorStore:
    """Small cosine-similarity vector store with optional FAISS backend."""

    def __init__(self, dim: int, use_faiss: bool = True) -> None:
        if dim <= 0:
            raise ValueError("dim must be positive")

        self.dim = int(dim)
        self.use_faiss = bool(use_faiss)
        self._doc_ids: list[str] = []
        self._vectors = np.empty((0, self.dim), dtype="float32")
        self._index = None
        self._backend = "numpy"

        if self.use_faiss:
            try:
                import faiss  # type: ignore

                self._index = faiss.IndexFlatIP(self.dim)
                self._backend = "faiss"
            except Exception:
                self._index = None
                self._backend = "numpy"

    @staticmethod
    def _normalize(vectors: np.ndarray) -> np.ndarray:
        vectors = np.asarray(vectors, dtype="float32")
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0.0] = 1.0
        return vectors / norms

    def add(self, doc_ids: Sequence[str], embeddings: np.ndarray) -> None:
        embeddings = np.asarray(embeddings, dtype="float32")

        if embeddings.ndim != 2 or embeddings.shape[1] != self.dim:
            raise ValueError(f"Expected embeddings of shape (n, {self.dim})")
        if len(doc_ids) != embeddings.shape[0]:
            raise ValueError("doc_ids and embeddings must have matching lengths")

        normalized = self._normalize(embeddings)
        ids = [str(doc_id) for doc_id in doc_ids]

        # Always keep the NumPy matrix, even when FAISS is active. This makes
        # persistence robust: index.faiss is fast, vectors.npy is the portable
        # fallback when FAISS cannot be imported later.
        self._vectors = np.vstack([self._vectors, normalized])

        if self._backend == "faiss" and self._index is not None:
            self._index.add(normalized)

        self._doc_ids.extend(ids)

    def search(self, query_embedding: np.ndarray, top_k: int = 10) -> List[VectorHit]:
        if top_k <= 0 or len(self) == 0:
            return []

        query = np.asarray(query_embedding, dtype="float32")
        if query.ndim == 1:
            query = query.reshape(1, -1)
        if query.ndim != 2 or query.shape != (1, self.dim):
            raise ValueError(f"Expected query embedding of shape ({self.dim},) or (1, {self.dim})")

        query = self._normalize(query)
        k = min(int(top_k), len(self))

        if self._backend == "faiss" and self._index is not None:
            scores, indices = self._index.search(query, k)
            pairs = [(int(idx), float(score)) for idx, score in zip(indices[0], scores[0]) if idx != -1]
        else:
            scores = self._vectors @ query[0]
            order = np.argsort(-scores)[:k]
            pairs = [(int(idx), float(scores[idx])) for idx in order]

        return [
            VectorHit(doc_id=self._doc_ids[idx], score=score, rank=rank)
            for rank, (idx, score) in enumerate(pairs, start=1)
        ]

    def __len__(self) -> int:
        return len(self._doc_ids)

    def save(self, directory: str) -> None:
        os.makedirs(directory, exist_ok=True)

        vectors_path = os.path.join(directory, "vectors.npy")
        np.save(vectors_path, self._vectors.astype("float32"))

        meta = {
            "dim": self.dim,
            "doc_ids": self._doc_ids,
            "backend": "numpy",
            "has_numpy_fallback": True,
        }

        if self._backend == "faiss" and self._index is not None:
            try:
                import faiss  # type: ignore

                faiss.write_index(self._index, os.path.join(directory, "index.faiss"))
                meta["backend"] = "faiss"
            except Exception:
                meta["backend"] = "numpy"

        with open(os.path.join(directory, "doc_ids.json"), "w", encoding="utf-8") as handle:
            json.dump(meta, handle, indent=2)

    @classmethod
    def load(cls, directory: str) -> "VectorStore":
        with open(os.path.join(directory, "doc_ids.json"), "r", encoding="utf-8") as handle:
            meta = json.load(handle)

        backend = meta.get("backend", "numpy")
        store = cls(dim=int(meta["dim"]), use_faiss=backend == "faiss")
        store._doc_ids = [str(doc_id) for doc_id in meta["doc_ids"]]

        faiss_path = os.path.join(directory, "index.faiss")
        vectors_path = os.path.join(directory, "vectors.npy")

        if os.path.exists(vectors_path):
            store._vectors = np.load(vectors_path).astype("float32")
        else:
            store._vectors = np.empty((0, store.dim), dtype="float32")

        if backend == "faiss" and os.path.exists(faiss_path):
            try:
                import faiss  # type: ignore

                store._index = faiss.read_index(faiss_path)
                store._backend = "faiss"
                return store
            except Exception:
                pass

        if len(store._doc_ids) and store._vectors.shape[0] != len(store._doc_ids):
            raise ValueError(
                "VectorStore fallback is incomplete: vectors.npy is missing or does not match doc_ids."
            )

        store._backend = "numpy"
        store._index = None
        return store
