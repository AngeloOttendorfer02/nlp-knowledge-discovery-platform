"""
vector_store.py — Lightweight FAISS-backed vector store.

Stores document embeddings and supports fast nearest-neighbour search by cosine
similarity. Embeddings are L2-normalised on insertion so that inner-product
search (FAISS ``IndexFlatIP``) is equivalent to cosine similarity.

The store keeps a parallel list of document ids so search results can be mapped
back to the original documents. It can be persisted to and loaded from disk.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np


@dataclass
class VectorHit:
    """A single nearest-neighbour search result."""

    doc_id: str
    score: float
    rank: int


class VectorStore:
    """
    A FAISS inner-product index over L2-normalised embeddings.

    Parameters
    ----------
    dim : int
        Dimensionality of the embeddings to be stored.
    """

    def __init__(self, dim: int) -> None:
        self.dim = dim
        try:
            import faiss

            self._index = faiss.IndexFlatIP(dim)
            self._vectors = None
        except ModuleNotFoundError:
            self._index = None
            self._vectors = np.empty((0, dim), dtype="float32")
        self._doc_ids: List[str] = []

    @staticmethod
    def _normalize(vectors: np.ndarray) -> np.ndarray:
        """L2-normalise a 2-D array of vectors row-wise."""
        vectors = np.asarray(vectors, dtype="float32")
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0  # avoid division by zero
        return vectors / norms

    def add(self, doc_ids: Sequence[str], embeddings: np.ndarray) -> None:
        """
        Add embeddings and their document ids to the store.

        Parameters
        ----------
        doc_ids : sequence of str
            Identifiers aligned with the rows of ``embeddings``.
        embeddings : np.ndarray
            2-D array of shape (n_docs, dim).
        """
        if embeddings.ndim != 2 or embeddings.shape[1] != self.dim:
            raise ValueError(f"Expected embeddings of shape (n, {self.dim})")
        if len(doc_ids) != embeddings.shape[0]:
            raise ValueError("doc_ids and embeddings must have matching lengths")

        normalized = self._normalize(embeddings)
        if self._index is not None:
            self._index.add(normalized)
        else:
            self._vectors = np.vstack([self._vectors, normalized])
        self._doc_ids.extend(doc_ids)

    def search(self, query_embedding: np.ndarray, top_k: int = 10) -> List[VectorHit]:
        """
        Find the most similar stored vectors to a query embedding.

        Parameters
        ----------
        query_embedding : np.ndarray
            1-D array of length ``dim`` (or a single-row 2-D array).
        top_k : int
            Number of neighbours to return.

        Returns
        -------
        list of VectorHit
            Ranked hits, highest cosine similarity first.
        """
        if len(self) == 0:
            return []

        query = np.asarray(query_embedding, dtype="float32").reshape(1, -1)
        query = self._normalize(query)

        if self._index is not None:
            scores, indices = self._index.search(query, min(top_k, len(self)))
            ranked_pairs = list(zip(indices[0], scores[0]))
        else:
            scores = self._vectors @ query[0]
            indices = np.argsort(scores)[::-1][: min(top_k, len(self))]
            ranked_pairs = [(idx, scores[idx]) for idx in indices]

        hits: List[VectorHit] = []
        for rank, (idx, score) in enumerate(ranked_pairs, start=1):
            if idx == -1:  # FAISS uses -1 to pad when fewer than k results exist
                continue
            hits.append(VectorHit(doc_id=self._doc_ids[idx], score=float(score), rank=rank))
        return hits

    def __len__(self) -> int:
        """Number of vectors currently stored."""
        if self._index is not None:
            return self._index.ntotal
        return int(self._vectors.shape[0])

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, directory: str) -> None:
        """
        Persist the index and document ids to a directory.

        Parameters
        ----------
        directory : str
            Target directory (created if it does not exist).
        """
        os.makedirs(directory, exist_ok=True)
        if self._index is not None:
            import faiss

            faiss.write_index(self._index, os.path.join(directory, "index.faiss"))
            backend = "faiss"
        else:
            np.save(os.path.join(directory, "index.npy"), self._vectors)
            backend = "numpy"
        with open(os.path.join(directory, "doc_ids.json"), "w", encoding="utf-8") as handle:
            json.dump({"dim": self.dim, "doc_ids": self._doc_ids, "backend": backend}, handle)

    @classmethod
    def load(cls, directory: str) -> "VectorStore":
        """
        Load a previously saved vector store.

        Parameters
        ----------
        directory : str
            Directory created by :meth:`save`.

        Returns
        -------
        VectorStore
            The restored store.
        """
        with open(os.path.join(directory, "doc_ids.json"), "r", encoding="utf-8") as handle:
            meta = json.load(handle)

        store = cls(dim=meta["dim"])
        if meta.get("backend") == "numpy":
            store._index = None
            store._vectors = np.load(os.path.join(directory, "index.npy"))
        else:
            import faiss

            store._index = faiss.read_index(os.path.join(directory, "index.faiss"))
            store._vectors = None
        store._doc_ids = list(meta["doc_ids"])
        return store
