"""
sentence_embeddings.py — Generate and compare sentence/document embeddings.

A focused wrapper around Sentence Transformers for producing dense vector
representations of documents, computing pairwise cosine-similarity matrices,
and answering "most similar document" queries. These embeddings are the shared
foundation for semantic retrieval, the semantic-similarity network, and the
clustering analysis.
"""

from __future__ import annotations

from functools import lru_cache
from typing import List, Optional, Sequence, Tuple

import numpy as np


@lru_cache(maxsize=2)
def _load_model(model_name: str):
    """Load and cache a SentenceTransformer model."""
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name)


class SentenceEmbedder:
    """
    Encode texts into dense embeddings and compute similarities.

    Parameters
    ----------
    model_name : str
        Sentence-Transformers model name.
    batch_size : int
        Batch size used during encoding.
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        batch_size: int = 32,
    ) -> None:
        self.model_name = model_name
        self.batch_size = batch_size
        self._model = _load_model(model_name)
        self._embeddings: Optional[np.ndarray] = None
        self._doc_ids: List[str] = []

    def encode(self, texts: Sequence[str], normalize: bool = True) -> np.ndarray:
        """
        Encode a list of texts into embeddings.

        Parameters
        ----------
        texts : sequence of str
            Texts to encode.
        normalize : bool
            L2-normalise embeddings so dot product equals cosine similarity.

        Returns
        -------
        np.ndarray
            2-D array of shape (n_texts, embedding_dim).
        """
        embeddings = self._model.encode(
            list(texts),
            batch_size=self.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=normalize,
        )
        return embeddings

    def fit(self, doc_ids: Sequence[str], texts: Sequence[str]) -> "SentenceEmbedder":
        """
        Encode and store a document collection for later similarity queries.

        Parameters
        ----------
        doc_ids : sequence of str
            Identifiers aligned with ``texts``.
        texts : sequence of str
            Document texts.

        Returns
        -------
        SentenceEmbedder
            The fitted embedder (for chaining).
        """
        if len(doc_ids) != len(texts):
            raise ValueError("doc_ids and texts must have the same length")

        self._doc_ids = list(doc_ids)
        self._embeddings = self.encode(texts, normalize=True)
        return self

    @property
    def embeddings(self) -> np.ndarray:
        """Return the stored document embeddings (requires :meth:`fit`)."""
        if self._embeddings is None:
            raise RuntimeError("Call fit() before accessing embeddings.")
        return self._embeddings

    @property
    def doc_ids(self) -> List[str]:
        """Return the stored document ids (requires :meth:`fit`)."""
        return self._doc_ids

    def similarity_matrix(self) -> np.ndarray:
        """
        Compute the full pairwise cosine-similarity matrix of stored documents.

        Returns
        -------
        np.ndarray
            Symmetric (n_docs, n_docs) similarity matrix.
        """
        emb = self.embeddings  # already L2-normalised
        return emb @ emb.T

    def most_similar(self, doc_index: int, top_k: int = 5) -> List[Tuple[str, float]]:
        """
        Return the most similar stored documents to a given document.

        Parameters
        ----------
        doc_index : int
            Index of the query document in the stored collection.
        top_k : int
            Number of neighbours to return (excluding the query itself).

        Returns
        -------
        list of (doc_id, similarity) tuples
            Sorted by descending similarity.
        """
        sims = self.embeddings @ self.embeddings[doc_index]
        # Exclude the document itself by setting its self-similarity to -inf
        sims[doc_index] = -np.inf
        top_idx = sims.argsort()[::-1][:top_k]
        return [(self._doc_ids[i], float(sims[i])) for i in top_idx]

    @staticmethod
    def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """
        Compute cosine similarity between two 1-D vectors.

        Parameters
        ----------
        a, b : np.ndarray
            Input vectors.

        Returns
        -------
        float
            Cosine similarity in [-1, 1].
        """
        denom = (np.linalg.norm(a) * np.linalg.norm(b)) or 1.0
        return float(np.dot(a, b) / denom)
