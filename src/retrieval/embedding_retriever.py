"""
embedding_retriever.py — Dense semantic retrieval with Sentence Transformers.

This retriever encodes documents and queries into dense vectors using a
Sentence-Transformers model and ranks documents by cosine similarity. Unlike
the BM25 baseline, it matches meaning rather than exact words, so it can
retrieve relevant papers that share no surface vocabulary with the query.

The actual nearest-neighbour search is delegated to :class:`VectorStore`.
"""

from __future__ import annotations

from functools import lru_cache
from typing import List, Optional, Sequence

import numpy as np

from src.retrieval.bm25_retriever import RetrievalResult
from src.retrieval.vector_store import VectorStore


@lru_cache(maxsize=2)
def _load_model(model_name: str):
    """Load and cache a SentenceTransformer model."""
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name)


class EmbeddingRetriever:
    """
    Dense retriever backed by Sentence Transformers and a FAISS vector store.

    Parameters
    ----------
    model_name : str
        Name of the Sentence-Transformers model.
    batch_size : int
        Batch size used when encoding documents.
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        batch_size: int = 32,
    ) -> None:
        self.model_name = model_name
        self.batch_size = batch_size
        self._model = _load_model(model_name)
        self._store: Optional[VectorStore] = None
        self._doc_texts: dict = {}

    def index(self, doc_ids: Sequence[str], texts: Sequence[str]) -> "EmbeddingRetriever":
        """
        Encode and index a document collection.

        Parameters
        ----------
        doc_ids : sequence of str
            Identifiers aligned with ``texts``.
        texts : sequence of str
            Raw document texts.

        Returns
        -------
        EmbeddingRetriever
            The indexed retriever (for chaining).
        """
        if len(doc_ids) != len(texts):
            raise ValueError("doc_ids and texts must have the same length")

        embeddings = self._model.encode(
            list(texts),
            batch_size=self.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
        )

        self._store = VectorStore(dim=embeddings.shape[1])
        self._store.add(doc_ids, embeddings)
        self._doc_texts = dict(zip(doc_ids, texts))
        return self

    def search(self, query: str, top_k: int = 10) -> List[RetrievalResult]:
        """
        Retrieve the top-k semantically closest documents for a query.

        Parameters
        ----------
        query : str
            Raw query string.
        top_k : int
            Number of documents to return.

        Returns
        -------
        list of RetrievalResult
            Ranked results, highest cosine similarity first.
        """
        if self._store is None:
            raise RuntimeError("Call index() before search().")

        query_vec = self._model.encode([query], convert_to_numpy=True)[0]
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
        """
        Encode arbitrary texts into embeddings (exposed for reuse elsewhere).

        Parameters
        ----------
        texts : sequence of str
            Texts to encode.

        Returns
        -------
        np.ndarray
            2-D array of embeddings.
        """
        return self._model.encode(
            list(texts),
            batch_size=self.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
