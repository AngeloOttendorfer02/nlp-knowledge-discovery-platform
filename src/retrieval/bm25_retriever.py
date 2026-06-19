"""
bm25_retriever.py — Classical keyword-based retrieval baseline (BM25).

The retriever prefers the optional ``rank-bm25`` package when it is installed,
but it also contains a small pure-Python/NumPy implementation. This keeps the
project runnable in clean grading environments where optional dependencies may
be missing.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from math import log
from typing import Callable, List, Optional, Sequence

import numpy as np

from src.preprocessing.text_cleaning import preprocess


def _simple_tokenize(text: str) -> List[str]:
    """Dependency-light fallback tokenizer used when spaCy is unavailable."""
    try:
        return preprocess(text)
    except Exception:
        import re

        return [token.lower() for token in re.findall(r"[A-Za-z]{3,}", str(text))]


@dataclass
class RetrievalResult:
    """A single ranked retrieval hit."""

    doc_id: str
    score: float
    rank: int
    text: str = ""


class _FallbackBM25:
    """Minimal BM25Okapi-compatible scorer for small and medium corpora."""

    def __init__(self, corpus: Sequence[Sequence[str]], k1: float, b: float) -> None:
        self.k1 = float(k1)
        self.b = float(b)
        self.corpus = [list(doc) for doc in corpus]
        self.doc_len = np.array([len(doc) for doc in self.corpus], dtype=float)
        self.avgdl = float(self.doc_len.mean()) if len(self.doc_len) else 0.0
        self.term_freqs = [Counter(doc) for doc in self.corpus]

        df: Counter[str] = Counter()
        for doc in self.corpus:
            df.update(set(doc))

        n_docs = len(self.corpus)
        self.idf = {
            term: log(1.0 + (n_docs - freq + 0.5) / (freq + 0.5))
            for term, freq in df.items()
        }

    def get_scores(self, query_tokens: Sequence[str]) -> np.ndarray:
        scores = np.zeros(len(self.corpus), dtype=float)
        if not query_tokens or not self.corpus:
            return scores

        for i, freqs in enumerate(self.term_freqs):
            length = self.doc_len[i] or 1.0
            length_norm = 1.0 - self.b + self.b * (length / (self.avgdl or 1.0))
            for term in query_tokens:
                tf = freqs.get(term, 0)
                if tf == 0:
                    continue
                numerator = tf * (self.k1 + 1.0)
                denominator = tf + self.k1 * length_norm
                scores[i] += self.idf.get(term, 0.0) * numerator / denominator
        return scores


class BM25Retriever:
    """BM25 keyword retriever over a fixed document collection."""

    def __init__(
        self,
        k1: float = 1.5,
        b: float = 0.75,
        tokenizer: Optional[Callable[[str], List[str]]] = None,
    ) -> None:
        if k1 <= 0:
            raise ValueError("k1 must be positive")
        if not 0 <= b <= 1:
            raise ValueError("b must be between 0 and 1")
        self.k1 = k1
        self.b = b
        self._tokenizer = tokenizer or _simple_tokenize
        self._bm25 = None
        self._doc_ids: List[str] = []
        self._doc_texts: List[str] = []

    def index(self, doc_ids: Sequence[str], texts: Sequence[str]) -> "BM25Retriever":
        """Build the BM25 index from a document collection."""
        if len(doc_ids) != len(texts):
            raise ValueError("doc_ids and texts must have the same length")
        if not doc_ids:
            raise ValueError("Cannot index an empty document collection")

        self._doc_ids = [str(doc_id) for doc_id in doc_ids]
        self._doc_texts = [str(text) for text in texts]
        tokenized_corpus = [self._tokenizer(text) for text in self._doc_texts]

        try:
            from rank_bm25 import BM25Okapi  # type: ignore

            self._bm25 = BM25Okapi(tokenized_corpus, k1=self.k1, b=self.b)
        except Exception:
            self._bm25 = _FallbackBM25(tokenized_corpus, k1=self.k1, b=self.b)
        return self

    def search(self, query: str, top_k: int = 10) -> List[RetrievalResult]:
        """Retrieve the top-k documents for a query."""
        if self._bm25 is None:
            raise RuntimeError("Call index() before search().")
        if top_k <= 0:
            raise ValueError("top_k must be positive")

        scores = self._bm25.get_scores(self._tokenizer(query))
        ranked_idx = sorted(
            range(len(scores)),
            key=lambda i: (float(scores[i]), -i),
            reverse=True,
        )[:top_k]

        return [
            RetrievalResult(
                doc_id=self._doc_ids[idx],
                score=float(scores[idx]),
                rank=rank,
                text=self._doc_texts[idx][:300],
            )
            for rank, idx in enumerate(ranked_idx, start=1)
        ]
