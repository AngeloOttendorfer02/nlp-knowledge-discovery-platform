"""
bm25_retriever.py — Classical keyword-based retrieval baseline (BM25).

BM25 is the project's retrieval baseline. It scores documents by exact term
overlap with the query, adjusted for term frequency saturation (k1) and
document length (b). It captures lexical matches well but, unlike embedding
retrieval, cannot match paraphrases or semantically related wording — which is
exactly the gap the rest of the platform investigates.

Built on the ``rank-bm25`` library.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from collections import Counter
from typing import Callable, List, Optional, Sequence

from src.preprocessing.text_cleaning import preprocess


@dataclass
class RetrievalResult:
    """
    A single ranked retrieval hit.

    Attributes
    ----------
    doc_id : str
        Identifier of the retrieved document.
    score : float
        Relevance score (higher is more relevant).
    rank : int
        1-based position in the ranked list.
    text : str
        The (possibly truncated) document text, for display.
    """

    doc_id: str
    score: float
    rank: int
    text: str = ""


class BM25Retriever:
    """
    BM25 keyword retriever over a fixed document collection.

    Parameters
    ----------
    k1 : float
        Term-frequency saturation parameter.
    b : float
        Length-normalisation parameter.
    tokenizer : callable, optional
        Function mapping a raw string to a list of tokens. Defaults to the
        project's :func:`preprocess` pipeline so the baseline matches the
        preprocessing used elsewhere.
    """

    def __init__(
        self,
        k1: float = 1.5,
        b: float = 0.75,
        tokenizer: Optional[Callable[[str], List[str]]] = None,
    ) -> None:
        self.k1 = k1
        self.b = b
        self._tokenizer = tokenizer or (lambda text: preprocess(text))
        self._bm25 = None
        self._doc_ids: List[str] = []
        self._doc_texts: List[str] = []

    def index(self, doc_ids: Sequence[str], texts: Sequence[str]) -> "BM25Retriever":
        """
        Build the BM25 index from a document collection.

        Parameters
        ----------
        doc_ids : sequence of str
            Identifiers aligned with ``texts``.
        texts : sequence of str
            Raw document texts.

        Returns
        -------
        BM25Retriever
            The indexed retriever (for chaining).
        """
        if len(doc_ids) != len(texts):
            raise ValueError("doc_ids and texts must have the same length")

        self._doc_ids = list(doc_ids)
        self._doc_texts = list(texts)

        tokenized_corpus = [self._tokenizer(text) for text in texts]
        try:
            from rank_bm25 import BM25Okapi

            self._bm25 = BM25Okapi(tokenized_corpus, k1=self.k1, b=self.b)
        except ModuleNotFoundError:
            self._bm25 = _SimpleBM25Okapi(tokenized_corpus, k1=self.k1, b=self.b)
        return self

    def search(self, query: str, top_k: int = 10) -> List[RetrievalResult]:
        """
        Retrieve the top-k documents for a query.

        Parameters
        ----------
        query : str
            Raw query string.
        top_k : int
            Number of documents to return.

        Returns
        -------
        list of RetrievalResult
            Ranked results, highest score first.
        """
        if self._bm25 is None:
            raise RuntimeError("Call index() before search().")

        tokenized_query = self._tokenizer(query)
        scores = self._bm25.get_scores(tokenized_query)

        # argsort descending, then take the requested number of hits
        ranked_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

        results: List[RetrievalResult] = []
        for rank, idx in enumerate(ranked_idx, start=1):
            results.append(
                RetrievalResult(
                    doc_id=self._doc_ids[idx],
                    score=float(scores[idx]),
                    rank=rank,
                    text=self._doc_texts[idx][:300],
                )
            )
        return results


class _SimpleBM25Okapi:
    """Small BM25 fallback used when the optional rank-bm25 package is absent."""

    def __init__(
        self,
        tokenized_corpus: Sequence[Sequence[str]],
        *,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        self.k1 = k1
        self.b = b
        self._doc_freqs = [Counter(document) for document in tokenized_corpus]
        self._doc_lengths = [len(document) for document in tokenized_corpus]
        self._avg_doc_length = (
            sum(self._doc_lengths) / len(self._doc_lengths)
            if self._doc_lengths
            else 0.0
        )
        self._idf = self._compute_idf()

    def get_scores(self, query_tokens: Sequence[str]) -> List[float]:
        scores: List[float] = []
        for frequencies, doc_length in zip(self._doc_freqs, self._doc_lengths):
            score = 0.0
            for token in query_tokens:
                frequency = frequencies.get(token, 0)
                if frequency == 0:
                    continue

                denominator = frequency + self.k1 * (
                    1.0
                    - self.b
                    + self.b * doc_length / (self._avg_doc_length or 1.0)
                )
                score += self._idf.get(token, 0.0) * (
                    frequency * (self.k1 + 1.0)
                ) / denominator
            scores.append(score)
        return scores

    def _compute_idf(self) -> dict[str, float]:
        document_count = len(self._doc_freqs)
        document_frequency: Counter[str] = Counter()
        for frequencies in self._doc_freqs:
            document_frequency.update(frequencies.keys())

        return {
            token: math.log(1.0 + (document_count - freq + 0.5) / (freq + 0.5))
            for token, freq in document_frequency.items()
        }
