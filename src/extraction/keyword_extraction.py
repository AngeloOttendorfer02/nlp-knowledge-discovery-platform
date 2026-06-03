"""
keyword_extraction.py — Keyword and key-phrase extraction.

Provides two complementary strategies:

1. **TF-IDF keywords** — corpus-level importance. A document's keywords are the
   terms with the highest TF-IDF weights, which captures words that are frequent
   in the document but rare across the corpus.

2. **TextRank-style keywords** — graph-based, single-document extraction built
   on token co-occurrence. This needs no corpus and works on one text at a time.

Both return ranked ``(keyword, score)`` lists so they can be compared directly.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np


class TfidfKeywordExtractor:
    """
    Corpus-level keyword extraction using TF-IDF.

    The extractor must be fit on the full corpus first; afterwards it can return
    the top keywords for any document in that corpus.

    Parameters
    ----------
    top_n : int
        Number of keywords to return per document.
    ngram_range : tuple of int
        Range of n-gram sizes considered (unigrams + bigrams by default).
    max_features : int
        Maximum vocabulary size kept by the vectorizer.
    """

    def __init__(
        self,
        top_n: int = 10,
        ngram_range: Tuple[int, int] = (1, 2),
        max_features: int = 10000,
    ) -> None:
        from sklearn.feature_extraction.text import TfidfVectorizer

        self.top_n = top_n
        self._vectorizer = TfidfVectorizer(
            ngram_range=ngram_range,
            max_features=max_features,
            stop_words="english",
        )
        self._matrix = None
        self._feature_names: Optional[np.ndarray] = None

    def fit(self, corpus: Sequence[str]) -> "TfidfKeywordExtractor":
        """
        Fit the TF-IDF vectorizer on a corpus of documents.

        Parameters
        ----------
        corpus : sequence of str
            Pre-cleaned document strings.

        Returns
        -------
        TfidfKeywordExtractor
            The fitted extractor (for chaining).
        """
        self._matrix = self._vectorizer.fit_transform(corpus)
        self._feature_names = np.array(self._vectorizer.get_feature_names_out())
        return self

    def get_keywords(self, doc_index: int) -> List[Tuple[str, float]]:
        """
        Return the top keywords for a document by its corpus index.

        Parameters
        ----------
        doc_index : int
            Row index of the document in the fitted matrix.

        Returns
        -------
        list of (keyword, score) tuples
            Sorted by descending TF-IDF weight.
        """
        if self._matrix is None or self._feature_names is None:
            raise RuntimeError("Call fit() before get_keywords().")

        row = self._matrix[doc_index].toarray().ravel()
        if not row.any():
            return []

        top_idx = row.argsort()[::-1][: self.top_n]
        return [(self._feature_names[i], float(row[i])) for i in top_idx if row[i] > 0]

    def get_keywords_for_text(self, text: str) -> List[Tuple[str, float]]:
        """
        Return keywords for an unseen document using the fitted vocabulary/IDF.

        Parameters
        ----------
        text : str
            A new (pre-cleaned) document string.

        Returns
        -------
        list of (keyword, score) tuples
        """
        if self._feature_names is None:
            raise RuntimeError("Call fit() before get_keywords_for_text().")

        row = self._vectorizer.transform([text]).toarray().ravel()
        if not row.any():
            return []

        top_idx = row.argsort()[::-1][: self.top_n]
        return [(self._feature_names[i], float(row[i])) for i in top_idx if row[i] > 0]


class TextRankKeywordExtractor:
    """
    Single-document keyword extraction using a TextRank-style algorithm.

    Tokens are connected in a co-occurrence graph within a sliding window, then
    ranked with the PageRank algorithm. Only nouns, proper nouns, and adjectives
    are considered as candidate keywords.

    Parameters
    ----------
    spacy_model : str
        spaCy model used for tokenization and POS tagging.
    window_size : int
        Co-occurrence window size.
    top_n : int
        Number of keywords to return.
    """

    _CANDIDATE_POS = {"NOUN", "PROPN", "ADJ"}

    def __init__(
        self,
        spacy_model: str = "en_core_web_sm",
        window_size: int = 4,
        top_n: int = 10,
    ) -> None:
        from src.preprocessing.text_cleaning import _load_spacy

        self.window_size = window_size
        self.top_n = top_n
        self._nlp = _load_spacy(spacy_model)

    def get_keywords(self, text: str) -> List[Tuple[str, float]]:
        """
        Extract ranked keywords from a single document.

        Parameters
        ----------
        text : str
            Raw input text.

        Returns
        -------
        list of (keyword, score) tuples
            Sorted by descending PageRank score.
        """
        import networkx as nx

        if not text or not text.strip():
            return []

        doc = self._nlp(text)

        # Keep candidate tokens (content words), lemmatized and lowercased
        candidates = [
            token.lemma_.lower()
            for token in doc
            if token.pos_ in self._CANDIDATE_POS and not token.is_stop and token.is_alpha
        ]
        if len(candidates) < 2:
            return [(candidates[0], 1.0)] if candidates else []

        # Build a co-occurrence graph over a sliding window
        graph = nx.Graph()
        graph.add_nodes_from(set(candidates))
        for i, word in enumerate(candidates):
            for j in range(i + 1, min(i + self.window_size, len(candidates))):
                neighbor = candidates[j]
                if word != neighbor:
                    weight = graph.get_edge_data(word, neighbor, {}).get("weight", 0) + 1
                    graph.add_edge(word, neighbor, weight=weight)

        scores = nx.pagerank(graph, weight="weight")
        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        return [(word, float(score)) for word, score in ranked[: self.top_n]]
