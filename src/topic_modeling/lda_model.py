"""
lda_model.py — Latent Dirichlet Allocation topic modeling.

Wraps gensim's LDA implementation behind a small, typed interface. LDA is the
classical, probabilistic topic-modeling baseline: it represents each document
as a mixture of topics and each topic as a distribution over words. It serves
as the reference point against which the transformer-based BERTopic model is
compared.

The model also computes topic coherence (c_v), the standard intrinsic metric
for choosing the number of topics.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple


class LDATopicModel:
    """
    Latent Dirichlet Allocation topic model (gensim backend).

    Parameters
    ----------
    num_topics : int
        Number of latent topics to discover.
    passes : int
        Number of passes over the corpus during training.
    iterations : int
        Maximum number of iterations per document.
    seed : int
        Random seed for reproducibility.
    """

    def __init__(
        self,
        num_topics: int = 10,
        passes: int = 10,
        iterations: int = 50,
        seed: int = 42,
    ) -> None:
        self.num_topics = num_topics
        self.passes = passes
        self.iterations = iterations
        self.seed = seed

        self._dictionary = None
        self._corpus = None
        self._model = None

    def fit(self, tokenized_docs: Sequence[Sequence[str]]) -> "LDATopicModel":
        """
        Train the LDA model on a tokenized corpus.

        Parameters
        ----------
        tokenized_docs : sequence of sequence of str
            Documents already tokenized (e.g. via the preprocessing module).

        Returns
        -------
        LDATopicModel
            The fitted model (for chaining).
        """
        from gensim import corpora
        from gensim.models import LdaModel

        # Build a dictionary and bag-of-words corpus
        self._dictionary = corpora.Dictionary(tokenized_docs)
        # Drop extremely rare and extremely common tokens to sharpen topics
        self._dictionary.filter_extremes(no_below=5, no_above=0.5)
        self._corpus = [self._dictionary.doc2bow(doc) for doc in tokenized_docs]

        self._model = LdaModel(
            corpus=self._corpus,
            id2word=self._dictionary,
            num_topics=self.num_topics,
            passes=self.passes,
            iterations=self.iterations,
            random_state=self.seed,
        )
        return self

    def get_topics(self, top_n_words: int = 10) -> List[List[Tuple[str, float]]]:
        """
        Return the top words and weights for each topic.

        Parameters
        ----------
        top_n_words : int
            Number of words per topic.

        Returns
        -------
        list of list of (word, weight) tuples
            One inner list per topic.
        """
        self._require_fitted()

        topics: List[List[Tuple[str, float]]] = []
        for topic_id in range(self.num_topics):
            words = self._model.show_topic(topic_id, topn=top_n_words)
            topics.append([(word, float(weight)) for word, weight in words])
        return topics

    def get_document_topics(self, doc_index: int) -> List[Tuple[int, float]]:
        """
        Return the topic distribution for a document in the training corpus.

        Parameters
        ----------
        doc_index : int
            Index of the document in the fitted corpus.

        Returns
        -------
        list of (topic_id, probability) tuples
            Sorted by descending probability.
        """
        self._require_fitted()
        dist = self._model.get_document_topics(self._corpus[doc_index])
        return sorted(((int(t), float(p)) for t, p in dist), key=lambda kv: kv[1], reverse=True)

    def coherence(self, tokenized_docs: Sequence[Sequence[str]]) -> float:
        """
        Compute the c_v topic-coherence score (higher is better).

        Parameters
        ----------
        tokenized_docs : sequence of sequence of str
            The same tokenized corpus used for training.

        Returns
        -------
        float
            The c_v coherence score.
        """
        from gensim.models import CoherenceModel

        self._require_fitted()
        coherence_model = CoherenceModel(
            model=self._model,
            texts=list(tokenized_docs),
            dictionary=self._dictionary,
            coherence="c_v",
        )
        return float(coherence_model.get_coherence())

    def topic_summary(self, top_n_words: int = 8) -> Dict[int, str]:
        """
        Return a human-readable one-line summary per topic.

        Parameters
        ----------
        top_n_words : int
            Number of words to include in each summary.

        Returns
        -------
        dict
            Mapping of topic id -> comma-separated top words.
        """
        summary: Dict[int, str] = {}
        for topic_id, words in enumerate(self.get_topics(top_n_words)):
            summary[topic_id] = ", ".join(word for word, _ in words)
        return summary

    def _require_fitted(self) -> None:
        """Raise if the model has not been trained yet."""
        if self._model is None:
            raise RuntimeError("Call fit() before using the model.")
