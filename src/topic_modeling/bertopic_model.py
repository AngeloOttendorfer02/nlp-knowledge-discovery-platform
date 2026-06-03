"""
bertopic_model.py — Transformer-based topic modeling with BERTopic.

BERTopic discovers topics by embedding documents with a Sentence-Transformers
model, reducing dimensionality (UMAP), clustering (HDBSCAN), and labelling each
cluster with a class-based TF-IDF (c-TF-IDF) representation. Compared with LDA,
it captures semantic similarity and usually produces more coherent, readable
topics on short scientific abstracts.

This wrapper exposes a minimal fit / inspect / visualise interface and keeps the
underlying BERTopic object accessible for advanced use.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple


class BERTopicModel:
    """
    Thin wrapper around the BERTopic library.

    Parameters
    ----------
    embedding_model : str
        Sentence-Transformers model used to embed documents.
    min_topic_size : int
        Minimum number of documents required to form a topic. Larger values
        yield fewer, broader topics.
    nr_topics : int or str, optional
        If set (e.g. an int or "auto"), BERTopic reduces topics to this number
        after the initial clustering.
    seed : int
        Random seed forwarded to UMAP for reproducible clustering.
    """

    def __init__(
        self,
        embedding_model: str = "all-MiniLM-L6-v2",
        min_topic_size: int = 15,
        nr_topics: Optional[object] = None,
        seed: int = 42,
    ) -> None:
        self.embedding_model = embedding_model
        self.min_topic_size = min_topic_size
        self.nr_topics = nr_topics
        self.seed = seed

        self._model = None
        self._topics: Optional[List[int]] = None

    def fit(self, documents: Sequence[str]) -> "BERTopicModel":
        """
        Fit BERTopic on a collection of raw documents.

        Note: BERTopic expects readable text (not pre-tokenized lists) because
        it embeds the documents with a transformer and builds c-TF-IDF labels.

        Parameters
        ----------
        documents : sequence of str
            Raw document strings (e.g. abstracts).

        Returns
        -------
        BERTopicModel
            The fitted model (for chaining).
        """
        from bertopic import BERTopic
        from umap import UMAP

        # Fix the UMAP seed so topic assignments are reproducible across runs
        umap_model = UMAP(
            n_neighbors=15,
            n_components=5,
            min_dist=0.0,
            metric="cosine",
            random_state=self.seed,
        )

        self._model = BERTopic(
            embedding_model=self.embedding_model,
            umap_model=umap_model,
            min_topic_size=self.min_topic_size,
            nr_topics=self.nr_topics,
            calculate_probabilities=False,
            verbose=False,
        )

        self._topics, _ = self._model.fit_transform(list(documents))
        return self

    def get_topic_info(self):
        """
        Return BERTopic's topic overview table.

        Returns
        -------
        pandas.DataFrame
            One row per topic with its id, size, and auto-generated name.
            Topic ``-1`` denotes outliers / unclustered documents.
        """
        self._require_fitted()
        return self._model.get_topic_info()

    def get_topic_words(self, topic_id: int, top_n: int = 10) -> List[Tuple[str, float]]:
        """
        Return the top words for a specific topic.

        Parameters
        ----------
        topic_id : int
            Topic identifier (use ids from :meth:`get_topic_info`).
        top_n : int
            Number of words to return.

        Returns
        -------
        list of (word, score) tuples
        """
        self._require_fitted()
        words = self._model.get_topic(topic_id)
        if not words:
            return []
        return [(word, float(score)) for word, score in words[:top_n]]

    def get_document_topics(self) -> List[int]:
        """
        Return the topic assignment for each training document.

        Returns
        -------
        list of int
            Topic id per document (in input order); ``-1`` marks outliers.
        """
        self._require_fitted()
        return list(self._topics) if self._topics is not None else []

    def visualize_topics(self, output_path: str) -> str:
        """
        Save BERTopic's interactive inter-topic distance map as HTML.

        Parameters
        ----------
        output_path : str
            Destination ``.html`` file.

        Returns
        -------
        str
            The path to the written file.
        """
        self._require_fitted()
        fig = self._model.visualize_topics()
        fig.write_html(output_path)
        return output_path

    def visualize_barchart(self, output_path: str, top_n_topics: int = 8) -> str:
        """
        Save a bar-chart of the top words for the largest topics as HTML.

        Parameters
        ----------
        output_path : str
            Destination ``.html`` file.
        top_n_topics : int
            Number of topics to display.

        Returns
        -------
        str
            The path to the written file.
        """
        self._require_fitted()
        fig = self._model.visualize_barchart(top_n_topics=top_n_topics)
        fig.write_html(output_path)
        return output_path

    def _require_fitted(self) -> None:
        """Raise if the model has not been trained yet."""
        if self._model is None:
            raise RuntimeError("Call fit() before using the model.")
