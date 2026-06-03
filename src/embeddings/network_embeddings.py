"""
network_embeddings.py — Build semantic networks from embeddings.

Given document (or concept) embeddings, this module constructs a semantic
similarity network: each document is a node, and an edge connects two documents
whose cosine similarity exceeds a threshold (or that are mutual k-nearest
neighbours). The resulting graph exposes clusters of related papers and can be
analysed with the same tools used for the knowledge graph.

It also offers simple community detection so semantic clusters can be compared
with the topics found by LDA / BERTopic.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


class SemanticNetworkBuilder:
    """
    Build and analyse a semantic-similarity network from embeddings.

    Parameters
    ----------
    similarity_threshold : float
        Minimum cosine similarity required to create an edge (used by the
        threshold-based constructor).
    """

    def __init__(self, similarity_threshold: float = 0.5) -> None:
        self.similarity_threshold = similarity_threshold

    # ------------------------------------------------------------------
    # Similarity helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _ensure_normalized(embeddings: np.ndarray) -> np.ndarray:
        """L2-normalise embeddings row-wise so dot product == cosine sim."""
        embeddings = np.asarray(embeddings, dtype="float32")
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return embeddings / norms

    def similarity_matrix(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Compute the pairwise cosine-similarity matrix.

        Parameters
        ----------
        embeddings : np.ndarray
            2-D array of document embeddings.

        Returns
        -------
        np.ndarray
            Symmetric (n, n) cosine-similarity matrix.
        """
        normalized = self._ensure_normalized(embeddings)
        return normalized @ normalized.T

    # ------------------------------------------------------------------
    # Network construction
    # ------------------------------------------------------------------

    def build_threshold_network(
        self,
        doc_ids: Sequence[str],
        embeddings: np.ndarray,
    ):
        """
        Connect document pairs whose similarity exceeds the threshold.

        Parameters
        ----------
        doc_ids : sequence of str
            Node identifiers aligned with the embedding rows.
        embeddings : np.ndarray
            2-D array of document embeddings.

        Returns
        -------
        networkx.Graph
            Undirected weighted similarity network.
        """
        import networkx as nx

        if len(doc_ids) != embeddings.shape[0]:
            raise ValueError("doc_ids and embeddings must have matching lengths")

        sim = self.similarity_matrix(embeddings)
        graph = nx.Graph()
        graph.add_nodes_from(doc_ids)

        n = len(doc_ids)
        # Iterate over the upper triangle only (matrix is symmetric)
        for i in range(n):
            for j in range(i + 1, n):
                weight = float(sim[i, j])
                if weight >= self.similarity_threshold:
                    graph.add_edge(doc_ids[i], doc_ids[j], weight=weight)

        return graph

    def build_knn_network(
        self,
        doc_ids: Sequence[str],
        embeddings: np.ndarray,
        k: int = 5,
    ):
        """
        Connect each document to its k most similar neighbours.

        A k-NN network avoids the "everything connects to everything" problem of
        a low fixed threshold and guarantees every node has neighbours.

        Parameters
        ----------
        doc_ids : sequence of str
            Node identifiers aligned with the embedding rows.
        embeddings : np.ndarray
            2-D array of document embeddings.
        k : int
            Number of neighbours per node.

        Returns
        -------
        networkx.Graph
            Undirected weighted k-NN similarity network.
        """
        import networkx as nx

        if len(doc_ids) != embeddings.shape[0]:
            raise ValueError("doc_ids and embeddings must have matching lengths")

        sim = self.similarity_matrix(embeddings)
        np.fill_diagonal(sim, -np.inf)  # never connect a node to itself

        graph = nx.Graph()
        graph.add_nodes_from(doc_ids)

        for i, doc_id in enumerate(doc_ids):
            neighbor_idx = np.argsort(sim[i])[::-1][:k]
            for j in neighbor_idx:
                weight = float(sim[i, j])
                if np.isfinite(weight):
                    graph.add_edge(doc_id, doc_ids[j], weight=weight)

        return graph

    # ------------------------------------------------------------------
    # Analysis
    # ------------------------------------------------------------------

    @staticmethod
    def detect_communities(graph) -> Dict[str, int]:
        """
        Detect communities using greedy modularity maximisation.

        Parameters
        ----------
        graph : networkx.Graph
            A semantic-similarity network.

        Returns
        -------
        dict
            Mapping of node id -> community index.
        """
        import networkx as nx
        from networkx.algorithms.community import greedy_modularity_communities

        if graph.number_of_edges() == 0:
            return {node: 0 for node in graph.nodes()}

        communities = greedy_modularity_communities(graph, weight="weight")
        mapping: Dict[str, int] = {}
        for community_id, members in enumerate(communities):
            for node in members:
                mapping[node] = community_id
        return mapping

    @staticmethod
    def network_stats(graph) -> Dict[str, float]:
        """
        Summarise the semantic network for the interpretation section.

        Parameters
        ----------
        graph : networkx.Graph
            A semantic-similarity network.

        Returns
        -------
        dict
            Node/edge counts, density, and average clustering coefficient.
        """
        import networkx as nx

        num_nodes = graph.number_of_nodes()
        return {
            "num_nodes": num_nodes,
            "num_edges": graph.number_of_edges(),
            "density": nx.density(graph) if num_nodes else 0.0,
            "avg_clustering": nx.average_clustering(graph, weight="weight") if num_nodes else 0.0,
        }
