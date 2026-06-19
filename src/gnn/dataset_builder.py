"""Build graph-learning inputs from the project knowledge graph."""

from __future__ import annotations
from dataclasses import dataclass
from typing import Mapping
import numpy as np


@dataclass
class GraphDataset:
    """Framework-neutral graph dataset container."""

    node_ids: list[str]
    features: np.ndarray
    edge_index: np.ndarray
    labels: np.ndarray | None = None


def build_graph_dataset(graph, node_features: Mapping[str, np.ndarray] | None = None, labels: Mapping[str, int] | None = None) -> GraphDataset:
    """Convert a NetworkX graph into node features and COO edge indices."""
    node_ids = list(graph.nodes())
    node_to_idx = {node: i for i, node in enumerate(node_ids)}

    if node_features:
        dim = len(next(iter(node_features.values())))
        features = np.zeros((len(node_ids), dim), dtype="float32")
        for node, vector in node_features.items():
            if node in node_to_idx:
                features[node_to_idx[node]] = np.asarray(vector, dtype="float32")
    else:
        # Lightweight default: one feature each for in-degree, out-degree, and total degree.
        features = np.array(
            [[graph.in_degree(n) if hasattr(graph, "in_degree") else graph.degree(n), graph.out_degree(n) if hasattr(graph, "out_degree") else graph.degree(n), graph.degree(n)] for n in node_ids],
            dtype="float32",
        )

    edges = [(node_to_idx[u], node_to_idx[v]) for u, v in graph.edges() if u in node_to_idx and v in node_to_idx]
    edge_index = np.array(edges, dtype="int64").T if edges else np.empty((2, 0), dtype="int64")
    y = None
    if labels:
        y = np.array([labels.get(node, -1) for node in node_ids], dtype="int64")
    return GraphDataset(node_ids=node_ids, features=features, edge_index=edge_index, labels=y)
