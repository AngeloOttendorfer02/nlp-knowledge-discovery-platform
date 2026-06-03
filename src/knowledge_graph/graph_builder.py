"""
graph_builder.py — Construct a knowledge graph from extracted information.

Turns documents, entities, and relations into a typed, directed
:class:`networkx.MultiDiGraph`. The schema follows the project proposal:

  Node types : PAPER, AUTHOR, CONCEPT (and other entity labels)
  Edge types : AUTHORED_BY, MENTIONS, RELATED_TO, BELONGS_TO_TOPIC, ...

Each node stores a ``node_type`` attribute and each edge a ``relation``
attribute and an integer ``weight``. The resulting graph can be queried,
visualised, and exported (GraphML / node-link JSON) for downstream use.
"""

from __future__ import annotations

import json
import os
from typing import Dict, List, Optional, Sequence

from src.extraction.entity_extraction import Entity
from src.extraction.relation_extraction import Relation


class KnowledgeGraphBuilder:
    """
    Incrementally build a scientific knowledge graph.

    Parameters
    ----------
    min_edge_weight : int
        Edges with weight below this threshold are dropped when pruning.
    """

    # Canonical edge labels used by the builder
    EDGE_AUTHORED_BY = "AUTHORED_BY"
    EDGE_MENTIONS = "MENTIONS"
    EDGE_BELONGS_TO_TOPIC = "BELONGS_TO_TOPIC"

    def __init__(self, min_edge_weight: int = 1) -> None:
        import networkx as nx

        self.min_edge_weight = min_edge_weight
        self.graph = nx.MultiDiGraph()

    # ------------------------------------------------------------------
    # Node / edge helpers
    # ------------------------------------------------------------------

    def _add_node(self, node_id: str, node_type: str, **attrs) -> None:
        """Add a node, preserving the first-seen type and merging attributes."""
        if node_id in self.graph:
            self.graph.nodes[node_id].update(attrs)
        else:
            # Default the display label to the node id unless one was provided
            attrs.setdefault("label", node_id)
            self.graph.add_node(node_id, node_type=node_type, **attrs)

    def _add_edge(self, source: str, target: str, relation: str, weight: int = 1) -> None:
        """Add or strengthen a typed edge between two existing nodes."""
        # If an identical (source, target, relation) edge exists, accumulate weight
        if self.graph.has_edge(source, target):
            for key, data in self.graph[source][target].items():
                if data.get("relation") == relation:
                    data["weight"] = data.get("weight", 1) + weight
                    return
        self.graph.add_edge(source, target, relation=relation, weight=weight)

    # ------------------------------------------------------------------
    # Public construction API
    # ------------------------------------------------------------------

    def add_paper(
        self,
        doc_id: str,
        title: str = "",
        authors: Optional[Sequence[str]] = None,
        categories: Optional[Sequence[str]] = None,
    ) -> None:
        """
        Add a paper node together with its authors and categories.

        Parameters
        ----------
        doc_id : str
            Unique paper identifier (used as the node id).
        title : str
            Paper title (stored as a node attribute and display label).
        authors : sequence of str, optional
            Author names; each becomes an AUTHOR node linked via AUTHORED_BY.
        categories : sequence of str, optional
            arXiv categories; each becomes a TOPIC node linked via
            BELONGS_TO_TOPIC.
        """
        paper_node = f"paper::{doc_id}"
        self._add_node(paper_node, node_type="PAPER", title=title, label=title or doc_id)

        for author in authors or []:
            author_node = f"author::{author}"
            self._add_node(author_node, node_type="AUTHOR", label=author)
            self._add_edge(paper_node, author_node, self.EDGE_AUTHORED_BY)

        for category in categories or []:
            topic_node = f"topic::{category}"
            self._add_node(topic_node, node_type="TOPIC", label=category)
            self._add_edge(paper_node, topic_node, self.EDGE_BELONGS_TO_TOPIC)

    def add_entities(self, doc_id: str, entities: Sequence[Entity]) -> None:
        """
        Link a paper to the entities mentioned in it.

        Parameters
        ----------
        doc_id : str
            Identifier of the paper the entities were extracted from.
        entities : sequence of Entity
            Entities to attach via MENTIONS edges.
        """
        paper_node = f"paper::{doc_id}"
        if paper_node not in self.graph:
            self._add_node(paper_node, node_type="PAPER", label=doc_id)

        for entity in entities:
            concept_node = f"concept::{entity.text.lower()}"
            self._add_node(
                concept_node,
                node_type="CONCEPT",
                label=entity.text,
                entity_label=entity.label,
            )
            self._add_edge(paper_node, concept_node, self.EDGE_MENTIONS)

    def add_relations(self, relations: Sequence[Relation]) -> None:
        """
        Add concept-to-concept relations produced by relation extraction.

        Parameters
        ----------
        relations : sequence of Relation
            Typed relations between entity surface forms.
        """
        for rel in relations:
            source_node = f"concept::{rel.source.lower()}"
            target_node = f"concept::{rel.target.lower()}"
            self._add_node(source_node, node_type="CONCEPT", label=rel.source)
            self._add_node(target_node, node_type="CONCEPT", label=rel.target)
            self._add_edge(source_node, target_node, rel.relation.upper(), weight=rel.weight)

    # ------------------------------------------------------------------
    # Maintenance / export
    # ------------------------------------------------------------------

    def prune(self) -> None:
        """Remove edges below ``min_edge_weight`` and any resulting isolates."""
        import networkx as nx

        to_remove = [
            (u, v, k)
            for u, v, k, data in self.graph.edges(keys=True, data=True)
            if data.get("weight", 1) < self.min_edge_weight
        ]
        self.graph.remove_edges_from(to_remove)
        self.graph.remove_nodes_from(list(nx.isolates(self.graph)))

    def stats(self) -> Dict[str, int]:
        """
        Return basic size statistics of the current graph.

        Returns
        -------
        dict
            Number of nodes, edges, and nodes per type.
        """
        type_counts: Dict[str, int] = {}
        for _, data in self.graph.nodes(data=True):
            node_type = data.get("node_type", "UNKNOWN")
            type_counts[node_type] = type_counts.get(node_type, 0) + 1

        return {
            "num_nodes": self.graph.number_of_nodes(),
            "num_edges": self.graph.number_of_edges(),
            **{f"nodes_{t}": c for t, c in type_counts.items()},
        }

    def save_graphml(self, path: str) -> None:
        """Export the graph to GraphML (readable by Gephi, Cytoscape, etc.)."""
        import networkx as nx

        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        nx.write_graphml(self.graph, path)

    def save_json(self, path: str) -> None:
        """Export the graph as node-link JSON."""
        import networkx as nx

        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        data = nx.node_link_data(self.graph)
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(data, handle, indent=2)
