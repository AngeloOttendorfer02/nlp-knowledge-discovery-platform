"""
graph_queries.py — Query and analyse the knowledge graph.

Provides a thin query layer over a NetworkX knowledge graph: neighbour lookup,
shortest paths, centrality-based importance ranking, type-filtered subgraphs,
and connected-component inspection. These queries power both the analysis
notebooks and the graph-grounded retrieval used later in the project.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple


class GraphQueryEngine:
    """
    Convenience query interface over a knowledge graph.

    Parameters
    ----------
    graph : networkx.MultiDiGraph
        A graph produced by :class:`KnowledgeGraphBuilder`.
    """

    def __init__(self, graph) -> None:
        self.graph = graph

    # ------------------------------------------------------------------
    # Neighbourhood queries
    # ------------------------------------------------------------------

    def neighbors(self, node: str, relation: Optional[str] = None) -> List[str]:
        """
        Return the direct successors of a node, optionally filtered by relation.

        Parameters
        ----------
        node : str
            Node id to look up.
        relation : str, optional
            If given, keep only edges with this relation label.

        Returns
        -------
        list of str
            Neighbouring node ids.
        """
        if node not in self.graph:
            return []

        result: List[str] = []
        for target in self.graph.successors(node):
            for _, data in self.graph[node][target].items():
                if relation is None or data.get("relation") == relation:
                    result.append(target)
                    break
        return result

    def get_papers_for_concept(self, concept: str) -> List[str]:
        """
        Find all papers that mention a given concept.

        Parameters
        ----------
        concept : str
            Concept surface form (case-insensitive).

        Returns
        -------
        list of str
            Paper node ids linked to the concept via MENTIONS.
        """
        concept_node = f"concept::{concept.lower()}"
        if concept_node not in self.graph:
            return []

        papers = []
        for predecessor in self.graph.predecessors(concept_node):
            for _, data in self.graph[predecessor][concept_node].items():
                if data.get("relation") == "MENTIONS":
                    papers.append(predecessor)
                    break
        return papers

    # ------------------------------------------------------------------
    # Path queries
    # ------------------------------------------------------------------

    def shortest_path(self, source: str, target: str) -> Optional[List[str]]:
        """
        Return the shortest path between two nodes, or None if unreachable.

        Parameters
        ----------
        source, target : str
            Node ids.

        Returns
        -------
        list of str or None
            Sequence of node ids from source to target.
        """
        import networkx as nx

        if source not in self.graph or target not in self.graph:
            return None
        try:
            return nx.shortest_path(self.graph, source=source, target=target)
        except nx.NetworkXNoPath:
            return None

    # ------------------------------------------------------------------
    # Importance / centrality
    # ------------------------------------------------------------------

    def most_central_nodes(
        self, top_n: int = 10, node_type: Optional[str] = None
    ) -> List[Tuple[str, float]]:
        """
        Rank nodes by degree centrality.

        Parameters
        ----------
        top_n : int
            Number of nodes to return.
        node_type : str, optional
            If given, restrict ranking to nodes of this type.

        Returns
        -------
        list of (node_id, centrality) tuples
            Sorted by descending centrality.
        """
        import networkx as nx

        centrality = nx.degree_centrality(self.graph)

        if node_type is not None:
            centrality = {
                node: score
                for node, score in centrality.items()
                if self.graph.nodes[node].get("node_type") == node_type
            }

        ranked = sorted(centrality.items(), key=lambda kv: kv[1], reverse=True)
        return [(node, float(score)) for node, score in ranked[:top_n]]

    def pagerank(self, top_n: int = 10) -> List[Tuple[str, float]]:
        """
        Rank nodes by PageRank, a robust global importance measure.

        Parameters
        ----------
        top_n : int
            Number of nodes to return.

        Returns
        -------
        list of (node_id, score) tuples
        """
        import networkx as nx

        scores = nx.pagerank(self.graph, weight="weight")
        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        return [(node, float(score)) for node, score in ranked[:top_n]]

    # ------------------------------------------------------------------
    # Subgraph extraction
    # ------------------------------------------------------------------

    def subgraph_by_type(self, node_type: str):
        """
        Extract the induced subgraph containing only nodes of a given type.

        Parameters
        ----------
        node_type : str
            Node type to keep (e.g. "CONCEPT").

        Returns
        -------
        networkx.MultiDiGraph
            The induced subgraph (a copy).
        """
        nodes = [
            node
            for node, data in self.graph.nodes(data=True)
            if data.get("node_type") == node_type
        ]
        return self.graph.subgraph(nodes).copy()

    def ego_graph(self, node: str, radius: int = 1):
        """
        Extract the neighbourhood subgraph around a node up to a given radius.

        Parameters
        ----------
        node : str
            Centre node id.
        radius : int
            Number of hops to include.

        Returns
        -------
        networkx.MultiDiGraph
            The ego graph centred on ``node``.
        """
        import networkx as nx

        if node not in self.graph:
            return self.graph.subgraph([]).copy()
        return nx.ego_graph(self.graph, node, radius=radius)

    def connected_components_summary(self) -> Dict[str, int]:
        """
        Summarise the weakly-connected-component structure of the graph.

        Returns
        -------
        dict
            Number of components and the size of the largest component.
        """
        import networkx as nx

        components = list(nx.weakly_connected_components(self.graph))
        if not components:
            return {"num_components": 0, "largest_component_size": 0}

        return {
            "num_components": len(components),
            "largest_component_size": max(len(c) for c in components),
        }
