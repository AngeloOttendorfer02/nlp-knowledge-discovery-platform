"""
kg_enhanced_retriever.py — Knowledge graph-enhanced retrieval.

This module reranks candidates from an existing retriever using evidence from a
knowledge graph.

Scoring formula:

    final_score =
        retrieval_weight * normalized_retrieval_score
      + graph_weight * normalized_graph_score

The graph score is based on whether query terms match graph concepts/topics and
whether candidate papers are connected to those matched graph nodes.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Sequence, Set

from src.retrieval.bm25_retriever import RetrievalResult


@dataclass(frozen=True)
class KGEnhancedResult:
    doc_id: str
    score: float
    rank: int
    retrieval_score: float
    graph_score: float
    text: str = ""


class KnowledgeGraphEnhancedRetriever:
    """
    Rerank base retrieval candidates using knowledge graph evidence.

    Parameters
    ----------
    base_retriever
        Any retriever exposing search(query, top_k).
    graph
        A NetworkX graph produced by KnowledgeGraphBuilder.
    retrieval_weight : float
        Weight assigned to normalized base retrieval scores.
    graph_weight : float
        Weight assigned to normalized graph evidence scores.
    """

    def __init__(
        self,
        base_retriever,
        graph,
        retrieval_weight: float = 0.7,
        graph_weight: float = 0.3,
    ) -> None:
        if retrieval_weight < 0 or graph_weight < 0:
            raise ValueError("Weights must be non-negative")

        if retrieval_weight == 0 and graph_weight == 0:
            raise ValueError("At least one weight must be positive")

        self.base_retriever = base_retriever
        self.graph = graph
        self.retrieval_weight = retrieval_weight
        self.graph_weight = graph_weight

    # ------------------------------------------------------------------
    # Query processing
    # ------------------------------------------------------------------

    @staticmethod
    def extract_query_terms(query: str) -> Set[str]:
        """
        Extract simple lowercase query terms.

        This lightweight approach is deterministic and avoids loading another
        NLP model inside the reranker.
        """
        if not query:
            return set()

        return {
            token.lower()
            for token in re.findall(r"[A-Za-z][A-Za-z0-9_+-]*", query)
            if len(token) >= 3
        }

    def matched_graph_nodes(self, query: str) -> Set[str]:
        """
        Match query terms against graph node labels and ids.
        """
        terms = self.extract_query_terms(query)
        matched: Set[str] = set()

        if not terms:
            return matched

        for node, data in self.graph.nodes(data=True):
            label = str(data.get("label", node)).lower()
            node_id = str(node).lower()

            if any(term in label or term in node_id for term in terms):
                matched.add(node)

        return matched

    # ------------------------------------------------------------------
    # Graph scoring
    # ------------------------------------------------------------------

    def graph_score_for_doc(self, doc_id: str, matched_nodes: Set[str]) -> float:
        """
        Compute graph evidence score for a candidate paper.

        A paper receives graph evidence if it is the matched node itself or if
        it is directly connected to matched concept/topic/author nodes.
        """
        if not matched_nodes:
            return 0.0

        paper_node = f"paper::{doc_id}"

        if paper_node not in self.graph:
            return 0.0

        score = 0.0

        for matched_node in matched_nodes:
            if matched_node == paper_node:
                score += 1.0
                continue

            if self.graph.has_edge(paper_node, matched_node):
                edge_data = self.graph.get_edge_data(paper_node, matched_node)
                score += self._edge_weight_sum(edge_data)

            if self.graph.has_edge(matched_node, paper_node):
                edge_data = self.graph.get_edge_data(matched_node, paper_node)
                score += self._edge_weight_sum(edge_data)

        return score

    @staticmethod
    def _edge_weight_sum(edge_data) -> float:
        """
        Sum edge weights for both in-memory MultiDiGraph edge data and
        GraphML-loaded edge data.

        In-memory MultiDiGraph format:
            {0: {"relation": "MENTIONS", "weight": 3}}

        GraphML-loaded format:
            {"id": "0", "relation": "MENTIONS", "weight": 3}
        """
        if not edge_data:
            return 0.0

        if isinstance(edge_data, dict) and "weight" in edge_data:
            return float(edge_data.get("weight", 1.0))

        total = 0.0

        for _, data in edge_data.items():
            if isinstance(data, dict):
                total += float(data.get("weight", 1.0))

        return total

    @staticmethod
    def normalize_scores(scores: Dict[str, float]) -> Dict[str, float]:
        """
        Normalize scores to [0, 1] by dividing by the maximum score.
        """
        if not scores:
            return {}

        max_score = max(scores.values())

        if max_score <= 0:
            return {key: 0.0 for key in scores}

        return {key: value / max_score for key, value in scores.items()}

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(self, query: str, top_k: int = 10, candidate_k: int | None = None) -> List[KGEnhancedResult]:
        """
        Retrieve candidates with the base retriever and rerank them using graph evidence.
        """
        if top_k <= 0:
            raise ValueError("top_k must be positive")

        candidate_k = candidate_k or max(top_k * 3, top_k)

        base_hits = self.base_retriever.search(query, top_k=candidate_k)
        matched_nodes = self.matched_graph_nodes(query)

        retrieval_scores = {
            str(hit.doc_id): float(hit.score)
            for hit in base_hits
        }

        graph_scores = {
            str(hit.doc_id): self.graph_score_for_doc(str(hit.doc_id), matched_nodes)
            for hit in base_hits
        }

        normalized_retrieval = self.normalize_scores(retrieval_scores)
        normalized_graph = self.normalize_scores(graph_scores)

        text_by_doc = {
            str(hit.doc_id): hit.text
            for hit in base_hits
        }

        reranked: List[KGEnhancedResult] = []

        for doc_id in retrieval_scores:
            final_score = (
                self.retrieval_weight * normalized_retrieval.get(doc_id, 0.0)
                + self.graph_weight * normalized_graph.get(doc_id, 0.0)
            )

            reranked.append(
                KGEnhancedResult(
                    doc_id=doc_id,
                    score=final_score,
                    rank=0,
                    retrieval_score=retrieval_scores[doc_id],
                    graph_score=graph_scores[doc_id],
                    text=text_by_doc.get(doc_id, ""),
                )
            )

        reranked.sort(
            key=lambda result: (
                result.score,
                result.graph_score,
                result.retrieval_score,
            ),
            reverse=True,
        )

        return [
            KGEnhancedResult(
                doc_id=result.doc_id,
                score=result.score,
                rank=rank,
                retrieval_score=result.retrieval_score,
                graph_score=result.graph_score,
                text=result.text,
            )
            for rank, result in enumerate(reranked[:top_k], start=1)
        ]