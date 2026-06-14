"""
Knowledge-graph-enhanced retrieval.

The module reranks candidates returned by an existing BM25 or semantic
retriever. Query terms are matched to concept, topic, and author nodes in the
knowledge graph. Papers directly connected to matched nodes, or to one-hop
related concept nodes, receive graph evidence. The final score is a documented
weighted combination of normalized retrieval and graph scores.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, MutableMapping, Sequence, Tuple

from src.retrieval.bm25_retriever import RetrievalResult

_TOKEN_RE = re.compile(r"[a-zA-Z][a-zA-Z0-9.+#-]*")
_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "in",
    "into",
    "is",
    "of",
    "on",
    "or",
    "the",
    "to",
    "using",
    "with",
}

_TOPIC_ALIASES = {
    "cs ai": ("artificial intelligence", "ai"),
    "cs cl": (
        "computational linguistics",
        "natural language processing",
        "nlp",
        "language model",
    ),
    "cs ir": ("information retrieval", "search", "retrieval"),
    "cs lg": ("machine learning", "ml", "deep learning"),
}

_NODE_TYPE_WEIGHTS = {
    "CONCEPT": 1.0,
    "TOPIC": 0.8,
    "AUTHOR": 0.25,
}


@dataclass(frozen=True)
class KGEnhancedResult:
    """A reranked retrieval hit with transparent component scores."""

    doc_id: str
    score: float
    rank: int
    text: str = ""
    base_score: float = 0.0
    normalized_base_score: float = 0.0
    graph_score: float = 0.0
    normalized_graph_score: float = 0.0
    evidence: Tuple[str, ...] = ()


def _normalize_label(value: object) -> str:
    text = str(value or "").lower().replace("_", " ")
    return " ".join(_TOKEN_RE.findall(text))


def _tokens(value: object) -> set[str]:
    return {
        token
        for token in _TOKEN_RE.findall(str(value or "").lower())
        if token not in _STOPWORDS
    }


def _minmax(values: Sequence[float]) -> List[float]:
    if not values:
        return []

    minimum = min(values)
    maximum = max(values)
    if math.isclose(minimum, maximum):
        fill = 1.0 if maximum > 0.0 else 0.0
        return [fill for _ in values]

    scale = maximum - minimum
    return [(value - minimum) / scale for value in values]


def _edge_weight(data: Mapping[str, object]) -> float:
    try:
        value = float(data.get("weight", 1.0))
    except (TypeError, ValueError):
        value = 1.0
    return max(value, 0.0)


def augment_graph_with_keywords(
    graph,
    keyword_rows: Iterable[Mapping[str, object]],
    *,
    top_n_per_document: int = 10,
    copy_graph: bool = True,
):
    """
    Add keyword concept nodes to an in-memory graph.

    The existing pipeline writes ``keywords.csv`` but does not currently add
    those keywords to the exported graph. This helper bridges that gap for the
    retrieval experiment without modifying the persisted graph or the pipeline.
    """
    if top_n_per_document <= 0:
        raise ValueError("top_n_per_document must be positive")

    result = graph.copy() if copy_graph else graph
    grouped: MutableMapping[str, List[Mapping[str, object]]] = {}

    for row in keyword_rows:
        doc_id = str(row.get("doc_id", "")).strip()
        keyword = str(row.get("keyword", "")).strip()
        if not doc_id or not keyword:
            continue
        grouped.setdefault(doc_id, []).append(row)

    for doc_id, rows in grouped.items():
        rows = sorted(
            rows,
            key=lambda row: float(row.get("score", 0.0) or 0.0),
            reverse=True,
        )[:top_n_per_document]

        paper_node = f"paper::{doc_id}"
        if paper_node not in result:
            continue

        for row in rows:
            keyword = str(row["keyword"]).strip()
            normalized = _normalize_label(keyword)
            if not normalized:
                continue

            concept_node = f"concept::{normalized}"
            if concept_node not in result:
                result.add_node(
                    concept_node,
                    node_type="CONCEPT",
                    label=keyword,
                    source="keyword_extraction",
                )

            score = float(row.get("score", 0.0) or 0.0)
            # A small positive default keeps zero-valued but valid keywords usable.
            weight = score if score > 0.0 else 1.0
            result.add_edge(
                paper_node,
                concept_node,
                relation="MENTIONS_KEYWORD",
                weight=weight,
            )

    return result


class KnowledgeGraphEnhancedRetriever:
    """
    Rerank base-retrieval candidates using knowledge-graph evidence.

    The score is computed as::

        final_score = retrieval_weight * normalized_retrieval_score \
                    + graph_weight * normalized_graph_score

    Parameters
    ----------
    base_retriever : object
        Indexed retriever exposing ``search(query, top_k)`` and returning
        objects compatible with :class:`RetrievalResult`.
    graph : networkx graph
        Graph containing PAPER nodes named ``paper::<doc_id>`` and typed
        concept/topic/author nodes.
    retrieval_weight, graph_weight : float
        Non-negative score weights. They are normalized to sum to one.
    candidate_multiplier : int
        Retrieve this multiple of ``top_k`` before reranking.
    expansion_weight : float
        Discount applied to one-hop concept evidence.
    """

    def __init__(
        self,
        base_retriever,
        graph,
        *,
        retrieval_weight: float = 0.75,
        graph_weight: float = 0.25,
        candidate_multiplier: int = 3,
        expansion_weight: float = 0.5,
    ) -> None:
        if retrieval_weight < 0.0 or graph_weight < 0.0:
            raise ValueError("retrieval_weight and graph_weight must be non-negative")
        total_weight = retrieval_weight + graph_weight
        if total_weight <= 0.0:
            raise ValueError("At least one score weight must be positive")
        if candidate_multiplier <= 0:
            raise ValueError("candidate_multiplier must be positive")
        if not 0.0 <= expansion_weight <= 1.0:
            raise ValueError("expansion_weight must be between 0 and 1")

        self.base_retriever = base_retriever
        self.graph = graph
        self.retrieval_weight = retrieval_weight / total_weight
        self.graph_weight = graph_weight / total_weight
        self.candidate_multiplier = candidate_multiplier
        self.expansion_weight = expansion_weight

        self._node_labels: Dict[str, str] = {}
        self._node_tokens: Dict[str, set[str]] = {}
        self._node_types: Dict[str, str] = {}
        self._paper_features: Dict[str, Dict[str, float]] = {}
        self._feature_relations: Dict[Tuple[str, str], str] = {}
        self._feature_neighbors: Dict[str, Dict[str, float]] = {}
        self._build_graph_index()

    def search(self, query: str, top_k: int = 10) -> List[KGEnhancedResult]:
        """Return graph-reranked results for ``query``."""
        if top_k <= 0:
            raise ValueError("top_k must be a positive integer")
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string")

        candidate_k = max(top_k, top_k * self.candidate_multiplier)
        base_results = list(self.base_retriever.search(query, top_k=candidate_k))
        if not base_results:
            return []

        matched_nodes = self.match_query_nodes(query)
        raw_graph_scores: List[float] = []
        evidence_by_doc: Dict[str, Tuple[str, ...]] = {}

        for result in base_results:
            graph_score, evidence = self._score_document(str(result.doc_id), matched_nodes)
            raw_graph_scores.append(graph_score)
            evidence_by_doc[str(result.doc_id)] = tuple(evidence)

        normalized_base = _minmax([float(result.score) for result in base_results])
        normalized_graph = _minmax(raw_graph_scores)

        reranked = []
        for original_position, (result, base_norm, graph_raw, graph_norm) in enumerate(
            zip(base_results, normalized_base, raw_graph_scores, normalized_graph),
            start=1,
        ):
            final_score = (
                self.retrieval_weight * base_norm
                + self.graph_weight * graph_norm
            )
            reranked.append(
                {
                    "doc_id": str(result.doc_id),
                    "score": float(final_score),
                    "text": str(getattr(result, "text", "")),
                    "base_score": float(result.score),
                    "normalized_base_score": float(base_norm),
                    "graph_score": float(graph_raw),
                    "normalized_graph_score": float(graph_norm),
                    "evidence": evidence_by_doc[str(result.doc_id)],
                    "original_position": original_position,
                }
            )

        reranked.sort(
            key=lambda item: (
                -item["score"],
                -item["normalized_base_score"],
                item["original_position"],
                item["doc_id"],
            )
        )

        return [
            KGEnhancedResult(
                doc_id=item["doc_id"],
                score=item["score"],
                rank=rank,
                text=item["text"],
                base_score=item["base_score"],
                normalized_base_score=item["normalized_base_score"],
                graph_score=item["graph_score"],
                normalized_graph_score=item["normalized_graph_score"],
                evidence=item["evidence"],
            )
            for rank, item in enumerate(reranked[:top_k], start=1)
        ]

    def match_query_nodes(self, query: str) -> List[str]:
        """Return graph feature nodes whose labels are supported by the query."""
        query_label = _normalize_label(query)
        query_tokens = _tokens(query)
        matches: List[Tuple[float, str]] = []

        for node, label in self._node_labels.items():
            node_type = self._node_types[node]
            label_tokens = self._node_tokens[node]
            if not label_tokens:
                continue

            score = 0.0
            if label and len(label) >= 3 and label in query_label:
                score = 1.0
            elif label_tokens.issubset(query_tokens):
                score = 0.95
            else:
                overlap = len(label_tokens & query_tokens)
                if overlap >= 2 and overlap / len(label_tokens) >= 0.67:
                    score = 0.75

            if node_type == "TOPIC":
                for alias in _TOPIC_ALIASES.get(label, ()):
                    alias_label = _normalize_label(alias)
                    alias_tokens = _tokens(alias)
                    if alias_label in query_label or alias_tokens.issubset(query_tokens):
                        score = max(score, 0.9)

            if score > 0.0:
                matches.append((score, node))

        matches.sort(key=lambda item: (-item[0], self._node_labels[item[1]], item[1]))
        return [node for _, node in matches]

    def _build_graph_index(self) -> None:
        for node, data in self.graph.nodes(data=True):
            node_type = str(data.get("node_type", "")).upper()
            if node_type not in _NODE_TYPE_WEIGHTS:
                continue

            label = _normalize_label(data.get("label", node))
            self._node_labels[str(node)] = label
            self._node_tokens[str(node)] = _tokens(label)
            self._node_types[str(node)] = node_type

        for source_node, target_node, data in self._iter_edges():
            source = str(source_node)
            target = str(target_node)
            source_type = str(
                self.graph.nodes[source_node].get("node_type", "")
            ).upper()
            target_type = str(
                self.graph.nodes[target_node].get("node_type", "")
            ).upper()
            weight = _edge_weight(data)
            relation = str(data.get("relation", "RELATED_TO"))

            if source_type == "PAPER" and target in self._node_labels:
                doc_id = self._paper_doc_id(source)
                features = self._paper_features.setdefault(doc_id, {})
                features[target] = features.get(target, 0.0) + weight
                self._feature_relations[(doc_id, target)] = relation
            elif target_type == "PAPER" and source in self._node_labels:
                doc_id = self._paper_doc_id(target)
                features = self._paper_features.setdefault(doc_id, {})
                features[source] = features.get(source, 0.0) + weight
                self._feature_relations[(doc_id, source)] = relation
            elif source in self._node_labels and target in self._node_labels:
                source_neighbors = self._feature_neighbors.setdefault(source, {})
                target_neighbors = self._feature_neighbors.setdefault(target, {})
                source_neighbors[target] = max(
                    weight, source_neighbors.get(target, 0.0)
                )
                target_neighbors[source] = max(
                    weight, target_neighbors.get(source, 0.0)
                )

    def _iter_edges(self):
        if getattr(self.graph, "is_multigraph", lambda: False)():
            for source, target, _key, data in self.graph.edges(keys=True, data=True):
                yield source, target, data
        else:
            for source, target, data in self.graph.edges(data=True):
                yield source, target, data

    @staticmethod
    def _paper_doc_id(node: str) -> str:
        return node.split("paper::", 1)[1] if node.startswith("paper::") else node

    def _score_document(
        self,
        doc_id: str,
        matched_nodes: Sequence[str],
    ) -> Tuple[float, List[str]]:
        features = self._paper_features.get(doc_id, {})
        if not features or not matched_nodes:
            return 0.0, []

        raw_score = 0.0
        evidence: List[str] = []
        seen_evidence = set()

        for matched_node in matched_nodes:
            node_type = self._node_types.get(matched_node, "CONCEPT")
            type_weight = _NODE_TYPE_WEIGHTS.get(node_type, 1.0)

            if matched_node in features:
                contribution = type_weight * math.log1p(features[matched_node])
                raw_score += contribution
                relation = self._feature_relations.get((doc_id, matched_node), "CONNECTED_TO")
                item = f"direct:{relation}:{self._node_labels[matched_node]}"
                if item not in seen_evidence:
                    evidence.append(item)
                    seen_evidence.add(item)

            for related_node, relation_weight in self._feature_neighbors.get(
                matched_node, {}
            ).items():
                if related_node not in features:
                    continue

                related_type = self._node_types.get(related_node, "CONCEPT")
                related_type_weight = _NODE_TYPE_WEIGHTS.get(related_type, 1.0)
                contribution = (
                    self.expansion_weight
                    * related_type_weight
                    * math.log1p(features[related_node])
                    * math.log1p(relation_weight)
                )
                raw_score += contribution
                item = (
                    "expanded:"
                    f"{self._node_labels[matched_node]}->"
                    f"{self._node_labels[related_node]}"
                )
                if item not in seen_evidence:
                    evidence.append(item)
                    seen_evidence.add(item)

        return raw_score, evidence
