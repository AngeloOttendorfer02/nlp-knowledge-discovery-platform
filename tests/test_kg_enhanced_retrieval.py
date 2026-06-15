from __future__ import annotations

import networkx as nx

from src.retrieval.bm25_retriever import RetrievalResult
from src.retrieval.kg_enhanced_retriever import (
    KnowledgeGraphEnhancedRetriever,
    augment_graph_with_keywords,
)


class FakeBaseRetriever:
    def search(self, query, top_k=10):
        return [
            RetrievalResult(doc_id="d1", score=1.0, rank=1, text="baseline first"),
            RetrievalResult(doc_id="d2", score=0.9, rank=2, text="graph relevant"),
        ][:top_k]


def _graph_with_graph_evidence():
    graph = nx.MultiDiGraph()
    graph.add_node("paper::d1", node_type="PAPER", label="Paper 1")
    graph.add_node("paper::d2", node_type="PAPER", label="Paper 2")
    graph.add_node(
        "concept::knowledge graph",
        node_type="CONCEPT",
        label="knowledge graph",
    )
    graph.add_edge(
        "paper::d2",
        "concept::knowledge graph",
        relation="MENTIONS",
        weight=2,
    )
    return graph


def test_graph_evidence_can_rerank_base_results():
    retriever = KnowledgeGraphEnhancedRetriever(
        FakeBaseRetriever(),
        _graph_with_graph_evidence(),
        retrieval_weight=0.3,
        graph_weight=0.7,
        candidate_multiplier=2,
    )

    results = retriever.search("knowledge graph methods", top_k=2)

    assert [result.doc_id for result in results] == ["d2", "d1"]
    assert results[0].graph_score > 0.0
    assert results[0].evidence
    assert results[0].rank == 1
    assert all(0.0 <= result.normalized_base_score <= 1.0 for result in results)
    assert all(0.0 <= result.normalized_graph_score <= 1.0 for result in results)


def test_no_graph_match_preserves_base_order():
    retriever = KnowledgeGraphEnhancedRetriever(
        FakeBaseRetriever(),
        _graph_with_graph_evidence(),
        retrieval_weight=0.3,
        graph_weight=0.7,
    )

    results = retriever.search("unrelated vision query", top_k=2)

    assert [result.doc_id for result in results] == ["d1", "d2"]
    assert all(result.graph_score == 0.0 for result in results)


def test_keyword_rows_can_enrich_graph_without_mutating_original():
    graph = nx.MultiDiGraph()
    graph.add_node("paper::d1", node_type="PAPER", label="Paper 1")

    enriched = augment_graph_with_keywords(
        graph,
        [{"doc_id": "d1", "keyword": "semantic retrieval", "score": 0.8}],
    )

    assert "concept::semantic retrieval" not in graph
    assert "concept::semantic retrieval" in enriched
    assert enriched.has_edge("paper::d1", "concept::semantic retrieval")


def test_invalid_score_weights_are_rejected():
    try:
        KnowledgeGraphEnhancedRetriever(
            FakeBaseRetriever(),
            _graph_with_graph_evidence(),
            retrieval_weight=0.0,
            graph_weight=0.0,
        )
    except ValueError as exc:
        assert "positive" in str(exc)
    else:
        raise AssertionError("Expected invalid zero weights to raise ValueError")


def test_kg_experiment_runner_writes_outputs(tmp_path):
    import json

    import pandas as pd

    from src.experiments.run_kg_enhanced_retrieval import (
        run_kg_enhanced_experiment,
    )

    documents_path = tmp_path / "processed_documents.csv"
    pd.DataFrame(
        [
            {"doc_id": "d1", "text": "generic retrieval"},
            {"doc_id": "d2", "text": "knowledge graph retrieval"},
        ]
    ).to_csv(documents_path, index=False)

    queries_path = tmp_path / "queries.json"
    queries_path.write_text(
        json.dumps(
            [
                {
                    "query_id": "q1",
                    "query": "knowledge graph",
                    "relevant_doc_ids": ["d2"],
                }
            ]
        ),
        encoding="utf-8",
    )

    graph_path = tmp_path / "knowledge_graph.graphml"
    nx.write_graphml(_graph_with_graph_evidence(), graph_path)

    class IndexableFakeRetriever(FakeBaseRetriever):
        def index(self, doc_ids, texts, **kwargs):
            return self

    results, metrics, result_path, metric_path = run_kg_enhanced_experiment(
        documents_path=documents_path,
        queries_path=queries_path,
        graph_path=graph_path,
        output_dir=tmp_path / "reports",
        base_method="semantic",
        base_retriever=IndexableFakeRetriever(),
        keywords_path=None,
        top_ks=(1, 2),
        retrieval_weight=0.3,
        graph_weight=0.7,
    )

    assert result_path.exists()
    assert metric_path.exists()
    assert results.iloc[0]["doc_id"] == "d2"
    assert "graph_evidence" in results.columns
    assert metrics["k"].tolist() == [1, 2]
    assert metrics.loc[metrics["k"] == 1, "precision_at_k"].iloc[0] == 1.0


def test_arxiv_topic_alias_matches_natural_language_query():
    graph = nx.MultiDiGraph()
    graph.add_node("paper::d1", node_type="PAPER", label="AI Paper")
    graph.add_node("topic::cs.AI", node_type="TOPIC", label="cs.AI")
    graph.add_edge(
        "paper::d1",
        "topic::cs.AI",
        relation="BELONGS_TO_TOPIC",
        weight=1,
    )

    retriever = KnowledgeGraphEnhancedRetriever(FakeBaseRetriever(), graph)

    matches = retriever.match_query_nodes("artificial intelligence methods")

    assert "topic::cs.AI" in matches


def test_short_topic_alias_does_not_match_inside_another_word():
    graph = nx.MultiDiGraph()
    graph.add_node("paper::d1", node_type="PAPER", label="AI Paper")
    graph.add_node("topic::cs.AI", node_type="TOPIC", label="cs.AI")
    graph.add_edge(
        "paper::d1",
        "topic::cs.AI",
        relation="BELONGS_TO_TOPIC",
        weight=1,
    )

    retriever = KnowledgeGraphEnhancedRetriever(FakeBaseRetriever(), graph)

    matches = retriever.match_query_nodes("training methods")

    assert "topic::cs.AI" not in matches
