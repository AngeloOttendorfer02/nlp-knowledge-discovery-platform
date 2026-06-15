import json

import networkx as nx
import pandas as pd

from src.experiments.run_kg_enhanced_retrieval import run_kg_enhanced_retrieval
from src.retrieval.bm25_retriever import RetrievalResult
from src.retrieval.kg_enhanced_retriever import KnowledgeGraphEnhancedRetriever


class FakeBaseRetriever:
    def search(self, query, top_k=10):
        return [
            RetrievalResult(
                doc_id="2",
                score=10.0,
                rank=1,
                text="Transformer document.",
            ),
            RetrievalResult(
                doc_id="1",
                score=5.0,
                rank=2,
                text="Graph document.",
            ),
        ]


def build_test_graph():
    graph = nx.MultiDiGraph()

    graph.add_node("paper::1", node_type="PAPER", label="Graph Paper")
    graph.add_node("paper::2", node_type="PAPER", label="Transformer Paper")
    graph.add_node("concept::graph", node_type="CONCEPT", label="graph")

    graph.add_edge("paper::1", "concept::graph", relation="MENTIONS", weight=3)

    return graph


def test_score_normalization():
    scores = {
        "a": 10.0,
        "b": 5.0,
    }

    normalized = KnowledgeGraphEnhancedRetriever.normalize_scores(scores)

    assert normalized["a"] == 1.0
    assert normalized["b"] == 0.5


def test_graph_evidence_can_rerank_candidate():
    graph = build_test_graph()

    retriever = KnowledgeGraphEnhancedRetriever(
        base_retriever=FakeBaseRetriever(),
        graph=graph,
        retrieval_weight=0.4,
        graph_weight=0.6,
    )

    results = retriever.search("graph retrieval", top_k=2)

    assert results[0].doc_id == "1"
    assert results[0].graph_score > 0


def test_kg_enhanced_experiment_writes_results_and_metrics(tmp_path):
    documents_path = tmp_path / "processed_documents.csv"
    queries_path = tmp_path / "queries.json"
    graph_path = tmp_path / "knowledge_graph.graphml"
    results_path = tmp_path / "kg_enhanced_results.csv"
    metrics_path = tmp_path / "kg_enhanced_metrics.csv"

    pd.DataFrame(
        [
            {
                "doc_id": "1",
                "text": "Graph neural networks for scientific document retrieval.",
            },
            {
                "doc_id": "2",
                "text": "Transformer models for named entity recognition.",
            },
        ]
    ).to_csv(documents_path, index=False)

    queries_path.write_text(
        json.dumps(
            [
                {
                    "query_id": "q1",
                    "query": "graph retrieval",
                    "relevant_doc_ids": ["1"],
                }
            ]
        ),
        encoding="utf-8",
    )

    graph = build_test_graph()
    nx.write_graphml(graph, graph_path)

    metrics = run_kg_enhanced_retrieval(
        documents_path=documents_path,
        queries_path=queries_path,
        graph_path=graph_path,
        results_path=results_path,
        metrics_path=metrics_path,
        top_k=2,
        retrieval_weight=0.5,
        graph_weight=0.5,
    )

    assert results_path.exists()
    assert metrics_path.exists()

    results_df = pd.read_csv(results_path)
    metrics_df = pd.read_csv(metrics_path)

    assert {
        "query_id",
        "query",
        "rank",
        "doc_id",
        "score",
        "retrieval_score",
        "graph_score",
        "is_relevant",
        "text",
    }.issubset(results_df.columns)

    assert {
        "method",
        "retrieval_weight",
        "graph_weight",
        "precision@2",
        "recall@2",
        "mrr",
    }.issubset(metrics_df.columns)

    assert metrics["recall@2"] == 1.0