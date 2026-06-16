from __future__ import annotations

import json

import networkx as nx
import pandas as pd

from src.experiments.compare_retrieval_methods import run_retrieval_comparison
from src.retrieval.bm25_retriever import RetrievalResult


class FakeSemanticRetriever:
    def index(self, doc_ids, texts, **kwargs):
        self.doc_ids = list(doc_ids)
        return self

    def search(self, query, top_k=10):
        if "language" in query.lower():
            order = ["d2", "d1", "d3"]
        else:
            order = ["d1", "d2", "d3"]
        return [
            RetrievalResult(doc_id=doc_id, score=1.0 / rank, rank=rank)
            for rank, doc_id in enumerate(order[:top_k], start=1)
        ]


class FakeKgBaseRetriever:
    def index(self, doc_ids, texts, **kwargs):
        return self

    def search(self, query, top_k=10):
        return [
            RetrievalResult(doc_id="d1", score=1.0, rank=1),
            RetrievalResult(doc_id="d2", score=0.9, rank=2),
            RetrievalResult(doc_id="d3", score=0.1, rank=3),
        ][:top_k]


def _write_comparison_fixtures(tmp_path):
    documents_path = tmp_path / "processed_documents.csv"
    pd.DataFrame(
        [
            {"doc_id": "d1", "text": "knowledge graph retrieval"},
            {"doc_id": "d2", "text": "language model embeddings"},
            {"doc_id": "d3", "text": "vision segmentation"},
        ]
    ).to_csv(documents_path, index=False)

    queries_path = tmp_path / "queries.json"
    queries_path.write_text(
        json.dumps(
            [
                {
                    "query_id": "q1",
                    "query": "knowledge graph",
                    "relevant_doc_ids": ["d1"],
                },
                {
                    "query_id": "q2",
                    "query": "language embeddings",
                    "relevant_doc_ids": ["d2"],
                },
            ]
        ),
        encoding="utf-8",
    )

    graph = nx.MultiDiGraph()
    graph.add_node("paper::d1", node_type="PAPER", label="Graph Paper")
    graph.add_node("paper::d2", node_type="PAPER", label="Language Paper")
    graph.add_node("concept::knowledge graph", node_type="CONCEPT", label="knowledge graph")
    graph.add_node("concept::language", node_type="CONCEPT", label="language")
    graph.add_edge(
        "paper::d1",
        "concept::knowledge graph",
        relation="MENTIONS",
        weight=2,
    )
    graph.add_edge("paper::d2", "concept::language", relation="MENTIONS", weight=2)
    graph_path = tmp_path / "knowledge_graph.graphml"
    nx.write_graphml(graph, graph_path)

    return documents_path, queries_path, graph_path


def test_retrieval_comparison_writes_tables_and_figure(tmp_path):
    documents_path, queries_path, graph_path = _write_comparison_fixtures(tmp_path)

    ranked, comparison, per_query, figure_path = run_retrieval_comparison(
        documents_path=documents_path,
        queries_path=queries_path,
        graph_path=graph_path,
        output_dir=tmp_path / "tables",
        figure_dir=tmp_path / "figures",
        top_ks=(1, 2),
        semantic_retriever=FakeSemanticRetriever(),
        kg_base_retriever=FakeKgBaseRetriever(),
        keywords_path=None,
        retrieval_weight=0.3,
        graph_weight=0.7,
    )

    assert (tmp_path / "tables" / "all_retrieval_results.csv").exists()
    assert (tmp_path / "tables" / "retrieval_comparison.csv").exists()
    assert (tmp_path / "tables" / "per_query_comparison.csv").exists()
    assert figure_path.exists()

    assert set(comparison["method"]) == {"bm25", "semantic", "kg_enhanced"}
    assert comparison["k"].tolist().count(1) == 3
    assert set(per_query.columns) >= {
        "method",
        "query_id",
        "k",
        "precision_at_k",
        "recall_at_k",
        "reciprocal_rank",
        "top_doc_id",
        "top_doc_relevant",
    }
    assert set(ranked["method"]) == {"bm25", "semantic", "kg_enhanced"}
    assert len(per_query) == 3 * 2 * 2


def test_comparison_can_run_selected_methods_only(tmp_path):
    documents_path, queries_path, graph_path = _write_comparison_fixtures(tmp_path)

    _, comparison, per_query, _ = run_retrieval_comparison(
        documents_path=documents_path,
        queries_path=queries_path,
        graph_path=graph_path,
        output_dir=tmp_path / "tables",
        figure_dir=tmp_path / "figures",
        methods=("semantic",),
        top_ks=(1,),
        semantic_retriever=FakeSemanticRetriever(),
        keywords_path=None,
    )

    assert comparison["method"].tolist() == ["semantic"]
    assert per_query["method"].unique().tolist() == ["semantic"]
