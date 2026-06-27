from __future__ import annotations

from pathlib import Path

import networkx as nx
import pandas as pd

from src.experiments.generate_report_assets import generate_report_assets


def test_generate_report_assets_creates_tables_and_figures(tmp_path):
    processed_docs = pd.DataFrame(
        [
            {
                "doc_id": "d1",
                "title": "Graph Neural Networks",
                "abstract": "Graphs and neural networks are used together.",
                "categories": "cs.AI cs.LG",
                "text": "Graph neural networks are useful.",
            },
            {
                "doc_id": "d2",
                "title": "Knowledge Graph Retrieval",
                "abstract": "Knowledge graphs improve retrieval systems.",
                "categories": "cs.IR cs.AI",
                "text": "Knowledge graphs improve retrieval.",
            },
        ]
    )
    entities = pd.DataFrame(
        [{"doc_id": "d1", "text": "graph", "label": "ORG"}, {"doc_id": "d2", "text": "retrieval", "label": "MISC"}]
    )
    keywords = pd.DataFrame(
        [{"doc_id": "d1", "keyword": "graph", "score": 0.9}, {"doc_id": "d2", "keyword": "retrieval", "score": 0.8}]
    )
    relations = pd.DataFrame([{"doc_id": "d1", "source": "graph", "target": "network", "relation": "RELATED_TO"}])

    graph = nx.MultiDiGraph()
    graph.add_node("paper::d1", node_type="PAPER", label="d1")
    graph.add_node("paper::d2", node_type="PAPER", label="d2")
    graph.add_node("author::Alice", node_type="AUTHOR", label="Alice")
    graph.add_node("concept::graph", node_type="CONCEPT", label="graph")
    graph.add_node("topic::cs.AI", node_type="TOPIC", label="cs.AI")
    graph.add_edge("paper::d1", "author::Alice", relation="AUTHORED_BY", weight=1)
    graph.add_edge("paper::d1", "concept::graph", relation="MENTIONS", weight=2)
    graph.add_edge("paper::d1", "topic::cs.AI", relation="BELONGS_TO_TOPIC", weight=1)
    graph.add_edge("paper::d2", "concept::graph", relation="MENTIONS", weight=1)

    processed_path = tmp_path / "processed_documents.csv"
    entities_path = tmp_path / "entities.csv"
    keywords_path = tmp_path / "keywords.csv"
    relations_path = tmp_path / "relations.csv"
    graph_path = tmp_path / "graph.json"
    output_dir = tmp_path / "reports"

    processed_docs.to_csv(processed_path, index=False)
    entities.to_csv(entities_path, index=False)
    keywords.to_csv(keywords_path, index=False)
    relations.to_csv(relations_path, index=False)
    nx.node_link_data(graph)
    import json

    with graph_path.open("w", encoding="utf-8") as handle:
        json.dump(nx.node_link_data(graph), handle)

    generated = generate_report_assets(
        processed_documents_path=processed_path,
        entities_path=entities_path,
        keywords_path=keywords_path,
        relations_path=relations_path,
        graph_path=graph_path,
        output_dir=output_dir,
        top_n=3,
    )

    assert generated["dataset_statistics"].exists()
    assert generated["graph_statistics"].exists()
    assert generated["top_authors"].exists()
    assert generated["top_concepts"].exists()
    assert generated["topic_summary"].exists()

    assert (output_dir / "figures" / "category_distribution.png").exists()
    assert (output_dir / "figures" / "degree_distribution.png").exists()
    assert (output_dir / "figures" / "knowledge_graph_top_nodes.png").exists()
    assert (output_dir / "figures" / "retrieval_metrics_comparison.png").exists()

    dataset_stats = pd.read_csv(generated["dataset_statistics"])
    graph_stats = pd.read_csv(generated["graph_statistics"])
    topic_summary = pd.read_csv(generated["topic_summary"])

    assert not dataset_stats.empty
    assert not graph_stats.empty
    assert not topic_summary.empty
    assert graph_stats.iloc[0]["num_nodes"] >= 4
