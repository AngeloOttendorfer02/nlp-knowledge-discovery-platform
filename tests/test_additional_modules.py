from __future__ import annotations

import numpy as np
import networkx as nx
import pytest

from src.bias.debiasing import bias_direction, neutralize, neutralize_embeddings
from src.bias.embedding_bias import association, cosine_similarity, group_bias_score, mean_vector
from src.embeddings.network_embeddings import SemanticNetworkBuilder
from src.gnn.dataset_builder import build_graph_dataset
from src.gnn.train_gnn import normalized_adjacency, train_gcn
from src.knowledge_graph.graph_queries import GraphQueryEngine
from src.llm_kg.graph_retrieval_qa import build_graph_qa_prompt, graph_context
from src.llm_kg.rag_pipeline import RAGPipeline, RetrievedContext, build_rag_prompt
from src.retrieval.bm25_retriever import RetrievalResult
from src.retrieval.vector_store import VectorStore


def test_vector_store_numpy_backend_search_and_persistence(tmp_path):
    store = VectorStore(dim=2, use_faiss=False)
    store.add(["d1", "d2"], np.array([[1.0, 0.0], [0.0, 1.0]], dtype="float32"))

    hits = store.search(np.array([0.9, 0.1], dtype="float32"), top_k=2)

    assert len(store) == 2
    assert [hit.doc_id for hit in hits] == ["d1", "d2"]
    assert hits[0].rank == 1
    assert hits[0].score > hits[1].score

    save_dir = tmp_path / "store"
    store.save(str(save_dir))
    restored = VectorStore.load(str(save_dir))

    assert [hit.doc_id for hit in restored.search(np.array([0.0, 1.0], dtype="float32"), top_k=1)] == ["d2"]


def test_vector_store_validates_input_shapes():
    store = VectorStore(dim=3, use_faiss=False)

    with pytest.raises(ValueError):
        store.add(["d1"], np.array([[1.0, 2.0]], dtype="float32"))

    store.add(["d1"], np.array([[1.0, 0.0, 0.0]], dtype="float32"))
    with pytest.raises(ValueError):
        store.search(np.array([1.0, 2.0], dtype="float32"))


def test_graph_query_engine_core_queries():
    graph = nx.MultiDiGraph()
    graph.add_node("paper::d1", node_type="PAPER")
    graph.add_node("concept::graph", node_type="CONCEPT")
    graph.add_node("topic::cs.CL", node_type="TOPIC")
    graph.add_edge("paper::d1", "concept::graph", relation="MENTIONS", weight=2)
    graph.add_edge("paper::d1", "topic::cs.CL", relation="BELONGS_TO_TOPIC", weight=1)

    engine = GraphQueryEngine(graph)

    assert engine.neighbors("paper::d1", relation="MENTIONS") == ["concept::graph"]
    assert engine.get_papers_for_concept("Graph") == ["paper::d1"]
    assert engine.shortest_path("paper::d1", "concept::graph") == ["paper::d1", "concept::graph"]
    assert engine.subgraph_by_type("CONCEPT").number_of_nodes() == 1
    assert engine.ego_graph("paper::d1").number_of_nodes() == 3
    assert engine.connected_components_summary()["largest_component_size"] == 3
    assert engine.most_central_nodes(top_n=1)[0][0] == "paper::d1"


def test_semantic_network_builder_threshold_knn_and_stats():
    builder = SemanticNetworkBuilder(similarity_threshold=0.7)
    doc_ids = ["d1", "d2", "d3"]
    embeddings = np.array([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0]], dtype="float32")

    threshold_graph = builder.build_threshold_network(doc_ids, embeddings)
    knn_graph = builder.build_knn_network(doc_ids, embeddings, k=1)
    communities = builder.detect_communities(threshold_graph)
    stats = builder.network_stats(threshold_graph)

    assert threshold_graph.has_edge("d1", "d2")
    assert knn_graph.number_of_edges() >= 2
    assert set(communities) == set(doc_ids)
    assert stats["num_nodes"] == 3
    assert stats["num_edges"] >= 1


def test_bias_helpers_compute_and_neutralize_vectors():
    embeddings = {
        "science": np.array([1.0, 0.0]),
        "arts": np.array([0.0, 1.0]),
        "logic": np.array([1.0, 0.0]),
        "creative": np.array([0.0, 1.0]),
        "target": np.array([1.0, 1.0]),
    }

    assert cosine_similarity(embeddings["science"], embeddings["logic"]) == 1.0
    assert np.allclose(mean_vector(["science", "logic"], embeddings), np.array([1.0, 0.0]))
    scores = association(["target"], ["logic"], ["creative"], embeddings)
    assert len(scores) == 1
    assert isinstance(group_bias_score(["science"], ["arts"], ["logic"], ["creative"], embeddings), float)

    direction = bias_direction(["science"], ["arts"], embeddings)
    neutral = neutralize(embeddings["target"], direction)
    updated = neutralize_embeddings(["target"], embeddings, direction)

    assert abs(np.dot(neutral, direction)) < 1e-8
    assert abs(np.dot(updated["target"], direction)) < 1e-8
    assert embeddings["target"] is not updated["target"]


def test_rag_pipeline_builds_prompt_and_returns_contexts():
    class Retriever:
        def search(self, query, top_k=5):
            return [RetrievalResult(doc_id="d1", score=2.0, rank=1, text=f"context for {query}")]

    prompts = []

    def generator(prompt: str) -> str:
        prompts.append(prompt)
        return "answer"

    prompt = build_rag_prompt("What is retrieval?", [RetrievedContext("d1", "Retrieval finds documents.", 1.0)])
    pipeline = RAGPipeline(Retriever(), generator)
    response = pipeline.answer("graph retrieval", top_k=1)

    assert "doc_id=d1" in prompt
    assert response["answer"] == "answer"
    assert response["contexts"][0].doc_id == "d1"
    assert "graph retrieval" in prompts[0]


def test_graph_retrieval_qa_context_and_prompt():
    graph = nx.MultiDiGraph()
    graph.add_node("paper::d1")
    graph.add_node("concept::graph")
    graph.add_edge("paper::d1", "concept::graph", relation="MENTIONS")

    context = graph_context(graph, "paper::d1")
    prompt = build_graph_qa_prompt("Which concept is mentioned?", context)

    assert "MENTIONS" in context
    assert "Which concept" in prompt
    assert graph_context(graph, "missing") == ""


def test_graph_dataset_builder_and_normalized_adjacency():
    graph = nx.MultiDiGraph()
    graph.add_edge("a", "b")
    dataset = build_graph_dataset(graph, labels={"a": 0, "b": 1})
    adjacency = normalized_adjacency(len(dataset.node_ids), dataset.edge_index)

    assert dataset.features.shape == (2, 3)
    assert dataset.edge_index.shape == (2, 1)
    assert dataset.labels.tolist() == [0, 1]
    assert adjacency.shape == (2, 2)
    assert np.allclose(adjacency, adjacency.T)


def test_train_gcn_runs_when_torch_is_available():
    torch = pytest.importorskip("torch")
    del torch
    features = np.array([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.1, 0.9]], dtype="float32")
    edge_index = np.array([[0, 1, 2], [1, 2, 3]], dtype="int64")
    labels = np.array([0, 0, 1, 1], dtype="int64")

    result = train_gcn(features, edge_index, labels, epochs=3, hidden_dim=4, seed=1)

    assert len(result.losses) == 3
    assert result.predictions.shape == (4,)
