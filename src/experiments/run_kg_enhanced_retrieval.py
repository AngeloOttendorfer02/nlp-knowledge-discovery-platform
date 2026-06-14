"""
Run a knowledge graph-enhanced retrieval experiment.

Example:
    python -m src.experiments.run_kg_enhanced_retrieval
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List

import networkx as nx
import pandas as pd

from src.evaluation.retrieval_metrics import evaluate_retrieval_run
from src.experiments.run_bm25_baseline import (
    load_processed_documents,
    load_relevance_queries,
)
from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.kg_enhanced_retriever import KnowledgeGraphEnhancedRetriever


def load_knowledge_graph(path: Path):
    if not path.exists():
        raise FileNotFoundError(
            f"Knowledge graph not found: {path}\n"
            "Run first:\n"
            "    python -m src.pipeline.run_pipeline --skip-embeddings"
        )

    return nx.read_graphml(path)


def run_kg_enhanced_retrieval(
    documents_path: Path,
    queries_path: Path,
    graph_path: Path,
    results_path: Path,
    metrics_path: Path,
    top_k: int = 5,
    retrieval_weight: float = 0.7,
    graph_weight: float = 0.3,
) -> Dict[str, float]:
    df = load_processed_documents(documents_path)
    queries = load_relevance_queries(queries_path)
    graph = load_knowledge_graph(graph_path)

    doc_ids = df["doc_id"].astype(str).tolist()
    texts = df["text"].fillna("").astype(str).tolist()

    base_retriever = BM25Retriever()
    base_retriever.index(doc_ids, texts)

    kg_retriever = KnowledgeGraphEnhancedRetriever(
        base_retriever=base_retriever,
        graph=graph,
        retrieval_weight=retrieval_weight,
        graph_weight=graph_weight,
    )

    results_by_query: Dict[str, List[str]] = {}
    relevance_sets: Dict[str, List[str]] = {}
    result_rows: List[Dict[str, Any]] = []

    for query_item in queries:
        query_id = str(query_item["query_id"])
        query_text = str(query_item["query"])
        relevant_doc_ids = [str(doc_id) for doc_id in query_item["relevant_doc_ids"]]
        relevant_set = set(relevant_doc_ids)

        hits = kg_retriever.search(query_text, top_k=top_k)
        retrieved_ids = [str(hit.doc_id) for hit in hits]

        results_by_query[query_id] = retrieved_ids
        relevance_sets[query_id] = relevant_doc_ids

        for hit in hits:
            result_rows.append(
                {
                    "query_id": query_id,
                    "query": query_text,
                    "rank": hit.rank,
                    "doc_id": str(hit.doc_id),
                    "score": hit.score,
                    "retrieval_score": hit.retrieval_score,
                    "graph_score": hit.graph_score,
                    "is_relevant": str(hit.doc_id) in relevant_set,
                    "text": hit.text,
                }
            )

    metrics = evaluate_retrieval_run(
        results=results_by_query,
        relevance_sets=relevance_sets,
        k=top_k,
    )

    results_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(result_rows).to_csv(results_path, index=False)
    pd.DataFrame(
        [
            {
                "method": "KG-enhanced BM25",
                "retrieval_weight": retrieval_weight,
                "graph_weight": graph_weight,
                **metrics,
            }
        ]
    ).to_csv(metrics_path, index=False)

    print("KG-enhanced retrieval experiment completed.")
    print(f"Results saved to: {results_path}")
    print(f"Metrics saved to: {metrics_path}")
    print(metrics)

    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run KG-enhanced retrieval experiment.")

    parser.add_argument(
        "--documents",
        type=Path,
        default=Path("data/processed/processed_documents.csv"),
    )
    parser.add_argument(
        "--queries",
        type=Path,
        default=Path("data/evaluation/retrieval_queries.example.json"),
    )
    parser.add_argument(
        "--graph",
        type=Path,
        default=Path("data/graphs/knowledge_graph.graphml"),
    )
    parser.add_argument(
        "--results",
        type=Path,
        default=Path("reports/tables/kg_enhanced_results.csv"),
    )
    parser.add_argument(
        "--metrics",
        type=Path,
        default=Path("reports/tables/kg_enhanced_metrics.csv"),
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--retrieval-weight",
        type=float,
        default=0.7,
    )
    parser.add_argument(
        "--graph-weight",
        type=float,
        default=0.3,
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    run_kg_enhanced_retrieval(
        documents_path=args.documents,
        queries_path=args.queries,
        graph_path=args.graph,
        results_path=args.results,
        metrics_path=args.metrics,
        top_k=args.top_k,
        retrieval_weight=args.retrieval_weight,
        graph_weight=args.graph_weight,
    )


if __name__ == "__main__":
    main()