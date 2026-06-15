"""Run the reproducible knowledge-graph-enhanced retrieval experiment."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Sequence

import networkx as nx
import yaml

from src.experiments.common import (
    evaluate_rankings,
    load_documents,
    load_evaluation_queries,
    normalize_top_ks,
    validate_relevance_against_corpus,
    write_experiment_outputs,
)
from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.embedding_retriever import EmbeddingRetriever
from src.retrieval.kg_enhanced_retriever import (
    KnowledgeGraphEnhancedRetriever,
    augment_graph_with_keywords,
)


def run_kg_enhanced_experiment(
    *,
    documents_path: str | Path,
    queries_path: str | Path,
    graph_path: str | Path,
    output_dir: str | Path = "reports/tables",
    base_method: str = "semantic",
    model_name: str = "all-MiniLM-L6-v2",
    batch_size: int = 32,
    bm25_k1: float = 1.5,
    bm25_b: float = 0.75,
    top_ks: Sequence[int] = (5, 10),
    retrieval_weight: float = 0.75,
    graph_weight: float = 0.25,
    candidate_multiplier: int = 3,
    expansion_weight: float = 0.5,
    keywords_path: str | Path | None = "data/processed/keywords.csv",
    keyword_top_n: int = 10,
    cache_path: str | Path | None = "artifacts/semantic_retrieval/document_embeddings.npz",
    force_recompute: bool = False,
    base_retriever=None,
):
    """Execute KG-enhanced reranking and save ranked results and metrics."""
    top_ks = normalize_top_ks(top_ks)
    method = base_method.lower().strip()
    if method not in {"semantic", "bm25"}:
        raise ValueError("base_method must be either 'semantic' or 'bm25'")

    documents = load_documents(documents_path)
    queries = load_evaluation_queries(queries_path)
    validate_relevance_against_corpus(queries, documents["doc_id"])

    graph_file = Path(graph_path)
    if not graph_file.exists():
        raise FileNotFoundError(
            f"Knowledge graph not found: {graph_file}. Run the pipeline first."
        )
    graph = nx.read_graphml(graph_file)

    if keywords_path is not None and Path(keywords_path).exists():
        with Path(keywords_path).open("r", encoding="utf-8", newline="") as handle:
            keyword_rows = list(csv.DictReader(handle))
        graph = augment_graph_with_keywords(
            graph,
            keyword_rows,
            top_n_per_document=keyword_top_n,
            copy_graph=False,
        )

    doc_ids = documents["doc_id"].astype(str).tolist()
    texts = documents["text"].fillna("").astype(str).tolist()

    if base_retriever is None:
        if method == "semantic":
            base_retriever = EmbeddingRetriever(
                model_name=model_name,
                batch_size=batch_size,
            )
            base_retriever.index(
                doc_ids,
                texts,
                cache_path=cache_path,
                force_recompute=force_recompute,
            )
        elif method == "bm25":
            base_retriever = BM25Retriever(k1=bm25_k1, b=bm25_b)
            base_retriever.index(doc_ids, texts)
    else:
        # Test doubles and externally prepared retrievers may implement index
        # without semantic-cache keyword arguments.
        try:
            base_retriever.index(
                doc_ids,
                texts,
                cache_path=cache_path,
                force_recompute=force_recompute,
            )
        except TypeError:
            base_retriever.index(doc_ids, texts)

    kg_retriever = KnowledgeGraphEnhancedRetriever(
        base_retriever,
        graph,
        retrieval_weight=retrieval_weight,
        graph_weight=graph_weight,
        candidate_multiplier=candidate_multiplier,
        expansion_weight=expansion_weight,
    )

    retrieval_depth = max(top_ks)
    rankings = {
        query.query_id: kg_retriever.search(query.query, top_k=retrieval_depth)
        for query in queries
    }

    results, metrics = evaluate_rankings(
        method=f"kg_enhanced_{method}",
        queries=queries,
        rankings=rankings,
        top_ks=top_ks,
    )
    result_path, metric_path = write_experiment_outputs(
        results,
        metrics,
        output_dir=output_dir,
        result_filename="kg_enhanced_results.csv",
        metric_filename="kg_enhanced_metrics.csv",
    )

    return results, metrics, result_path, metric_path


def _config_defaults(config_path: str | Path) -> dict:
    defaults = {
        "model_name": "all-MiniLM-L6-v2",
        "batch_size": 32,
        "top_k": 10,
        "bm25_k1": 1.5,
        "bm25_b": 0.75,
        "retrieval_weight": 0.75,
        "graph_weight": 0.25,
        "candidate_multiplier": 3,
        "expansion_weight": 0.5,
    }
    path = Path(config_path)
    if not path.exists():
        return defaults

    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}

    retrieval = config.get("retrieval", {})
    embeddings = config.get("embeddings", {})
    kg = config.get("knowledge_graph", {})
    defaults.update(
        {
            "model_name": retrieval.get(
                "embedding_model",
                embeddings.get("model", defaults["model_name"]),
            ),
            "batch_size": embeddings.get("batch_size", defaults["batch_size"]),
            "top_k": retrieval.get("top_k", defaults["top_k"]),
            "bm25_k1": retrieval.get("bm25_k1", defaults["bm25_k1"]),
            "bm25_b": retrieval.get("bm25_b", defaults["bm25_b"]),
            "retrieval_weight": kg.get(
                "retrieval_weight", defaults["retrieval_weight"]
            ),
            "graph_weight": kg.get("graph_weight", defaults["graph_weight"]),
            "candidate_multiplier": kg.get(
                "candidate_multiplier", defaults["candidate_multiplier"]
            ),
            "expansion_weight": kg.get(
                "expansion_weight", defaults["expansion_weight"]
            ),
        }
    )
    return defaults


def parse_args() -> argparse.Namespace:
    defaults = _config_defaults("config.yaml")
    parser = argparse.ArgumentParser(
        description="Evaluate graph-enhanced BM25 or semantic retrieval."
    )
    parser.add_argument(
        "--documents",
        default="data/processed/processed_documents.csv",
    )
    parser.add_argument(
        "--queries",
        default="data/evaluation/retrieval_queries.example.json",
    )
    parser.add_argument(
        "--graph",
        default="data/graphs/knowledge_graph.graphml",
    )
    parser.add_argument(
        "--keywords",
        default="data/processed/keywords.csv",
        help="Optional keyword CSV used to enrich the graph in memory.",
    )
    parser.add_argument("--output-dir", default="reports/tables")
    parser.add_argument(
        "--base-method",
        choices=("semantic", "bm25"),
        default="semantic",
    )
    parser.add_argument("--model", default=defaults["model_name"])
    parser.add_argument("--batch-size", type=int, default=defaults["batch_size"])
    parser.add_argument("--bm25-k1", type=float, default=defaults["bm25_k1"])
    parser.add_argument("--bm25-b", type=float, default=defaults["bm25_b"])
    parser.add_argument(
        "--top-k",
        type=int,
        nargs="+",
        default=sorted({5, int(defaults["top_k"])}),
    )
    parser.add_argument(
        "--retrieval-weight",
        type=float,
        default=defaults["retrieval_weight"],
    )
    parser.add_argument(
        "--graph-weight",
        type=float,
        default=defaults["graph_weight"],
    )
    parser.add_argument(
        "--candidate-multiplier",
        type=int,
        default=defaults["candidate_multiplier"],
    )
    parser.add_argument(
        "--expansion-weight",
        type=float,
        default=defaults["expansion_weight"],
    )
    parser.add_argument("--keyword-top-n", type=int, default=10)
    parser.add_argument(
        "--cache-path",
        default="artifacts/semantic_retrieval/document_embeddings.npz",
    )
    parser.add_argument("--force-recompute", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _, metrics, result_path, metric_path = run_kg_enhanced_experiment(
        documents_path=args.documents,
        queries_path=args.queries,
        graph_path=args.graph,
        output_dir=args.output_dir,
        base_method=args.base_method,
        model_name=args.model,
        batch_size=args.batch_size,
        bm25_k1=args.bm25_k1,
        bm25_b=args.bm25_b,
        top_ks=args.top_k,
        retrieval_weight=args.retrieval_weight,
        graph_weight=args.graph_weight,
        candidate_multiplier=args.candidate_multiplier,
        expansion_weight=args.expansion_weight,
        keywords_path=args.keywords,
        keyword_top_n=args.keyword_top_n,
        cache_path=args.cache_path,
        force_recompute=args.force_recompute,
    )

    print(f"KG-enhanced ranked results: {result_path}")
    print(f"KG-enhanced summary metrics: {metric_path}")
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
