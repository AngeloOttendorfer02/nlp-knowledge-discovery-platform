"""Compare BM25, semantic, and KG-enhanced retrieval under one protocol."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence

import networkx as nx
import pandas as pd
import yaml

from src.evaluation.retrieval_metrics import (
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)
from src.experiments.common import (
    EvaluationQuery,
    evaluate_rankings,
    load_documents,
    load_evaluation_queries,
    normalize_top_ks,
    validate_relevance_against_corpus,
)
from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.embedding_retriever import EmbeddingRetriever
from src.retrieval.kg_enhanced_retriever import (
    KnowledgeGraphEnhancedRetriever,
    augment_graph_with_keywords,
)


DEFAULT_METHODS = ("bm25", "semantic", "kg_enhanced")


def run_retrieval_comparison(
    *,
    documents_path: str | Path,
    queries_path: str | Path,
    graph_path: str | Path = "data/graphs/knowledge_graph.graphml",
    output_dir: str | Path = "reports/tables",
    figure_dir: str | Path = "reports/figures",
    methods: Sequence[str] = DEFAULT_METHODS,
    top_ks: Sequence[int] = (5, 10),
    model_name: str = "all-MiniLM-L6-v2",
    batch_size: int = 32,
    bm25_k1: float = 1.5,
    bm25_b: float = 0.75,
    retrieval_weight: float = 0.75,
    graph_weight: float = 0.25,
    candidate_multiplier: int = 3,
    expansion_weight: float = 0.5,
    keywords_path: str | Path | None = "data/processed/keywords.csv",
    keyword_top_n: int = 10,
    cache_path: str | Path | None = "artifacts/semantic_retrieval/document_embeddings.npz",
    force_recompute: bool = False,
    semantic_retriever=None,
    kg_base_retriever=None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Path]:
    """Run all requested retrieval methods and write comparison artifacts."""
    selected_methods = _normalize_methods(methods)
    top_ks = normalize_top_ks(top_ks)

    documents = load_documents(documents_path)
    queries = load_evaluation_queries(queries_path)
    validate_relevance_against_corpus(queries, documents["doc_id"])

    doc_ids = documents["doc_id"].astype(str).tolist()
    texts = documents["text"].fillna("").astype(str).tolist()
    retrieval_depth = max(top_ks)

    rankings_by_method: Dict[str, Mapping[str, Sequence[object]]] = {}

    if "bm25" in selected_methods:
        bm25 = BM25Retriever(k1=bm25_k1, b=bm25_b)
        bm25.index(doc_ids, texts)
        rankings_by_method["bm25"] = _search_all(bm25, queries, retrieval_depth)

    if "semantic" in selected_methods:
        semantic = semantic_retriever or EmbeddingRetriever(
            model_name=model_name,
            batch_size=batch_size,
        )
        _index_retriever(
            semantic,
            doc_ids,
            texts,
            cache_path=cache_path,
            force_recompute=force_recompute,
        )
        rankings_by_method["semantic"] = _search_all(semantic, queries, retrieval_depth)

    if "kg_enhanced" in selected_methods:
        graph = _load_augmented_graph(
            graph_path=graph_path,
            keywords_path=keywords_path,
            keyword_top_n=keyword_top_n,
        )
        base_retriever = kg_base_retriever
        if base_retriever is None:
            base_retriever = EmbeddingRetriever(
                model_name=model_name,
                batch_size=batch_size,
            )
        _index_retriever(
            base_retriever,
            doc_ids,
            texts,
            cache_path=cache_path,
            force_recompute=force_recompute,
        )
        kg_retriever = KnowledgeGraphEnhancedRetriever(
            base_retriever,
            graph,
            retrieval_weight=retrieval_weight,
            graph_weight=graph_weight,
            candidate_multiplier=candidate_multiplier,
            expansion_weight=expansion_weight,
        )
        rankings_by_method["kg_enhanced"] = _search_all(
            kg_retriever,
            queries,
            retrieval_depth,
        )

    ranked_results = []
    summary_metrics = []
    for method in selected_methods:
        results, metrics = evaluate_rankings(
            method=method,
            queries=queries,
            rankings=rankings_by_method[method],
            top_ks=top_ks,
        )
        ranked_results.append(results)
        summary_metrics.append(metrics)

    ranked_results_df = pd.concat(ranked_results, ignore_index=True)
    comparison_df = pd.concat(summary_metrics, ignore_index=True)
    per_query_df = build_per_query_comparison(
        queries=queries,
        rankings_by_method=rankings_by_method,
        top_ks=top_ks,
    )

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    ranked_results_df.to_csv(output_path / "all_retrieval_results.csv", index=False)
    comparison_df.to_csv(output_path / "retrieval_comparison.csv", index=False)
    per_query_df.to_csv(output_path / "per_query_comparison.csv", index=False)

    figure_path = Path(figure_dir) / "retrieval_metrics_comparison.png"
    plot_metric_comparison(comparison_df, figure_path)

    return ranked_results_df, comparison_df, per_query_df, figure_path


def build_per_query_comparison(
    *,
    queries: Sequence[EvaluationQuery],
    rankings_by_method: Mapping[str, Mapping[str, Sequence[object]]],
    top_ks: Sequence[int],
) -> pd.DataFrame:
    """Create per-query metric rows for qualitative error analysis."""
    top_ks = normalize_top_ks(top_ks)
    rows: List[dict] = []

    for method, rankings in rankings_by_method.items():
        for query in queries:
            hits = list(rankings.get(query.query_id, []))
            retrieved_ids = [str(hit.doc_id) for hit in hits]
            rr = reciprocal_rank(retrieved_ids, query.relevant_doc_ids)
            first_relevant_rank = int(round(1.0 / rr)) if rr > 0.0 else None
            top_hit = hits[0] if hits else None

            for k in top_ks:
                rows.append(
                    {
                        "method": method,
                        "query_id": query.query_id,
                        "query": query.query,
                        "k": k,
                        "precision_at_k": precision_at_k(
                            retrieved_ids,
                            query.relevant_doc_ids,
                            k,
                        ),
                        "recall_at_k": recall_at_k(
                            retrieved_ids,
                            query.relevant_doc_ids,
                            k,
                        ),
                        "reciprocal_rank": rr,
                        "first_relevant_rank": first_relevant_rank,
                        "top_doc_id": str(top_hit.doc_id) if top_hit else "",
                        "top_score": float(top_hit.score) if top_hit else 0.0,
                        "top_doc_relevant": (
                            str(top_hit.doc_id) in query.relevant_doc_ids
                            if top_hit
                            else False
                        ),
                    }
                )

    return pd.DataFrame(rows)


def plot_metric_comparison(metrics: pd.DataFrame, output_path: str | Path) -> Path:
    """Save a compact comparison plot for Precision@K, Recall@K, and MRR."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
    metric_specs = [
        ("precision_at_k", "Precision@K"),
        ("recall_at_k", "Recall@K"),
        ("mrr", "MRR"),
    ]

    for axis, (column, title) in zip(axes, metric_specs):
        pivot = metrics.pivot(index="k", columns="method", values=column).sort_index()
        pivot.plot(kind="bar", ax=axis)
        axis.set_title(title)
        axis.set_xlabel("K")
        axis.set_ylabel("Score")
        axis.set_ylim(0.0, 1.0)
        axis.legend(title="Method", fontsize=8)

    fig.savefig(target, dpi=160)
    plt.close(fig)
    return target


def _search_all(retriever, queries: Sequence[EvaluationQuery], top_k: int):
    return {
        query.query_id: retriever.search(query.query, top_k=top_k)
        for query in queries
    }


def _index_retriever(
    retriever,
    doc_ids: Sequence[str],
    texts: Sequence[str],
    *,
    cache_path: str | Path | None,
    force_recompute: bool,
) -> None:
    try:
        retriever.index(
            doc_ids,
            texts,
            cache_path=cache_path,
            force_recompute=force_recompute,
        )
    except TypeError:
        retriever.index(doc_ids, texts)


def _load_augmented_graph(
    *,
    graph_path: str | Path,
    keywords_path: str | Path | None,
    keyword_top_n: int,
):
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

    return graph


def _normalize_methods(methods: Iterable[str]) -> List[str]:
    aliases = {
        "bm25": "bm25",
        "semantic": "semantic",
        "kg": "kg_enhanced",
        "kg_enhanced": "kg_enhanced",
        "kg-enhanced": "kg_enhanced",
    }
    selected = []
    for method in methods:
        key = str(method).lower().strip()
        if key not in aliases:
            raise ValueError(
                "Unknown method. Choose from: bm25, semantic, kg_enhanced"
            )
        normalized = aliases[key]
        if normalized not in selected:
            selected.append(normalized)
    if not selected:
        raise ValueError("At least one retrieval method is required")
    return selected


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
        description="Compare BM25, semantic, and KG-enhanced retrieval."
    )
    parser.add_argument(
        "--documents",
        default="data/processed/processed_documents.csv",
    )
    parser.add_argument(
        "--queries",
        default="data/evaluation/retrieval_queries.example.json",
    )
    parser.add_argument("--graph", default="data/graphs/knowledge_graph.graphml")
    parser.add_argument("--keywords", default="data/processed/keywords.csv")
    parser.add_argument("--output-dir", default="reports/tables")
    parser.add_argument("--figure-dir", default="reports/figures")
    parser.add_argument(
        "--methods",
        nargs="+",
        default=list(DEFAULT_METHODS),
        help="Methods to compare: bm25 semantic kg_enhanced",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        nargs="+",
        default=sorted({5, int(defaults["top_k"])}),
    )
    parser.add_argument("--model", default=defaults["model_name"])
    parser.add_argument("--batch-size", type=int, default=defaults["batch_size"])
    parser.add_argument("--bm25-k1", type=float, default=defaults["bm25_k1"])
    parser.add_argument("--bm25-b", type=float, default=defaults["bm25_b"])
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
    _, comparison, per_query, figure_path = run_retrieval_comparison(
        documents_path=args.documents,
        queries_path=args.queries,
        graph_path=args.graph,
        output_dir=args.output_dir,
        figure_dir=args.figure_dir,
        methods=args.methods,
        top_ks=args.top_k,
        model_name=args.model,
        batch_size=args.batch_size,
        bm25_k1=args.bm25_k1,
        bm25_b=args.bm25_b,
        retrieval_weight=args.retrieval_weight,
        graph_weight=args.graph_weight,
        candidate_multiplier=args.candidate_multiplier,
        expansion_weight=args.expansion_weight,
        keywords_path=args.keywords,
        keyword_top_n=args.keyword_top_n,
        cache_path=args.cache_path,
        force_recompute=args.force_recompute,
    )

    print("Retrieval comparison written to reports/tables/retrieval_comparison.csv")
    print("Per-query comparison written to reports/tables/per_query_comparison.csv")
    print(f"Comparison figure written to {figure_path}")
    print(comparison.to_string(index=False))
    print(f"Per-query rows: {len(per_query)}")


if __name__ == "__main__":
    main()
