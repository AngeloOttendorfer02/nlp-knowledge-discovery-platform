"""
Run a reproducible semantic retrieval experiment.

This experiment uses the same processed documents, relevance labels, and top_k
setting as the BM25 baseline, but replaces lexical BM25 retrieval with dense
embedding-based retrieval using the existing EmbeddingRetriever.

Example:
    python -m src.experiments.run_semantic_retrieval
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from src.evaluation.retrieval_metrics import evaluate_retrieval_run
from src.retrieval.embedding_retriever import EmbeddingRetriever
from src.retrieval.vector_store import VectorStore

try:
    from src.experiments.run_bm25_baseline import (
        load_processed_documents,
        load_relevance_queries,
    )
except ImportError:  # pragma: no cover
    load_processed_documents = None
    load_relevance_queries = None


def dataset_fingerprint(doc_ids: List[str], texts: List[str], model_name: str) -> str:
    """
    Create a stable fingerprint for a document set and embedding model.

    This is used to cache embeddings and avoid recomputing them unnecessarily.
    """
    hasher = hashlib.sha256()
    hasher.update(model_name.encode("utf-8"))

    for doc_id, text in zip(doc_ids, texts):
        hasher.update(str(doc_id).encode("utf-8"))
        hasher.update(b"\0")
        hasher.update(str(text).encode("utf-8"))
        hasher.update(b"\0")

    return hasher.hexdigest()[:16]


def load_cached_embeddings(cache_dir: Path, fingerprint: str) -> Optional[np.ndarray]:
    path = cache_dir / f"embeddings_{fingerprint}.npy"

    if path.exists():
        return np.load(path)

    return None


def save_cached_embeddings(cache_dir: Path, fingerprint: str, embeddings: np.ndarray) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)

    path = cache_dir / f"embeddings_{fingerprint}.npy"
    np.save(path, embeddings)

    return path


def index_retriever_with_embeddings(
    retriever: EmbeddingRetriever,
    doc_ids: List[str],
    texts: List[str],
    embeddings: np.ndarray,
) -> None:
    """
    Build the retriever's vector store from precomputed embeddings.

    This lets the experiment cache document embeddings while still using the
    existing EmbeddingRetriever for query encoding and search.
    """
    store = VectorStore(dim=embeddings.shape[1])
    store.add(doc_ids, embeddings)

    retriever._store = store
    retriever._doc_texts = dict(zip(doc_ids, texts))


def run_semantic_retrieval(
    documents_path: Path,
    queries_path: Path,
    results_path: Path,
    metrics_path: Path,
    cache_dir: Path,
    top_k: int = 5,
    model_name: str = "all-MiniLM-L6-v2",
) -> Dict[str, float]:
    if load_processed_documents is None or load_relevance_queries is None:
        raise ImportError("BM25 experiment helpers could not be imported.")

    df = load_processed_documents(documents_path)
    queries = load_relevance_queries(queries_path)

    doc_ids = df["doc_id"].astype(str).tolist()
    texts = df["text"].fillna("").astype(str).tolist()

    retriever = EmbeddingRetriever(model_name=model_name)

    fingerprint = dataset_fingerprint(doc_ids, texts, model_name)
    cached_embeddings = load_cached_embeddings(cache_dir, fingerprint)

    if cached_embeddings is None:
        embeddings = retriever.encode(texts)
        save_cached_embeddings(cache_dir, fingerprint, embeddings)
    else:
        embeddings = cached_embeddings

    index_retriever_with_embeddings(
        retriever=retriever,
        doc_ids=doc_ids,
        texts=texts,
        embeddings=embeddings,
    )

    results_by_query: Dict[str, List[str]] = {}
    relevance_sets: Dict[str, List[str]] = {}
    result_rows: List[Dict[str, Any]] = []

    for query_item in queries:
        query_id = str(query_item["query_id"])
        query_text = str(query_item["query"])
        relevant_doc_ids = [str(doc_id) for doc_id in query_item["relevant_doc_ids"]]
        relevant_set = set(relevant_doc_ids)

        hits = retriever.search(query_text, top_k=top_k)
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
    pd.DataFrame([{"method": "Semantic", "model": model_name, **metrics}]).to_csv(
        metrics_path,
        index=False,
    )

    print("Semantic retrieval experiment completed.")
    print(f"Results saved to: {results_path}")
    print(f"Metrics saved to: {metrics_path}")
    print(metrics)

    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run semantic retrieval experiment.")

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
        "--results",
        type=Path,
        default=Path("reports/tables/semantic_results.csv"),
    )
    parser.add_argument(
        "--metrics",
        type=Path,
        default=Path("reports/tables/semantic_metrics.csv"),
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("data/processed/embedding_cache"),
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="all-MiniLM-L6-v2",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    run_semantic_retrieval(
        documents_path=args.documents,
        queries_path=args.queries,
        results_path=args.results,
        metrics_path=args.metrics,
        cache_dir=args.cache_dir,
        top_k=args.top_k,
        model_name=args.model_name,
    )


if __name__ == "__main__":
    main()