"""Retrieval methods for the NLP Knowledge Discovery Platform."""

from src.retrieval.bm25_retriever import BM25Retriever, RetrievalResult
from src.retrieval.embedding_retriever import EmbeddingRetriever
from src.retrieval.kg_enhanced_retriever import (
    KGEnhancedResult,
    KnowledgeGraphEnhancedRetriever,
    augment_graph_with_keywords,
)

__all__ = [
    "BM25Retriever",
    "EmbeddingRetriever",
    "KGEnhancedResult",
    "KnowledgeGraphEnhancedRetriever",
    "RetrievalResult",
    "augment_graph_with_keywords",
]
