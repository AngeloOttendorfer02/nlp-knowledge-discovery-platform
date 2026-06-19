"""Retrieval-augmented generation utilities for local/Ollama-style LLM calls."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence


@dataclass(frozen=True)
class RetrievedContext:
    """Context snippet passed into a RAG prompt."""

    doc_id: str
    text: str
    score: float


def build_rag_prompt(query: str, contexts: Sequence[RetrievedContext], system_instruction: str | None = None) -> str:
    """Build a grounded answer prompt from retrieved contexts."""
    instruction = system_instruction or "Answer using only the provided scientific document context."
    context_block = "\n\n".join(
        f"[{i}] doc_id={ctx.doc_id} score={ctx.score:.4f}\n{ctx.text}"
        for i, ctx in enumerate(contexts, start=1)
    )
    return f"{instruction}\n\nContext:\n{context_block}\n\nQuestion: {query}\nAnswer:"


class RAGPipeline:
    """Small orchestrator that combines a retriever and a generation function."""

    def __init__(self, retriever, generator: Callable[[str], str]) -> None:
        self.retriever = retriever
        self.generator = generator

    def answer(self, query: str, top_k: int = 5) -> dict:
        results = self.retriever.search(query, top_k=top_k)
        contexts = [RetrievedContext(r.doc_id, getattr(r, "text", ""), float(r.score)) for r in results]
        prompt = build_rag_prompt(query, contexts)
        return {"answer": self.generator(prompt), "contexts": contexts, "prompt": prompt}
