"""Retrieval completo per una domanda: candidati ibridi (dense + sparse + codici) e reranker."""

from __future__ import annotations

from src.vectorstore import (
    DEFAULT_CANDIDATES,
    BGEM3Embedder,
    QdrantVectorStore,
    Reranker,
    SearchResult,
    get_reranker,
)

DEFAULT_CONTEXT_CHUNKS: int = 6  # chunk passati all'LLM


def retrieve(
    question: str,
    store: QdrantVectorStore,
    *,
    embedder: BGEM3Embedder | None = None,
    reranker: Reranker | None = None,
    doc_id: str | None = None,
    top_k: int = DEFAULT_CONTEXT_CHUNKS,
    candidates: int = DEFAULT_CANDIDATES,
) -> list[SearchResult]:
    """Restituisce i `top_k` chunk più pertinenti alla domanda, dal migliore al peggiore."""
    pool = store.hybrid_candidates(question, embedder=embedder, limit=candidates, doc_id=doc_id)
    return (reranker or get_reranker()).rerank(question, pool, top_k=top_k)
