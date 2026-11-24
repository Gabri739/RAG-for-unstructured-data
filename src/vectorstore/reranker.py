"""Reranking dei candidati con `Qwen/Qwen3-Reranker-0.6B` (cross-encoder di `sentence-transformers`).

Il reranker legge insieme domanda e chunk e assegna un punteggio di pertinenza. Il punteggio
serve solo a ORDINARE i candidati di una stessa domanda: non è una probabilità e non separa in
modo affidabile le domande con risposta da quelle fuori perimetro, quindi non si usa come soglia.
"""

from __future__ import annotations

from dataclasses import replace
from functools import lru_cache
from typing import Sequence

import torch

from .qdrant_store import SearchResult

RERANKER_MODEL_NAME: str = "Qwen/Qwen3-Reranker-0.6B"
DEFAULT_RERANK_MAX_LENGTH: int = 1024  # domanda + chunk, in token


class Reranker:
    """Riordina i `SearchResult` per pertinenza rispetto alla domanda."""

    def __init__(
        self,
        model_name: str = RERANKER_MODEL_NAME,
        *,
        device: str | None = None,
        max_length: int = DEFAULT_RERANK_MAX_LENGTH,
        batch_size: int = 8,
    ) -> None:
        from sentence_transformers import CrossEncoder

        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        dtype = torch.float16 if self.device.startswith("cuda") else torch.float32
        self.model = CrossEncoder(
            model_name,
            device=self.device,
            max_length=max_length,
            model_kwargs={"dtype": dtype},
        )

    def rerank(
        self,
        query: str,
        hits: Sequence[SearchResult],
        *,
        top_k: int | None = None,
    ) -> list[SearchResult]:
        """Restituisce i candidati dal più al meno pertinente (solo i primi `top_k`, se indicato).

        Nei risultati `score` è il punteggio del reranker; gli oggetti in input non cambiano.
        """
        if not hits:
            return []
        scores = self.model.predict(
            [(query, h.content) for h in hits],
            batch_size=self.batch_size,
            show_progress_bar=False,
        )
        rescored = [replace(h, score=round(float(s), 4)) for h, s in zip(hits, scores)]
        rescored.sort(key=lambda h: h.score, reverse=True)
        return rescored[:top_k]


@lru_cache(maxsize=1)
def get_reranker(
    model_name: str = RERANKER_MODEL_NAME,
    max_length: int = DEFAULT_RERANK_MAX_LENGTH,
) -> Reranker:
    """Carica e mette in cache una singola istanza di `Reranker`."""
    return Reranker(model_name=model_name, max_length=max_length)
