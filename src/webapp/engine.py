"""Motore della web app: modelli caricati una sola volta, domande e caricamenti in sequenza.

Qdrant su disco locale e la GPU non si possono usare da più thread insieme: ogni operazione passa
da un unico lock. Durante un caricamento le domande vengono rifiutate subito con `EngineBusy`.
"""

from __future__ import annotations

import gc
import logging
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from src.rag import Answer, OllamaGenerator, retrieve
from src.rag.progress import IngestProgress
from src.vectorstore import QdrantVectorStore, get_embedder, get_reranker

_log = logging.getLogger(__name__)


class EngineBusy(RuntimeError):
    """Un documento è in elaborazione, o un'altra domanda sta occupando la GPU."""


class EngineNotReady(RuntimeError):
    """I modelli non sono ancora caricati, o il caricamento è fallito."""


@dataclass
class Reply:
    answer: Answer
    retrieval_s: float
    generation_s: float


class RagEngine:
    def __init__(self, qdrant_path: Path, *, generator: OllamaGenerator | None = None) -> None:
        self.qdrant_path = qdrant_path
        self.generator = generator or OllamaGenerator()
        self.state = "loading"  # loading | ready | error
        self.error = ""
        self._lock = threading.Lock()
        self._ingesting = threading.Event()
        self._counts: dict[str, int] = {}

    def load(self) -> None:
        """Apre l'indice e carica embedder e reranker (da eseguire in un thread)."""
        try:
            with self._lock:
                self._store = QdrantVectorStore(path=self.qdrant_path)
                self._counts = self._store.document_counts()
                self._embedder = get_embedder()
                self._reranker = get_reranker()
            self.state = "ready"
        except Exception as exc:
            _log.exception("Caricamento dei modelli fallito")
            in_use = "already accessed" in str(exc)
            self.error = (
                "L'indice è aperto da un altro programma: chiudi gli altri script e riavvia."
                if in_use
                else f"Caricamento dei modelli fallito: {exc}"
            )
            self.state = "error"

    def close(self) -> None:
        store = getattr(self, "_store", None)
        if store is not None:
            store.close()

    @property
    def ingesting(self) -> bool:
        return self._ingesting.is_set()

    def document_counts(self) -> dict[str, int]:
        return dict(self._counts)

    def ask(self, question: str, doc_id: str) -> Reply:
        self._require_ready()
        if self.ingesting:
            raise EngineBusy("Sto elaborando un documento: riprova tra poco.")
        if not self._lock.acquire(timeout=60):
            raise EngineBusy("Sto rispondendo a un'altra domanda: riprova tra poco.")
        try:
            t0 = time.perf_counter()
            hits = retrieve(
                question,
                self._store,
                embedder=self._embedder,
                reranker=self._reranker,
                doc_id=doc_id,
            )
            t1 = time.perf_counter()
            answer = self.generator.answer(question, hits)
            return Reply(answer, round(t1 - t0, 2), round(time.perf_counter() - t1, 2))
        finally:
            self._lock.release()

    def delete(self, doc_id: str) -> None:
        """Toglie dall'indice tutti gli estratti di un documento."""
        if doc_id not in self._counts:
            return
        self._require_ready()
        if self.ingesting:
            raise EngineBusy("Sto elaborando un documento: riprova tra poco.")
        if not self._lock.acquire(timeout=60):
            raise EngineBusy("Sto rispondendo a una domanda: riprova tra poco.")
        try:
            self._store.delete_document(doc_id)
            self._counts = self._store.document_counts()
        finally:
            self._lock.release()

    def reserve_ingest(self) -> None:
        """Prenota l'elaborazione di un documento; va seguita da `ingest` o `cancel_ingest`."""
        self._require_ready()
        if self.ingesting:
            raise EngineBusy("Un altro documento è già in elaborazione.")
        self._ingesting.set()

    def cancel_ingest(self) -> None:
        self._ingesting.clear()

    def ingest(
        self,
        pdf_path: Path,
        out_dir: Path,
        *,
        doc_id: str,
        reference_codes: Sequence[str],
        progress: IngestProgress,
    ):
        """Elabora un PDF già prenotato con `reserve_ingest` (da eseguire in un thread)."""
        from src.extractors.docling_parser import get_converter
        from src.rag.ingest import ingest_pdf

        try:
            progress.start("estrazione", "Preparo la GPU")
            with self._lock:
                # Docling, embedder, reranker e LLM insieme non stanno in 8 GB
                self.generator.unload()
                try:
                    return ingest_pdf(
                        pdf_path,
                        out_dir,
                        doc_id=doc_id,
                        store=self._store,
                        embedder=self._embedder,
                        reference_codes=reference_codes,
                        progress=progress,
                    )
                finally:
                    get_converter.cache_clear()
                    gc.collect()
                    _empty_gpu_cache()
                    self._counts = self._store.document_counts()
        finally:
            self._ingesting.clear()

    def _require_ready(self) -> None:
        if self.state == "error":
            raise EngineNotReady(self.error)
        if self.state != "ready":
            raise EngineNotReady("I modelli si stanno caricando: riprova tra poco.")


def _empty_gpu_cache() -> None:
    import torch

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
