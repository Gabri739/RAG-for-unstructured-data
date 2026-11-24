"""Avanzamento dell'elaborazione di un documento, fase per fase.

Le percentuali sono reali (pagine lette, estratti calcolati, estratti scritti). Quando un passaggio
non è misurabile la percentuale è `None`: l'interfaccia mostra che è in corso, senza inventarla.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Any

PHASES: tuple[tuple[str, str], ...] = (
    ("estrazione", "Estrazione"),
    ("tabelle", "Tabelle"),
    ("chunking", "Chunking"),
    ("embeddings", "Embeddings"),
    ("vectordb", "VectorDB"),
)


@dataclass
class _Phase:
    id: str
    name: str
    state: str = "waiting"  # waiting | active | done | error
    percent: int | None = 0
    detail: str = ""


class IngestProgress:
    """Stato condiviso tra il thread che elabora e le richieste che lo leggono."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._phases = {id_: _Phase(id_, name) for id_, name in PHASES}
        self._current: _Phase | None = None
        self._action = "In coda"
        self._log: list[dict[str, Any]] = []
        self._start = time.monotonic()
        self._end: float | None = None

    def start(self, phase: str, action: str) -> None:
        """Inizia una fase; quella precedente risulta completata."""
        with self._lock:
            self._complete_current()
            self._current = self._phases[phase]
            self._current.state = "active"
            self._current.percent = None
            self._action = action

    def action(self, text: str) -> None:
        """Cambia l'azione in corso, dentro la stessa fase."""
        with self._lock:
            self._action = text

    def update(self, done: int, total: int, unit: str) -> None:
        """Avanzamento misurato della fase in corso, es. `update(45, 109, "pagine")`."""
        with self._lock:
            if self._current is not None and total > 0:
                self._current.percent = min(100, int(100 * done / total))
                self._current.detail = f"{done} di {total} {unit}"

    def step_done(self, text: str) -> None:
        """Aggiunge un passaggio concluso all'elenco."""
        with self._lock:
            phase = self._current.id if self._current is not None else ""
            self._log.append({"phase": phase, "text": text, "seconds": self._elapsed()})

    def finish(self) -> None:
        with self._lock:
            self._complete_current()
            self._current = None
            self._action = "Documento pronto"
            self._end = time.monotonic()

    def fail(self, message: str) -> None:
        """Segna come fallita la fase in corso; quelle concluse restano tali."""
        with self._lock:
            if self._current is not None:
                self._current.state = "error"
            self._action = message
            self._end = time.monotonic()

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "action": self._action,
                "seconds": self._elapsed(),
                "phases": [
                    {
                        "id": p.id,
                        "name": p.name,
                        "state": p.state,
                        "percent": p.percent,
                        "detail": p.detail,
                    }
                    for p in self._phases.values()
                ],
                "log": [dict(entry) for entry in self._log],
            }

    def _complete_current(self) -> None:
        if self._current is not None:
            self._current.state = "done"
            self._current.percent = 100

    def _elapsed(self) -> float:
        return round((self._end or time.monotonic()) - self._start, 1)
