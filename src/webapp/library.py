"""Documenti della web app: file su disco e metadati.

Ogni documento caricato ha il PDF in `data/raw/<doc_id>/source.pdf` e i risultati in
`data/processed/web/<doc_id>/` (con `meta.json`). Sono riconosciuti anche i documenti indicizzati
dagli script, che hanno `timing.json` in `data/processed/**/<doc_id>/`.
"""

from __future__ import annotations

import json
import re
import shutil
import threading
from datetime import datetime
from pathlib import Path
from typing import Any

_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,119}")


def make_doc_id(filename: str, now: datetime | None = None) -> str:
    """Identificativo leggibile e unico, es. `relazione-2024-20261124-130501-123456`."""
    slug = re.sub(r"[^a-z0-9]+", "-", Path(filename).stem.lower()).strip("-")[:60] or "documento"
    return f"{slug}-{(now or datetime.now()).strftime('%Y%m%d-%H%M%S-%f')}"


class Library:
    def __init__(self, data_dir: Path) -> None:
        self.data_dir = data_dir.resolve()
        self.raw_dir = self.data_dir / "raw"
        self.processed_dir = self.data_dir / "processed"
        self.web_dir = self.processed_dir / "web"
        self._lock = threading.RLock()  # meta.json si legge e si scrive da thread diversi

    # ------------------------------------------------------------------ percorsi
    def source_path(self, doc_id: str) -> Path:
        return self.raw_dir / self._check(doc_id) / "source.pdf"

    def output_dir(self, doc_id: str) -> Path:
        return self.web_dir / self._check(doc_id)

    def find_output_dir(self, doc_id: str) -> Path | None:
        """Cartella dei risultati, anche per i documenti creati dagli script."""
        if not _ID_RE.fullmatch(doc_id):
            return None
        for base in (self.web_dir, self.processed_dir / "docling", self.processed_dir):
            if (base / doc_id).is_dir():
                return base / doc_id
        return None

    def find_pdf(self, doc_id: str) -> Path | None:
        """PDF originale, solo se si trova dentro la cartella dei dati."""
        out = self.find_output_dir(doc_id)
        candidates: list[Path] = []
        if out is not None:
            recorded = self._read(out / "timing.json").get("pdf")
            if recorded:
                candidates.append(self.data_dir.parent / str(recorded).replace("\\", "/"))
        if _ID_RE.fullmatch(doc_id):
            candidates.append(self.raw_dir / doc_id / "source.pdf")
        for path in candidates:
            path = path.resolve()
            if path.is_file() and path.suffix.lower() == ".pdf" and self._inside(path):
                return path
        return None

    def find_image(self, doc_id: str, name: str) -> Path | None:
        """Ritaglio PNG di una tabella."""
        out = self.find_output_dir(doc_id)
        if out is None or not _ID_RE.fullmatch(name) or not name.endswith(".png"):
            return None
        path = (out / "table_images" / name).resolve()
        return path if path.is_file() and self._inside(path) else None

    # ------------------------------------------------------------------ eliminazione
    def is_managed(self, doc_id: str) -> bool:
        """True per i documenti caricati dall'app, False per quelli creati dagli script."""
        return bool(_ID_RE.fullmatch(doc_id)) and (self.web_dir / doc_id / "meta.json").is_file()

    def remove_files(self, doc_id: str) -> None:
        """Cancella PDF e risultati di un documento caricato dall'app.

        I metadati si tolgono per ultimi: se un file è in uso (`OSError`) il documento resta
        nell'elenco e si può riprovare.
        """
        if not self.is_managed(doc_id):
            raise ValueError(f"Documento non caricato dall'app: {doc_id!r}")
        raw, out = self.source_path(doc_id).parent, self.output_dir(doc_id)
        if raw.is_dir():
            shutil.rmtree(raw)
        with self._lock:
            for child in out.iterdir():
                if child.is_dir():
                    shutil.rmtree(child)
                elif child.name != "meta.json":
                    child.unlink()
            shutil.rmtree(out)

    # ------------------------------------------------------------------ metadati
    def write_meta(self, doc_id: str, **fields: Any) -> None:
        out = self.output_dir(doc_id)
        with self._lock:
            out.mkdir(parents=True, exist_ok=True)
            meta = {**self._read(out / "meta.json"), **fields}
            (out / "meta.json").write_text(
                json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
            )

    def mark_interrupted(self) -> None:
        """Segna come falliti i caricamenti rimasti a metà alla chiusura precedente."""
        for path in self.web_dir.glob("*/meta.json"):
            if self._read(path).get("status") == "processing":
                self.write_meta(
                    path.parent.name, status="error", message="Elaborazione interrotta: ricarica il PDF."
                )

    def list(self, indexed: dict[str, int]) -> list[dict[str, Any]]:
        """Documenti consultabili o in elaborazione, dal più recente.

        `indexed`: numero di chunk su Qdrant per ogni `doc_id`.
        """
        docs: dict[str, dict[str, Any]] = {}
        with self._lock:
            metas = {p.parent.name: self._read(p) for p in self.web_dir.glob("*/meta.json")}
        for doc_id, meta in metas.items():
            docs[doc_id] = {
                "id": doc_id,
                "title": meta.get("title") or doc_id,
                "status": meta.get("status", "error"),
                "message": meta.get("message", ""),
                "pages": meta.get("pages"),
                "created": meta.get("created", ""),
            }
        for doc_id in indexed:
            if doc_id not in docs:
                out = self.find_output_dir(doc_id)
                timing = self._read(out / "timing.json") if out is not None else {}
                docs[doc_id] = {
                    "id": doc_id,
                    "title": doc_id,
                    "status": "ready",
                    "message": "",
                    "pages": timing.get("pages"),
                    "created": "",
                }
        for doc_id, entry in docs.items():
            entry["chunks"] = indexed.get(doc_id, 0)
            entry["has_pdf"] = self.find_pdf(doc_id) is not None
            entry["managed"] = self.is_managed(doc_id)
            if entry["status"] == "ready" and not entry["chunks"]:
                entry.update(status="error", message="Documento non presente nell'indice.")
        return sorted(docs.values(), key=lambda d: (d["created"], d["title"]), reverse=True)

    # ------------------------------------------------------------------ interni
    def _inside(self, path: Path) -> bool:
        return path.is_relative_to(self.data_dir)

    @staticmethod
    def _check(doc_id: str) -> str:
        if not _ID_RE.fullmatch(doc_id):
            raise ValueError(f"Identificativo non valido: {doc_id!r}")
        return doc_id

    @staticmethod
    def _read(path: Path) -> dict[str, Any]:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}
        return data if isinstance(data, dict) else {}
