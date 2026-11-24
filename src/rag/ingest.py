"""Ingestione di un PDF qualsiasi: parsing, fusione tabelle, chunking e indicizzazione su Qdrant.

Usa gli stessi passaggi e parametri di `scripts/run_full_pipeline.py`, senza gli artefatti di analisi.
"""

from __future__ import annotations

import dataclasses
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from src.chunkers import REFERENCE_PATTERNS, MarkdownChunker
from src.extractors.docling_parser import convert_pdf
from src.extractors.table_fusion import fuse_document_tables
from src.extractors.table_images import MARKDOWN_BLOCKED_META, save_table_images
from src.vectorstore import BGEM3Embedder, QdrantVectorStore, get_token_counter

from .progress import IngestProgress

EMBED_BATCH: int = 16


class EmptyDocumentError(ValueError):
    """Dal PDF non è stato estratto alcun contenuto indicizzabile."""


@dataclass
class IngestReport:
    doc_id: str
    pages: int
    tables: int
    chunks: int
    ocr_pages: list[int]
    seconds: float


def _count(n: int, singular: str, plural: str) -> str:
    return f"{n} {singular if n == 1 else plural}"


def ingest_pdf(
    pdf_path: Path,
    out_dir: Path,
    *,
    doc_id: str,
    store: QdrantVectorStore,
    embedder: BGEM3Embedder,
    reference_codes: Sequence[str] = (),
    max_table_rows: int = 25,
    progress: IngestProgress | None = None,
) -> IngestReport:
    """Elabora `pdf_path` e ne indicizza i chunk con `doc_id`, sostituendo quelli esistenti.

    `reference_codes`: nomi dei codici da riconoscere (`REFERENCE_PATTERNS`, es. `"gri"`).
    `progress`: riceve l'avanzamento di ogni fase; in caso di errore lo chiude chi chiama.
    """
    t0 = time.perf_counter()
    progress = progress or IngestProgress()
    out_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------ estrazione
    progress.start("estrazione", "Controllo se ci sono pagine scansionate")
    failed_pages: list[int] = []

    def on_docling(event: Any) -> None:
        kind = event.kind.value
        if kind == "phase_started":
            phase = event.phase.value
            if phase == "initialize":
                progress.action("Carico i modelli di lettura")
            elif phase == "build":
                progress.step_done("Modelli di lettura pronti")
                progress.action("Leggo le pagine e riconosco testo e tabelle")
            elif phase == "assemble":
                progress.action("Ricompongo il documento")
        elif kind == "page_completed":
            if not event.success:
                failed_pages.append(event.page_no)
            progress.update(event.completed_pages, event.total_pages, "pagine")

    doc, ocr_pages = convert_pdf(pdf_path, progress_callback=on_docling)
    n_pages = len(doc.pages)
    progress.step_done(
        f"{_count(n_pages, 'pagina letta', 'pagine lette')}, "
        f"{_count(len(doc.tables), 'tabella trovata', 'tabelle trovate')}"
    )
    if ocr_pages:
        progress.step_done(f"Testo riconosciuto con OCR: {_count(len(ocr_pages), 'pagina', 'pagine')}")
    if failed_pages:
        progress.step_done(f"Pagine non lette: {', '.join(map(str, sorted(failed_pages)))}")

    # ------------------------------------------------------------ tabelle
    progress.start("tabelle", "Unisco le tabelle che continuano su più pagine")
    fusion = fuse_document_tables(doc)
    merged = len(fusion.merged_chains)
    progress.step_done(
        f"{_count(merged, 'tabella riunita', 'tabelle riunite')} su più pagine"
        if merged
        else "Nessuna tabella da riunire"
    )
    if fusion.empty_tables_removed:
        n_empty = len(fusion.empty_tables_removed)
        progress.step_done(_count(n_empty, "tabella vuota rimossa", "tabelle vuote rimosse"))
    progress.action("Salvo le immagini delle tabelle")
    images = save_table_images(doc, out_dir / "table_images", relative_to=out_dir)
    manifest = out_dir / "table_images.json"
    manifest.write_text(
        json.dumps(dataclasses.asdict(images), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    progress.step_done(_count(images.image_count, "immagine salvata", "immagini salvate"))
    (out_dir / "source_fused.md").write_text(
        doc.export_to_markdown(blocked_meta_names=MARKDOWN_BLOCKED_META), encoding="utf-8"
    )

    # ------------------------------------------------------------ chunking
    progress.start("chunking", "Divido il documento in estratti")
    chunker = MarkdownChunker(
        headers_to_split_on=["#", "##", "###", "####"],
        chunk_size=1500,
        chunk_overlap=200,
        max_table_rows=max_table_rows,
        count_tokens=get_token_counter(),
        reference_patterns=[REFERENCE_PATTERNS[name] for name in reference_codes],
    )
    chunks = [
        c.to_dict()
        for c in chunker.chunk_document(doc, doc_id=doc_id, table_images_manifest=manifest)
    ]
    if not chunks:
        raise EmptyDocumentError("Dal PDF non è stato estratto alcun testo.")
    (out_dir / "chunks.json").write_text(
        json.dumps(chunks, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    n_tables = sum(c["chunk_type"] == "table" for c in chunks)
    progress.step_done(
        f"{_count(len(chunks), 'estratto creato', 'estratti creati')} "
        f"({len(chunks) - n_tables} di testo, {n_tables} di tabelle)"
    )
    if reference_codes:
        n_codes = sum(bool(c["metadata"].get("reference_codes")) for c in chunks)
        names = ", ".join(name.upper() for name in reference_codes)
        progress.step_done(f"Codici {names} riconosciuti in {_count(n_codes, 'estratto', 'estratti')}")

    # ------------------------------------------------------------ embeddings
    progress.start("embeddings", "Calcolo i vettori di significato e di parole")
    texts = [str(c["content"]) for c in chunks]
    dense: list[list[float]] = []
    sparse: list[Any] = []
    progress.update(0, len(texts), "estratti")
    for start in range(0, len(texts), EMBED_BATCH):
        out = embedder.embed_texts_both(texts[start : start + EMBED_BATCH], batch_size=EMBED_BATCH)
        dense += out.dense_vecs
        sparse += out.sparse_vecs
        progress.update(len(dense), len(texts), "estratti")
    progress.step_done(f"Vettori calcolati per {_count(len(texts), 'estratto', 'estratti')}")

    # ------------------------------------------------------------ vectordb
    progress.start("vectordb", "Scrivo gli estratti nell'indice")
    store.index_chunks(
        chunks,
        dense_vectors=dense,
        sparse_vectors=sparse,
        batch_size=EMBED_BATCH,
        replace_existing_doc=True,
        on_progress=lambda done, total: progress.update(done, total, "estratti"),
    )
    progress.step_done(_count(len(chunks), "estratto indicizzato", "estratti indicizzati"))
    return IngestReport(
        doc_id=doc_id,
        pages=n_pages,
        tables=len(doc.tables),
        chunks=len(chunks),
        ocr_pages=ocr_pages,
        seconds=round(time.perf_counter() - t0, 1),
    )
