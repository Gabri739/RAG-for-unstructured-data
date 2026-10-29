"""Pipeline end-to-end completa (Step 1 -> Step 6) sull'intero documento PDF.

Esegue in sequenza in memoria (salvando tutti gli artefatti intermedi su disco):
  1. Parsing completo del PDF con Docling (TableFormer ACCURATE + immagini pagina)
  2. Fusione delle tabelle multi-pagina (`fuse_document_tables`)
  3. Salvataggio su disco dei ritagli PNG delle tabelle (`save_table_images`)
  4. Chunking strutturato per intestazioni e righe tabella (`MarkdownChunker`)
  5. Calcolo degli embedding `BAAI/bge-m3` (sia `dense` 1024-d sia `sparse`) e salvataggio su Qdrant
  6. Valutazione quantitativa del retrieval sulle domande di test (`06_eval_retrieval.py`)

Uso:
    python scripts/run_full_pipeline.py [percorso_pdf] [cartella_output] [max_table_rows]
"""

from __future__ import annotations

import dataclasses
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from src.chunkers import MarkdownChunker  # noqa: E402
from src.extractors.docling_parser import get_converter  # noqa: E402
from src.extractors.table_fusion import fuse_document_tables  # noqa: E402
from src.extractors.table_images import MARKDOWN_BLOCKED_META, save_table_images  # noqa: E402
from src.vectorstore import (  # noqa: E402
    DEFAULT_COLLECTION_NAME,
    DEFAULT_QDRANT_PATH,
    QdrantVectorStore,
    get_embedder,
)


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main() -> None:
    pdf_path = (
        Path(sys.argv[1])
        if len(sys.argv) > 1
        else ROOT / "data/raw/20260511_131038/source.pdf"
    )
    out_dir = (
        Path(sys.argv[2])
        if len(sys.argv) > 2
        else ROOT / "data/processed/docling/dnf2023"
    )
    max_table_rows = int(sys.argv[3]) if len(sys.argv) > 3 else 25
    doc_id = out_dir.name

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF non trovato: {pdf_path}")

    out_dir.mkdir(parents=True, exist_ok=True)
    t_pipeline_start = time.perf_counter()

    cuda = torch.cuda.is_available()
    device_str = f"cuda ({torch.cuda.get_device_name(0)})" if cuda else "cpu"
    log(f"Avvio pipeline completa su: {pdf_path} -> {out_dir} (doc_id={doc_id!r}, device={device_str})")

    # ------------------------------------------------------------------
    # STEP 1: Parsing PDF con Docling
    # ------------------------------------------------------------------
    log("STEP 1/6 - Caricamento converter Docling e conversione integrale del PDF...")
    t0 = time.perf_counter()
    converter = get_converter()
    doc = converter.convert(pdf_path).document
    parse_s = time.perf_counter() - t0
    n_pages = len(doc.pages)
    raw_tables = len(doc.tables)
    log(
        f"STEP 1/6 completato in {parse_s:.1f}s ({parse_s / max(n_pages, 1):.2f}s/pag) | "
        f"Pagine: {n_pages} | Tabelle grezze rilevate: {raw_tables}"
    )

    # ------------------------------------------------------------------
    # STEP 2: Fusione tabelle multi-pagina
    # ------------------------------------------------------------------
    log("STEP 2/6 - Fusione tabelle multi-pagina consecutive...")
    t1 = time.perf_counter()
    fusion_report = fuse_document_tables(doc)
    fuse_s = time.perf_counter() - t1
    log(
        f"STEP 2/6 completato in {fuse_s * 1000:.1f}ms | "
        f"Tabelle prima: {fusion_report.original_table_count} -> dopo: {fusion_report.fused_table_count} "
        f"(catene multi-pagina fuse: {len(fusion_report.merged_chains)}, "
        f"vuote rimosse: {len(fusion_report.empty_tables_removed)})"
    )
    (out_dir / "fusion_report.json").write_text(
        json.dumps(dataclasses.asdict(fusion_report), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # ------------------------------------------------------------------
    # STEP 3: Salvataggio immagini tabelle (1 PNG per pagina)
    # ------------------------------------------------------------------
    log("STEP 3/6 - Ritaglio e salvataggio PNG delle tabelle su disco...")
    t2 = time.perf_counter()
    images_report = save_table_images(doc, out_dir / "table_images", relative_to=out_dir)
    img_s = time.perf_counter() - t2
    log(
        f"STEP 3/6 completato in {img_s:.2f}s | "
        f"Tabelle: {images_report.table_count} | PNG salvati: {images_report.image_count} | "
        f"Mancanti: {len(images_report.missing)}"
    )

    (out_dir / "table_images.json").write_text(
        json.dumps(dataclasses.asdict(images_report), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    fused_md = doc.export_to_markdown(blocked_meta_names=MARKDOWN_BLOCKED_META)
    (out_dir / "source_fused.md").write_text(fused_md, encoding="utf-8")

    fused_md_dir = out_dir / "tables_fused_md"
    fused_md_dir.mkdir(parents=True, exist_ok=True)
    for old_md in fused_md_dir.glob("*.md"):
        old_md.unlink()
    for i, t in enumerate(doc.tables):
        pages = sorted({p.page_no for p in t.prov})
        p_tag = f"p{pages[0]}" if len(pages) == 1 else f"p{pages[0]}-{pages[-1]}"
        (fused_md_dir / f"t{i}_{p_tag}.md").write_text(
            t.export_to_markdown(doc=doc), encoding="utf-8"
        )

    # Rimuove le immagini intere di pagina base64 prima di serializzare il JSON su disco
    # (i ritagli PNG delle tabelle sono già salvati in table_images/)
    for page_obj in doc.pages.values():
        page_obj.image = None
    doc.save_as_json(out_dir / "document_images.json")

    (out_dir / "timing.json").write_text(
        json.dumps(
            {
                "pdf": str(pdf_path),
                "pages": n_pages,
                "parse_seconds": round(parse_s, 2),
                "seconds_per_page": round(parse_s / max(n_pages, 1), 3),
                "device": "cuda" if cuda else "cpu",
                "raw_tables": raw_tables,
                "fused_tables": fusion_report.fused_table_count,
                "merged_chains": len(fusion_report.merged_chains),
                "table_png_count": images_report.image_count,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    # ------------------------------------------------------------------
    # STEP 4: Chunking strutturato (intestazioni + righe tabella)
    # ------------------------------------------------------------------
    log(f"STEP 4/6 - Chunking del documento (max_table_rows={max_table_rows})...")
    t3 = time.perf_counter()
    chunker = MarkdownChunker(
        headers_to_split_on=["#", "##", "###", "####"],
        chunk_size=1500,
        chunk_overlap=200,
        max_table_rows=max_table_rows,
    )
    chunks = chunker.chunk_document(
        doc,
        doc_id=doc_id,
        table_images_manifest=out_dir / "table_images.json",
    )
    chunk_s = time.perf_counter() - t3

    text_chunks = [c for c in chunks if c.chunk_type == "text"]
    table_chunks = [c for c in chunks if c.chunk_type == "table"]
    split_tables = [c for c in table_chunks if c.metadata.get("total_parts", 1) > 1]
    incomplete_tables = [c for c in table_chunks if c.metadata.get("possible_incomplete_table")]

    log(
        f"STEP 4/6 completato in {chunk_s * 1000:.1f}ms | "
        f"Chunk totali: {len(chunks)} (testo: {len(text_chunks)}, tabelle: {len(table_chunks)}, "
        f"blocchi tabella divisi per righe: {len(split_tables)}, "
        f"possibili tabelle incomplete: {len(incomplete_tables)})"
    )

    chunks_dicts = [c.to_dict() for c in chunks]
    (out_dir / "chunks.json").write_text(
        json.dumps(chunks_dicts, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    preview_lines: list[str] = [f"# Anteprima Chunk ({len(chunks)} totali)\n"]
    for idx, c in enumerate(chunks):
        preview_lines.append(
            f"---\n## Chunk {idx} (`{c.chunk_type}` — `{c.metadata.get('chunk_id')}`)\n"
        )
        preview_lines.append(
            f"```json\n{json.dumps(c.metadata, ensure_ascii=False, indent=2)}\n```\n"
        )
        preview_lines.append(c.content + "\n")
    (out_dir / "chunks_preview.md").write_text("\n".join(preview_lines), encoding="utf-8")

    # ------------------------------------------------------------------
    # STEP 5: Embedding BAAI/bge-m3 (dense + sparse) e storage su Qdrant
    # ------------------------------------------------------------------
    log("STEP 5/6 - Caricamento BAAI/bge-m3 e indicizzazione (dense + sparse) su Qdrant...")
    t4 = time.perf_counter()
    embedder = get_embedder()
    store = QdrantVectorStore(
        collection_name=DEFAULT_COLLECTION_NAME,
        path=ROOT / DEFAULT_QDRANT_PATH,
    )
    # Rimuove eventuali chunk del precedente mini-test su 2 pagine per evitare duplicati
    removed_test = store.delete_document("dnf2023_test_p43_44")
    if removed_test:
        log(f"  Rimossi {removed_test} chunk di test ('dnf2023_test_p43_44') da Qdrant.")

    idx_report = store.index_chunks(
        chunks_dicts,
        embedder=embedder,
        batch_size=16,
        replace_existing_doc=True,
    )
    index_s = time.perf_counter() - t4
    total_qdrant_points = store.count_points()
    log(
        f"STEP 5/6 completato in {index_s:.1f}s | "
        f"Punti indicizzati: {idx_report.indexed_points} (totale collezione '{idx_report.collection_name}': {total_qdrant_points}) | "
        f"Vettori salvati: dense ({idx_report.vector_dimension}-d) + sparse"
    )

    # ------------------------------------------------------------------
    # STEP 6: Valutazione Retrieval sul documento completo
    # ------------------------------------------------------------------
    questions_path = ROOT / "tests/eval_questions_dnf2023.json"
    if questions_path.exists():
        log(f"STEP 6/6 - Valutazione retrieval su {questions_path.name}...")
        # Import dinamico dallo script 06 per riutilizzare esattamente la stessa logica
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "eval_retrieval", ROOT / "scripts/06_eval_retrieval.py"
        )
        if spec and spec.loader:
            eval_mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(eval_mod)
            questions = json.loads(questions_path.read_text(encoding="utf-8"))
            eval_report = eval_mod.evaluate_questions(
                store,
                questions,
                embedder=embedder,
                doc_id=doc_id,
                top_k=20,
                score_threshold=eval_mod.DEFAULT_SCORE_THRESHOLD,
            )
            out_eval = ROOT / "data/qdrant/eval_report.json"
            out_eval.write_text(
                json.dumps(eval_report, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            log(
                f"STEP 6/6 completato | "
                f"1° posto: {eval_report['first_count']}/{eval_report['positive_questions']} "
                f"({eval_report['hit_at_1']*100:.1f}%) | "
                f"Top-5: {eval_report['top5_count']}/{eval_report['positive_questions']} "
                f"({eval_report['hit_at_5']*100:.1f}%) | "
                f"Top-20: {eval_report['top20_count']}/{eval_report['positive_questions']} "
                f"({eval_report['hit_at_20']*100:.1f}%) | "
                f"MRR: {eval_report['mrr']:.4f} | "
                f"Fuori perimetro scartate: {eval_report['negative_correct']}/{eval_report['negative_questions']}"
            )

    store.close()
    total_s = time.perf_counter() - t_pipeline_start
    log(f"PIPELINE COMPLETATA CON SUCCESSO in {total_s:.1f}s ({total_s / 60:.1f} min)!")


if __name__ == "__main__":
    main()

