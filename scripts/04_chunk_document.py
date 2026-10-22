"""Step 4 - Chunking per intestazioni e per righe tabella su DoclingDocument.

Carica `document_images.json` (o `document_fused.json`) e il manifest `table_images.json`
prodotti dallo Step 3/2, esegue `MarkdownChunker.chunk_document(doc)` e salva:
  - <out>/chunks.json         lista completa dei chunk con content, headers e metadata
  - <out>/chunks_preview.md   anteprima leggibile di tutti i chunk generati

Uso:
    uv run python scripts/04_chunk_document.py [cartella_output] [max_table_rows]
"""

import json
import sys
import time
from pathlib import Path

from docling_core.types.doc import DoclingDocument

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.chunkers import MarkdownChunker  # noqa: E402

OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "data/processed/docling/dnf2023"
MAX_TABLE_ROWS = int(sys.argv[2]) if len(sys.argv) > 2 else 25

DOC_JSON = OUT / "document_images.json"
if not DOC_JSON.exists():
    DOC_JSON = OUT / "document_fused.json"
if not DOC_JSON.exists():
    DOC_JSON = OUT / "document.json"
if not DOC_JSON.exists():
    raise FileNotFoundError(
        f"Nessun file Docling trovato in {OUT}. Esegui prima gli script 01, 02 e 03."
    )

MANIFEST_JSON = OUT / "table_images.json"

t0 = time.perf_counter()
doc = DoclingDocument.load_from_json(DOC_JSON)
load_s = time.perf_counter() - t0

t1 = time.perf_counter()
chunker = MarkdownChunker(
    headers_to_split_on=["#", "##", "###", "####"],
    chunk_size=1500,
    chunk_overlap=200,
    max_table_rows=MAX_TABLE_ROWS,
)
chunks = chunker.chunk_document(
    doc,
    doc_id=OUT.name,
    table_images_manifest=MANIFEST_JSON if MANIFEST_JSON.exists() else None,
)
chunk_s = time.perf_counter() - t1

text_chunks = [c for c in chunks if c.chunk_type == "text"]
table_chunks = [c for c in chunks if c.chunk_type == "table"]
split_table_chunks = [c for c in table_chunks if c.metadata.get("total_parts", 1) > 1]
incomplete_tables = [c for c in table_chunks if c.metadata.get("possible_incomplete_table")]

print(f"Caricamento {DOC_JSON.name}: {load_s:.2f}s | Chunking: {chunk_s * 1000:.1f}ms")
print(
    f"Totale chunk: {len(chunks)} | "
    f"Testo: {len(text_chunks)} | "
    f"Tabella: {len(table_chunks)} "
    f"(di cui divisi per righe: {len(split_table_chunks)}, possibili incompleti: {len(incomplete_tables)})"
)
print("-" * 95)

for idx, c in enumerate(chunks):
    m = c.metadata
    p_str = m.get("page_range") or "-"
    sec_str = m.get("section") or "(nessuna sezione)"
    if c.chunk_type == "table":
        empty_cols_str = f" | col_vuote={m['empty_columns']}" if m.get("empty_columns") else ""
        print(
            f"  [{idx:03d}] TABLE pag {p_str:7s} | righe {m['row_range']:7s} "
            f"(parte {m['part']}/{m['total_parts']}) | img={len(m['image_paths'])} | "
            f"incompleta={m['possible_incomplete_table']} ({m['empty_cells_ratio']*100:.0f}% vuote){empty_cols_str} | "
            f"titolo={(m.get('table_title') or '')[:40]}"
        )
    else:
        print(
            f"  [{idx:03d}] TEXT  pag {p_str:7s} | {len(c.content):4d} chars "
            f"(parte {m['part']}/{m['total_parts']}) | sez={sec_str[:55]}"
        )

(OUT / "chunks.json").write_text(
    json.dumps([c.to_dict() for c in chunks], ensure_ascii=False, indent=2),
    encoding="utf-8",
)

preview_lines: list[str] = [f"# Anteprima Chunk ({len(chunks)} totali)\n"]
for idx, c in enumerate(chunks):
    preview_lines.append(f"---\n## Chunk {idx} (`{c.chunk_type}` — `{c.metadata.get('chunk_id')}`)\n")
    preview_lines.append(f"```json\n{json.dumps(c.metadata, ensure_ascii=False, indent=2)}\n```\n")
    preview_lines.append(c.content + "\n")

(OUT / "chunks_preview.md").write_text("\n".join(preview_lines), encoding="utf-8")

print("-" * 95)
print(f"Salvati: {OUT / 'chunks.json'} e {OUT / 'chunks_preview.md'}")
