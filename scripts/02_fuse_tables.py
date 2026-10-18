"""Step 2 - Fusione delle tabelle multi-pagina su DoclingDocument.

Carica `document.json` prodotto dallo Step 1, applica `fuse_document_tables(doc)` e salva:
  - <out>/source_fused.md           Markdown completo del documento con tabelle fuse
  - <out>/document_fused.json       DoclingDocument aggiornato con tabelle e provenance unificate
  - <out>/fusion_report.json        Report dettagliato delle catene fuse e delle righe ricucite
  - <out>/tables_fused_md/tN_pP.md  Markdown di ogni tabella dopo la fusione

Uso:
    uv run python scripts/02_fuse_tables.py [cartella_output]
"""

import dataclasses
import json
import sys
import time
from pathlib import Path

from docling_core.types.doc import DoclingDocument

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.extractors.table_fusion import fuse_document_tables  # noqa: E402

OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "data/processed/docling/dnf2023"
DOC_JSON = OUT / "document.json"

if not DOC_JSON.exists():
    raise FileNotFoundError(
        f"File {DOC_JSON} non trovato. Esegui prima: uv run python scripts/01_parse.py"
    )

t0 = time.perf_counter()
doc = DoclingDocument.load_from_json(DOC_JSON)
load_s = time.perf_counter() - t0

t1 = time.perf_counter()
report = fuse_document_tables(doc)
fuse_s = time.perf_counter() - t1

print(f"Caricamento document.json: {load_s:.2f}s | Fusione tabelle: {fuse_s * 1000:.1f}ms")
print(
    f"Tabelle prima: {report.original_table_count} | "
    f"Vuote rimosse: {len(report.empty_tables_removed)} {report.empty_tables_removed} | "
    f"Catene multi-pagina fuse: {len(report.merged_chains)} | "
    f"Tabelle finali: {report.fused_table_count}"
)
print("-" * 95)

for m in report.merged_chains:
    pages_str = "->".join(str(p) for p in m.pages)
    idx_str = " + ".join(f"T{i}" for i in m.merged_indices)
    cap_str = f" | caption: {m.caption[:50]}" if m.caption else ""
    print(
        f"  [{idx_str:18s}] pag {pages_str:12s} -> {m.num_rows:2d} righe x {m.num_cols} col "
        f"(righe spezzate ricucite: {m.stitched_rows}){cap_str}"
    )

# Salva artefatti aggiornati
(OUT / "source_fused.md").write_text(doc.export_to_markdown(), encoding="utf-8")
doc.save_as_json(OUT / "document_fused.json")
(OUT / "fusion_report.json").write_text(
    json.dumps(dataclasses.asdict(report), ensure_ascii=False, indent=2),
    encoding="utf-8",
)

fused_md_dir = OUT / "tables_fused_md"
fused_md_dir.mkdir(parents=True, exist_ok=True)
for f in fused_md_dir.glob("*.md"):
    f.unlink()

for i, t in enumerate(doc.tables):
    pages = sorted({p.page_no for p in t.prov})
    p_tag = f"p{pages[0]}" if len(pages) == 1 else f"p{pages[0]}-{pages[-1]}"
    (fused_md_dir / f"t{i}_{p_tag}.md").write_text(
        t.export_to_markdown(doc=doc), encoding="utf-8"
    )

print("-" * 95)
print(f"Salvati: {OUT / 'source_fused.md'}, {OUT / 'fusion_report.json'} e {len(doc.tables)} file in {fused_md_dir}")

