"""Step 1 - Parsing di un PDF con Docling per ispezionare le tabelle.

Esegue Docling (TableFormer ACCURATE, GPU) su un PDF nativo digitale e produce:
  - <out>/<stem>.md          Markdown completo del documento
  - <out>/tables/tN_pP.png   ritaglio di ogni tabella
  - <out>/tables.json        elenco tabelle (pagina, righe, colonne, bbox, header)
  - <out>/tables_md/tN_pP.md Markdown della singola tabella
  - <out>/timing.json        tempi di conversione

Uso:
    uv run python scripts/01_parse.py [percorso_pdf] [cartella_output]
"""

import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.extractors.docling_parser import get_converter  # noqa: E402

PDF = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "data/raw/20260511_131038/source.pdf"
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT / "data/processed/docling/dnf2023"

(OUT / "tables").mkdir(parents=True, exist_ok=True)
(OUT / "tables_md").mkdir(parents=True, exist_ok=True)

cuda = torch.cuda.is_available()
print(f"torch {torch.__version__} | CUDA disponibile: {cuda}"
      + (f" ({torch.cuda.get_device_name(0)})" if cuda else ""))

converter = get_converter()

t0 = time.perf_counter()
doc = converter.convert(PDF).document
elapsed = time.perf_counter() - t0
n_pages = len(doc.pages)
print(f"Conversione: {elapsed:.1f}s per {n_pages} pagine ({elapsed / max(n_pages, 1):.2f} s/pag)")

(OUT / f"{PDF.stem}.md").write_text(doc.export_to_markdown(), encoding="utf-8")
doc.save_as_json(OUT / "document.json")

tables_info = []
for i, t in enumerate(doc.tables):
    p = t.prov[0]
    bbox = p.bbox
    # Prima riga della tabella: utile per capire se le continuazioni hanno l'header
    first_row = [c.text for c in t.data.grid[0]] if t.data.grid else []
    has_col_header = any(c.column_header for c in t.data.table_cells)

    print(f"Tabella {i}: pag {p.page_no}, "
          f"{t.data.num_rows} righe x {t.data.num_cols} colonne, "
          f"bbox (l={bbox.l:.1f}, t={bbox.t:.1f}, r={bbox.r:.1f}, b={bbox.b:.1f}, {bbox.coord_origin.value}), "
          f"header={has_col_header}")

    tables_info.append({
        "index": i,
        "page": p.page_no,
        "pages": sorted({pr.page_no for pr in t.prov}),
        "rows": t.data.num_rows,
        "cols": t.data.num_cols,
        "bbox": {"l": bbox.l, "t": bbox.t, "r": bbox.r, "b": bbox.b,
                 "origin": bbox.coord_origin.value},
        "has_column_header": has_col_header,
        "first_row": first_row,
        "caption": t.caption_text(doc),
    })

    (OUT / "tables_md" / f"t{i}_p{p.page_no}.md").write_text(
        t.export_to_markdown(doc=doc), encoding="utf-8"
    )
    img = t.get_image(doc)
    if img is not None:
        img.save(OUT / "tables" / f"t{i}_p{p.page_no}.png")

(OUT / "tables.json").write_text(
    json.dumps(tables_info, ensure_ascii=False, indent=2), encoding="utf-8"
)
(OUT / "timing.json").write_text(json.dumps({
    "pdf": str(PDF), "pages": n_pages, "seconds": round(elapsed, 2),
    "seconds_per_page": round(elapsed / max(n_pages, 1), 3),
    "device": "cuda" if cuda else "cpu", "n_tables": len(doc.tables),
}, indent=2), encoding="utf-8")

print(f"\n{len(doc.tables)} tabelle trovate. Output in: {OUT}")

