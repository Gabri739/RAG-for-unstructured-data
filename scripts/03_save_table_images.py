"""Step 3 - Salvataggio delle immagini delle tabelle (una per pagina) con percorso nei metadati.

Carica `document_fused.json` prodotto dallo Step 2 (contiene le immagini di pagina embedded),
applica `save_table_images(doc)` e salva:
  - <out>/table_images/tN_pP.png      ritaglio di ogni tabella, uno per pagina
                                      (tabella multi-pagina -> tN_p44.png, tN_p45.png, ...)
  - <out>/document_images.json        DoclingDocument con i percorsi in table.meta["rag__table_images"]
  - <out>/table_images.json           manifest: tabella -> pagine, percorsi immagine, caption

I percorsi nei metadati sono relativi a <out> (es. "table_images/t12_p44.png").

Uso:
    uv run python scripts/03_save_table_images.py [cartella_output]
"""

import dataclasses
import json
import sys
import time
from pathlib import Path

from docling_core.types.doc import DoclingDocument

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.extractors.table_images import save_table_images  # noqa: E402

OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "data/processed/docling/dnf2023"
DOC_JSON = OUT / "document_fused.json"

if not DOC_JSON.exists():
    raise FileNotFoundError(
        f"File {DOC_JSON} non trovato. Esegui prima: uv run python scripts/02_fuse_tables.py"
    )

t0 = time.perf_counter()
doc = DoclingDocument.load_from_json(DOC_JSON)
load_s = time.perf_counter() - t0

t1 = time.perf_counter()
report = save_table_images(doc, OUT / "table_images", relative_to=OUT)
save_s = time.perf_counter() - t1

print(f"Caricamento document_fused.json: {load_s:.2f}s | Salvataggio immagini: {save_s:.2f}s")
print(f"Tabelle: {report.table_count} | Immagini salvate: {report.image_count}")
print("-" * 95)

for info in report.tables:
    if len(info.pages) > 1:
        print(f"  T{info.table_index:<3d} pag {'->'.join(map(str, info.pages)):12s} -> "
              f"{', '.join(Path(p).name for p in info.image_paths)}")

if report.missing:
    print(f"\nATTENZIONE: {len(report.missing)} ritagli non generati (immagine di pagina assente):")
    for m in report.missing:
        print(f"  T{m['table_index']} pag {m['page']} (prov {m['prov_index']})")

doc.save_as_json(OUT / "document_images.json")
(OUT / "table_images.json").write_text(
    json.dumps(dataclasses.asdict(report), ensure_ascii=False, indent=2),
    encoding="utf-8",
)

print("-" * 95)
print(f"Salvati: {report.image_count} PNG in {OUT / 'table_images'}, "
      f"{OUT / 'document_images.json'} e {OUT / 'table_images.json'}")

