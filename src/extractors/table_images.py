"""Salvataggio su disco dell'immagine di ogni tabella di un DoclingDocument.

Terzo blocco della pipeline di ingestione:
  PDF nativi -> Docling -> Fusione (tabelle e righe) -> Immagini -> Chunking -> ...

Per ogni tabella (già fusa da `table_fusion.fuse_document_tables`):
  - Un'immagine PNG per ciascun ProvenanceItem, quindi una per pagina se la tabella
    è multi-pagina (la capofila conserva la lista completa di `prov` dopo la fusione).
  - Il ritaglio parte dall'immagine di pagina generata da Docling
    (`generate_page_images=True`, `images_scale=2.0` -> 144 DPI) con un piccolo margine
    per non tagliare i bordi della tabella.
  - I percorsi (relativi a `relative_to`) vengono scritti nei metadati della tabella in
    `table.meta["rag__table_images"]` (campo custom namespaced di docling-core), così
    restano nel `document_fused.json` e sono disponibili al chunker / vector store.

Formato di ogni voce dei metadati:
    {
      "page": 44,                      # numero di pagina (1-based, come Docling)
      "prov_index": 0,                 # indice nella lista table.prov
      "image_path": "table_images/t12_p44.png",
      "bbox": {"l": .., "t": .., "r": .., "b": .., "origin": "BOTTOMLEFT"},
      "width": 1012, "height": 433     # dimensioni in pixel dell'immagine salvata
    }
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from docling_core.types.doc import DoclingDocument, ProvenanceItem, TableItem
from docling_core.types.doc.common.meta import FloatingMeta
from PIL import Image

META_NAMESPACE: str = "rag"
META_FIELD: str = "table_images"
META_KEY: str = f"{META_NAMESPACE}__{META_FIELD}"  # -> "rag__table_images"

DEFAULT_PADDING_PT: float = 4.0  # margine attorno al bbox (in punti PDF)

# ATTENZIONE: il serializer Markdown di docling-core stampa anche i campi meta custom.
# Per un Markdown pulito usare: doc.export_to_markdown(blocked_meta_names=MARKDOWN_BLOCKED_META)
MARKDOWN_BLOCKED_META: set[str] = {META_KEY}


@dataclass
class TableImageInfo:
    """Riepilogo delle immagini salvate per una singola tabella."""

    table_index: int
    self_ref: str
    pages: list[int]
    image_paths: list[str]
    caption: str = ""


@dataclass
class TableImagesReport:
    """Riepilogo dell'operazione di salvataggio immagini su un DoclingDocument."""

    images_dir: str
    table_count: int
    image_count: int
    tables: list[TableImageInfo] = field(default_factory=list)
    missing: list[dict] = field(default_factory=list)  # tabelle/pagine senza immagine di pagina


def _crop_prov(
    doc: DoclingDocument,
    prov: ProvenanceItem,
    padding_pt: float,
) -> Image.Image | None:
    """Ritaglia dall'immagine di pagina l'area del bbox (+ margine), come `DocItem.get_image`."""
    page = doc.pages.get(prov.page_no)
    if page is None or page.size is None or page.image is None:
        return None
    page_image = page.image.pil_image
    if page_image is None:
        return None

    bbox = prov.bbox.to_top_left_origin(page_height=page.size.height)
    scale_x = page_image.width / page.size.width
    scale_y = page_image.height / page.size.height

    left = max(0.0, (bbox.l - padding_pt) * scale_x)
    top = max(0.0, (bbox.t - padding_pt) * scale_y)
    right = min(float(page_image.width), (bbox.r + padding_pt) * scale_x)
    bottom = min(float(page_image.height), (bbox.b + padding_pt) * scale_y)
    if right <= left or bottom <= top:
        return None
    return page_image.crop((round(left), round(top), round(right), round(bottom)))


def _to_rel_path(path: Path, relative_to: Path | None) -> str:
    """Percorso in formato POSIX, relativo a `relative_to` se possibile."""
    if relative_to is not None:
        try:
            return path.resolve().relative_to(relative_to.resolve()).as_posix()
        except ValueError:
            pass
    return path.resolve().as_posix()


def _set_table_meta(table: TableItem, entries: list[dict]) -> None:
    """Scrive la lista delle immagini nei metadati custom della tabella."""
    if table.meta is None:
        table.meta = FloatingMeta()
    table.meta.set_custom_field(namespace=META_NAMESPACE, name=META_FIELD, value=entries)


def get_table_images(table: TableItem) -> list[dict]:
    """Legge dai metadati della tabella la lista delle immagini salvate (vuota se assente)."""
    if table.meta is None:
        return []
    return list(table.meta.get_custom_part().get(META_KEY, []))


def save_table_images(
    doc: DoclingDocument,
    images_dir: Path,
    *,
    relative_to: Path | None = None,
    padding_pt: float = DEFAULT_PADDING_PT,
    clean: bool = True,
) -> TableImagesReport:
    """Salva un PNG per ogni pagina di ogni tabella e registra i percorsi in `table.meta`.

    Args:
        doc: DoclingDocument (idealmente già passato da `fuse_document_tables`) con le
            immagini di pagina disponibili (`generate_page_images=True`, oppure JSON
            salvato con `ImageRefMode.EMBEDDED`, default di `save_as_json`).
        images_dir: cartella di destinazione dei PNG.
        relative_to: base per i percorsi scritti nei metadati (default: `images_dir.parent`).
        padding_pt: margine in punti PDF attorno al bbox della tabella.
        clean: se True rimuove i PNG preesistenti in `images_dir` (evita file orfani
            quando cambia la numerazione delle tabelle dopo la fusione).

    Nomi file: `t{indice}_p{pagina}.png` (es. `t12_p44.png`, `t12_p45.png` per una tabella
    su due pagine); se la stessa pagina compare più volte nella prov: `t12_p44_2.png`.
    """
    images_dir.mkdir(parents=True, exist_ok=True)
    if clean:
        for f in images_dir.glob("t*_p*.png"):
            f.unlink()
    base = relative_to if relative_to is not None else images_dir.parent

    infos: list[TableImageInfo] = []
    missing: list[dict] = []
    image_count = 0

    for t_idx, table in enumerate(doc.tables):
        entries: list[dict] = []
        seen_pages: dict[int, int] = {}

        for k, prov in enumerate(table.prov):
            if not isinstance(prov, ProvenanceItem):
                continue
            img = _crop_prov(doc, prov, padding_pt)
            if img is None:
                missing.append({"table_index": t_idx, "self_ref": table.self_ref,
                                "page": prov.page_no, "prov_index": k})
                continue

            seen_pages[prov.page_no] = seen_pages.get(prov.page_no, 0) + 1
            suffix = "" if seen_pages[prov.page_no] == 1 else f"_{seen_pages[prov.page_no]}"
            out_path = images_dir / f"t{t_idx}_p{prov.page_no}{suffix}.png"
            img.save(out_path)
            image_count += 1

            b = prov.bbox
            entries.append({
                "page": prov.page_no,
                "prov_index": k,
                "image_path": _to_rel_path(out_path, base),
                "bbox": {
                    "l": round(b.l, 1), "t": round(b.t, 1),
                    "r": round(b.r, 1), "b": round(b.b, 1),
                    "origin": b.coord_origin.value,
                },
                "width": img.width,
                "height": img.height,
            })

        _set_table_meta(table, entries)
        infos.append(TableImageInfo(
            table_index=t_idx,
            self_ref=table.self_ref,
            pages=[e["page"] for e in entries],
            image_paths=[e["image_path"] for e in entries],
            caption=table.caption_text(doc),
        ))

    return TableImagesReport(
        images_dir=_to_rel_path(images_dir, base),
        table_count=len(doc.tables),
        image_count=image_count,
        tables=infos,
        missing=missing,
    )
