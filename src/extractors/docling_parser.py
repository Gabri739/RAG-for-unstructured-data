"""Parser PDF basato su Docling, con OCR automatico per i documenti scansionati.

Primo blocco della pipeline di ingestione: PDF -> DoclingDocument.
L'OCR si attiva solo se almeno una pagina è un'immagine senza testo digitale.
"""

import logging
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions, TableFormerMode
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling_core.types.doc import DoclingDocument, TableItem, TextItem

from .table_fusion import FusionReport, fuse_document_tables
from .table_images import MARKDOWN_BLOCKED_META, TableImagesReport, save_table_images

_log = logging.getLogger(__name__)

# Una pagina è considerata scansionata se ha pochissimo testo ed è coperta da un'immagine
MIN_PAGE_CHARS: int = 50
MIN_IMAGE_COVERAGE: float = 0.50


@dataclass
class ParseResult:
    """Risultato del parsing di un PDF (con tabelle multi-pagina fuse)."""

    document: DoclingDocument
    markdown: str
    pages: int
    tables: int
    fusion_report: FusionReport | None = None
    table_images: TableImagesReport | None = None
    ocr_pages: list[int] | None = None


def find_scanned_pages(pdf_path: Path) -> list[int]:
    """Numeri (da 1) delle pagine senza testo digitale, che richiedono l'OCR."""
    import pypdfium2 as pdfium
    import pypdfium2.raw as pdfium_c

    scanned: list[int] = []
    pdf = pdfium.PdfDocument(pdf_path)
    try:
        for index in range(len(pdf)):
            page = pdf[index]
            if len(page.get_textpage().get_text_range().strip()) >= MIN_PAGE_CHARS:
                continue
            area = page.get_width() * page.get_height()
            covered = 0.0
            for obj in page.get_objects(max_depth=3):
                if obj.type == pdfium_c.FPDF_PAGEOBJ_IMAGE:
                    left, bottom, right, top = obj.get_bounds()
                    covered += abs((right - left) * (top - bottom))
            if area and covered / area >= MIN_IMAGE_COVERAGE:
                scanned.append(index + 1)
    finally:
        pdf.close()
    return scanned


@lru_cache(maxsize=2)
def get_converter(ocr: bool = False) -> DocumentConverter:
    """Crea (una sola volta per modalità) il converter Docling: caricare i modelli è costoso."""
    import torch

    opts = PdfPipelineOptions()
    opts.do_ocr = ocr
    opts.do_table_structure = True
    opts.table_structure_options.mode = TableFormerMode.ACCURATE
    opts.table_structure_options.do_cell_matching = True
    opts.generate_page_images = True  # necessario per i ritagli delle tabelle
    opts.generate_picture_images = True
    opts.images_scale = 2.0
    opts.accelerator_options = AcceleratorOptions(
        device=AcceleratorDevice.CUDA if torch.cuda.is_available() else AcceleratorDevice.CPU,
        num_threads=8,
    )
    return DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=opts)}
    )


def convert_pdf(pdf_path: Path, *, ocr: bool | None = None) -> tuple[DoclingDocument, list[int]]:
    """Converte un PDF in DoclingDocument.

    Con `ocr=None` (default) l'OCR si attiva solo se ci sono pagine scansionate.
    Restituisce `(documento, pagine_scansionate)`.
    """
    scanned = find_scanned_pages(pdf_path)
    use_ocr = bool(scanned) if ocr is None else ocr
    if scanned:
        _log.info("Pagine senza testo digitale: %s (OCR %s)", scanned, "attivo" if use_ocr else "spento")
    doc = get_converter(use_ocr).convert(pdf_path).document

    if scanned and use_ocr:
        with_text = {
            p.page_no for item, _ in doc.iterate_items() if isinstance(item, (TextItem, TableItem)) for p in item.prov
        }
        missing = [p for p in scanned if p not in with_text]
        if missing:
            _log.warning("Nessun testo riconosciuto nelle pagine scansionate: %s", missing)
    return doc, scanned


def parse_pdf(
    pdf_path: Path,
    output_dir: Path | None = None,
    *,
    fuse_tables: bool = True,
    save_images: bool = True,
) -> ParseResult:
    """Converte un PDF in DoclingDocument, fonde le tabelle multi-pagina e salva gli artefatti.

    Se `output_dir` è indicato e `save_images` è True, salva anche un PNG per ogni pagina di
    ogni tabella in `<output_dir>/table_images/` e ne registra i percorsi (relativi a
    `output_dir`) nei metadati della tabella (`table.meta["rag__table_images"]`).
    """
    doc, ocr_pages = convert_pdf(pdf_path)
    fusion_report = fuse_document_tables(doc) if fuse_tables else None
    table_images: TableImagesReport | None = None

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        if save_images:
            table_images = save_table_images(
                doc, output_dir / "table_images", relative_to=output_dir
            )

    markdown = doc.export_to_markdown(blocked_meta_names=MARKDOWN_BLOCKED_META)

    if output_dir is not None:
        (output_dir / "document.md").write_text(markdown, encoding="utf-8")
        doc.save_as_json(output_dir / "document.json")

    return ParseResult(
        document=doc,
        markdown=markdown,
        pages=len(doc.pages),
        tables=len(doc.tables),
        fusion_report=fusion_report,
        table_images=table_images,
        ocr_pages=ocr_pages,
    )

