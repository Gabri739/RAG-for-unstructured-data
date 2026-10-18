"""Parser PDF basato su Docling (PDF nativi digitali, senza OCR).

Primo blocco della pipeline di ingestione: PDF -> DoclingDocument.
"""

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions, TableFormerMode
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling_core.types.doc import DoclingDocument

from .table_fusion import FusionReport, fuse_document_tables


@dataclass
class ParseResult:
    """Risultato del parsing di un PDF (con tabelle multi-pagina fuse)."""

    document: DoclingDocument
    markdown: str
    pages: int
    tables: int
    fusion_report: FusionReport | None = None


@lru_cache(maxsize=1)
def get_converter() -> DocumentConverter:
    """Crea (una sola volta) il converter Docling: il caricamento dei modelli è costoso."""
    import torch

    opts = PdfPipelineOptions()
    opts.do_ocr = False  # PDF nativi digitali
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


def parse_pdf(
    pdf_path: Path,
    output_dir: Path | None = None,
    *,
    fuse_tables: bool = True,
) -> ParseResult:
    """Converte un PDF in DoclingDocument, fonde le tabelle multi-pagina e salva gli artefatti."""
    doc = get_converter().convert(pdf_path).document
    fusion_report = fuse_document_tables(doc) if fuse_tables else None
    markdown = doc.export_to_markdown()

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "document.md").write_text(markdown, encoding="utf-8")
        doc.save_as_json(output_dir / "document.json")

    return ParseResult(
        document=doc,
        markdown=markdown,
        pages=len(doc.pages),
        tables=len(doc.tables),
        fusion_report=fusion_report,
    )

