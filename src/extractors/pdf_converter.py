"""
PDF to Markdown converter using Ollama vision models.
Core logic extracted from webapp for use in RAG pipelines.
"""

import asyncio
import base64
import json
import os
import re
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path
from typing import AsyncIterator, Callable

import fitz  # PyMuPDF
import httpx
from PIL import Image

# Configuration from environment
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
VISION_MODEL = os.environ.get("VISION_MODEL", "qwen3.5:397b-cloud")
COMPLEX_MODEL = os.environ.get("COMPLEX_MODEL", "Maternion/LightOnOCR-2:latest")

OCR_PROMPT = os.environ.get(
    "OCR_PROMPT",
    "Extract all content from this document image and output clean, well-formatted Markdown. "
    "Preserve headings, paragraphs, lists, and tables. "
    "For tables, ALWAYS use Markdown table syntax with pipes (|) and dashes (-). "
    "Example table format:\n"
    "| Column 1 | Column 2 |\n"
    "|----------|----------|\n"
    "| Value A  | Value B  | "
    "NEVER use HTML tags such as <table>, <tr>, or <td>. "
    "Describe figures concisely in italics. Do not add commentary; return only the Markdown.",
)

VISION_PROMPT = os.environ.get(
    "VISION_PROMPT",
    "Analyze this document image and extract ALL content into clean, well-formatted Markdown. "
    "Preserve headings, paragraphs, lists, and tables with proper formatting. "
    "For tables, ALWAYS use Markdown table syntax with pipes (|) and dashes (-), NEVER HTML tags. "
    "Describe figures concisely in italics. "
    "Output ONLY the final Markdown content without any introductory text.",
)

RENDER_DPI = int(os.environ.get("RENDER_DPI", "150"))


@dataclass
class ConversionResult:
    """Result of PDF/image conversion."""

    markdown: str
    pages: int
    strategy: str


class TableParser(HTMLParser):
    """Parse HTML tables and convert to Markdown."""

    def __init__(self) -> None:
        super().__init__()
        self.tables: list[list[list[tuple[str, int, int]]]] = []
        self._current_table: list[list[tuple[str, int, int]]] = []
        self._current_row: list[tuple[str, int, int]] = []
        self._in_cell = False
        self._cell_text: list[str] = []
        self._cell_attrs: dict[str, str | None] = {}

    def _clear_cell(self) -> None:
        self._in_cell = False
        self._cell_text = []
        self._cell_attrs = {}

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag = tag.lower()
        if tag == "table":
            self._current_table = []
        elif tag == "tr":
            self._current_row = []
        elif tag in ("td", "th"):
            self._in_cell = True
            self._cell_text = []
            self._cell_attrs = dict(attrs)

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag in ("td", "th"):
            text = "".join(self._cell_text).strip()
            rowspan = int(self._cell_attrs.get("rowspan", 1) or 1)
            colspan = int(self._cell_attrs.get("colspan", 1) or 1)
            self._current_row.append((text, rowspan, colspan))
            self._clear_cell()
        elif tag == "tr":
            if self._current_row:
                self._current_table.append(self._current_row)
        elif tag == "table":
            if self._current_table:
                self.tables.append(self._current_table)

    def handle_data(self, data: str) -> None:
        if self._in_cell:
            self._cell_text.append(data)


def html_tables_to_md(text: str) -> str:
    """Convert any HTML <table> blocks in text to Markdown tables."""
    if "<table" not in text.lower():
        return text

    def replace_table(match: re.Match) -> str:
        html = match.group(0)
        parser = TableParser()
        try:
            parser.feed(html)
        except Exception:
            return html

        if not parser.tables:
            return html

        table = parser.tables[0]
        # Build matrix handling colspan and rowspan
        matrix: list[list[str | None]] = []
        pending: dict[tuple[int, int], str] = {}

        for row_idx, row in enumerate(table):
            matrix_row: list[str | None] = []
            col_idx = 0
            while col_idx < len(matrix_row) or (col_idx == len(matrix_row) and row):
                # Check pending rowspan from above
                if (row_idx, col_idx) in pending:
                    matrix_row.append(pending[(row_idx, col_idx)])
                    col_idx += 1
                    continue
                if not row:
                    break
                cell_text, rowspan, colspan = row.pop(0)
                for c in range(colspan):
                    matrix_row.append(cell_text if c == 0 else "")
                    if rowspan > 1:
                        for r in range(1, rowspan):
                            pending[(row_idx + r, col_idx + c)] = cell_text if c == 0 else ""
                col_idx += colspan
            matrix.append(matrix_row)

        if not matrix:
            return html

        max_cols = max(len(r) for r in matrix)
        md_lines: list[str] = []
        for i, row in enumerate(matrix):
            padded = row + [""] * (max_cols - len(row))
            md_lines.append("| " + " | ".join(str(c) for c in padded) + " |")
            if i == 0:
                md_lines.append("|" + "|".join(["---"] * max_cols) + "|")

        return "\n".join(md_lines)

    result = re.sub(r"<table[^>]*>.*?</table>", replace_table, text, flags=re.DOTALL | re.IGNORECASE)
    # Strip any leftover table-related tags
    result = re.sub(r"</?(?:table|thead|tbody|tr|th|td)[^>]*>", "", result, flags=re.IGNORECASE)
    return result


def flush_html_buffer(buffer: str) -> tuple[str, str]:
    """Process buffer and emit Markdown for complete HTML tables.

    Returns (output_to_send, remaining_buffer).
    """
    if "<table" not in buffer.lower():
        return buffer, ""

    output = ""
    remaining = buffer

    while True:
        match = re.search(r"<table.*?</table>", remaining, re.DOTALL | re.IGNORECASE)
        if not match:
            break
        before = remaining[: match.start()]
        table_md = html_tables_to_md(match.group(0))
        output += before + table_md
        remaining = remaining[match.end() :]

    # If no incomplete table is pending, flush remaining text
    if "<table" not in remaining.lower():
        output += remaining
        remaining = ""

    return output, remaining


def render_pdf_to_pngs(pdf_path: Path, out_dir: Path) -> int:
    """Render PDF pages to PNG images.

    Returns number of pages rendered.
    """
    doc = fitz.open(pdf_path)
    try:
        zoom = RENDER_DPI / 72.0
        mat = fitz.Matrix(zoom, zoom)
        for i, page in enumerate(doc, start=1):
            pix = page.get_pixmap(matrix=mat, alpha=False)
            pix.save(out_dir / f"page-{i:04d}.png")
        return doc.page_count
    finally:
        doc.close()


def save_image_as_page(src: Path, out_dir: Path) -> int:
    """Convert single image to PNG page."""
    with Image.open(src) as im:
        im = im.convert("RGB")
        im.save(out_dir / "page-0001.png", format="PNG")
    return 1


async def ollama_ocr(
    image_path: Path,
    strategy: str = "vision",
    model: str | None = None,
    prompt: str | None = None,
) -> str:
    """Extract text from image using Ollama vision model.

    Args:
        image_path: Path to image file
        strategy: "auto" (LightOnOCR) or "vision" (direct vision model)
        model: Override default model
        prompt: Override default prompt

    Returns:
        Extracted markdown text
    """
    img_b64 = base64.b64encode(image_path.read_bytes()).decode("ascii")
    timeout = httpx.Timeout(connect=10.0, read=600.0, write=60.0, pool=10.0)

    # Determine model and prompt based on strategy
    if strategy == "auto":
        use_model = model or COMPLEX_MODEL
        use_prompt = prompt or OCR_PROMPT
    else:  # vision
        use_model = model or VISION_MODEL
        use_prompt = prompt or VISION_PROMPT

    payload = {
        "model": use_model,
        "prompt": use_prompt,
        "images": [img_b64],
        "stream": True,
        "options": {"temperature": 0.1, "num_predict": 16384},
    }

    final_text = ""
    html_buffer = ""

    async with httpx.AsyncClient(timeout=timeout) as client:
        async with client.stream("POST", f"{OLLAMA_URL}/api/generate", json=payload) as resp:
            if resp.status_code != 200:
                body = (await resp.aread()).decode("utf-8", errors="replace")
                raise RuntimeError(f"OCR HTTP {resp.status_code}: {body[:500]}")

            async for line in resp.aiter_lines():
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue

                chunk = obj.get("response", "")
                if chunk:
                    html_buffer += chunk
                    to_send, html_buffer = flush_html_buffer(html_buffer)
                    if to_send:
                        final_text += to_send

                if obj.get("done"):
                    break

    # Flush any remaining HTML buffer
    if html_buffer:
        final_text += html_tables_to_md(html_buffer)

    return final_text


async def convert_pdf(
    pdf_path: Path,
    output_dir: Path,
    strategy: str = "vision",
    progress_callback: Callable | None = None,
) -> ConversionResult:
    """Convert PDF to Markdown using vision models.

    Args:
        pdf_path: Input PDF file
        output_dir: Directory for intermediate PNGs and output
        strategy: "auto" or "vision"
        progress_callback: Optional callback(page_number, total_pages, stage)

    Returns:
        ConversionResult with markdown and page count
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Render PDF to images
    if progress_callback:
        await progress_callback(0, 0, "rendering")

    n_pages = render_pdf_to_pngs(pdf_path, output_dir)

    # Process each page with OCR
    all_markdown: list[str] = []

    for page_num in range(1, n_pages + 1):
        if progress_callback:
            await progress_callback(page_num, n_pages, "ocr")

        img_path = output_dir / f"page-{page_num:04d}.png"
        md_path = output_dir / f"page-{page_num:04d}.md"

        # Skip if already cached
        if md_path.exists():
            text = md_path.read_text(encoding="utf-8")
        else:
            text = await ollama_ocr(img_path, strategy=strategy)
            md_path.write_text(text, encoding="utf-8")

        all_markdown.append(f"<!-- Page {page_num} -->\n\n{text}")

    full_markdown = "\n\n---\n\n".join(all_markdown)

    # Save combined output
    (output_dir / "output.md").write_text(full_markdown, encoding="utf-8")

    return ConversionResult(
        markdown=full_markdown,
        pages=n_pages,
        strategy=strategy,
    )


async def convert_image(
    image_path: Path,
    output_dir: Path,
    strategy: str = "vision",
) -> ConversionResult:
    """Convert single image to Markdown.

    Args:
        image_path: Input image file (PNG, JPG, etc.)
        output_dir: Directory for output
        strategy: "auto" or "vision"

    Returns:
        ConversionResult with markdown
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Convert to PNG page
    save_image_as_page(image_path, output_dir)

    # OCR the image
    img_path = output_dir / "page-0001.png"
    md_path = output_dir / "page-0001.md"

    if md_path.exists():
        text = md_path.read_text(encoding="utf-8")
    else:
        text = await ollama_ocr(img_path, strategy=strategy)
        md_path.write_text(text, encoding="utf-8")

    (output_dir / "output.md").write_text(text, encoding="utf-8")

    return ConversionResult(markdown=text, pages=1, strategy=strategy)
