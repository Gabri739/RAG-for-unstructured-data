"""Chunker strutturato per documenti Docling e Markdown:
  - Chunking per intestazioni (mantenendo la gerarchia `#` -> `##` -> `###` in ogni chunk).
  - Separazione atomica delle tabelle dal testo discorsivo.
  - Per le tabelle lunghe (`rows > max_table_rows`): suddivisione per blocchi di righe con
    ripetizione dell'intestazione di sezione, del titolo della tabella e dell'header delle
    colonne in ciascun chunk.
  - Metadati completi per chunk: pagina/intervallo di pagine, sezione, tipo (`text` / `table`),
    percorsi delle immagini (`image_paths`) e flag `possible_incomplete_table`.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any

from .table_utils import (
    TableBlock,
    extract_table_block_from_docling,
    find_all_tables,
    split_large_table,
)

if TYPE_CHECKING:
    from docling_core.types.doc import DoclingDocument


def _format_page_range(pages: list[int]) -> str | None:
    """Restituisce `'43'` per pagina singola o `'43-44'` per intervallo di pagine."""
    if not pages:
        return None
    lo, hi = min(pages), max(pages)
    return str(lo) if lo == hi else f"{lo}-{hi}"


def _format_section_path(headers: list[str]) -> str:
    """Formatta lo stack di intestazioni come percorso leggibile (es. `'Cap 1 > [GRI 204-1]'`)."""
    cleaned = [re.sub(r"^#+\s*", "", h).strip() for h in headers if h.strip()]
    return " > ".join(cleaned)


@dataclass
class Chunk:
    """Rappresenta un chunk pronto per l'embedding e il salvataggio nel Vector DB."""

    content: str
    headers: list[str] = field(default_factory=list)
    chunk_type: str = "text"  # "text" oppure "table"
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serializza il chunk in dizionario JSON-compatibile."""
        return asdict(self)


class MarkdownChunker:
    """Divide un `DoclingDocument` (o un testo Markdown) in chunk per intestazioni e per righe tabella."""

    def __init__(
        self,
        headers_to_split_on: list[str] | None = None,
        chunk_size: int = 1500,
        chunk_overlap: int = 200,
        max_table_rows: int = 25,
        empty_cells_threshold: float = 0.35,
        preserve_tables: bool = True,
    ) -> None:
        """Inizializza il chunker.

        Args:
            headers_to_split_on: Marcatori Markdown su cui spezzare le sezioni (default `["#", "##", "###", "####"]`).
            chunk_size: Dimensione massima (in caratteri) di un chunk di testo prima dello split con overlap.
            chunk_overlap: Sovrapposizione (in caratteri) tra chunk di testo consecutivi della stessa sezione.
            max_table_rows: Numero massimo di righe dati per chunk di tabella; oltre questa soglia la tabella
                viene divisa in più chunk per righe, ripetendo header colonne e titolo in ciascuno.
            empty_cells_threshold: Frazione di celle vuote (`0.0 - 1.0`) oltre la quale una tabella viene
                marcata con `possible_incomplete_table = True`.
            preserve_tables: Se True (default), estrae le tabelle come chunk dedicati di tipo `"table"`.
        """
        raw_headers = headers_to_split_on or ["#", "##", "###", "####"]
        self._raw_headers = sorted({h.strip() for h in raw_headers if h.strip()}, key=len)
        self._allowed_levels = {len(h) for h in self._raw_headers}
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.max_table_rows = max_table_rows
        self.empty_cells_threshold = empty_cells_threshold
        self.preserve_tables = preserve_tables

    # ------------------------------------------------------------------
    # Costruzione dei chunk (testo e tabella)
    # ------------------------------------------------------------------

    def _build_text_chunks(
        self,
        paragraphs: list[str],
        header_stack: list[str],
        pages: list[int],
    ) -> list[Chunk]:
        """Costruisce uno o più chunk di testo per una sezione, ripetendo le intestazioni in ogni parte."""
        clean_paras = [p.strip() for p in paragraphs if p and p.strip()]
        if not clean_paras:
            return []

        header_prefix = "\n".join(header_stack)
        full_body = "\n\n".join(clean_paras)
        header_budget = len(header_prefix) + 2 if header_prefix else 0
        max_body_chars = max(200, self.chunk_size - header_budget)

        body_parts = self._split_text_body(full_body, max_body_chars, self.chunk_overlap)
        total_parts = len(body_parts)
        sorted_pages = sorted(set(pages))
        page_range = _format_page_range(sorted_pages)
        section_str = _format_section_path(header_stack)

        chunks: list[Chunk] = []
        for idx, body in enumerate(body_parts, start=1):
            content = f"{header_prefix}\n\n{body}" if header_prefix else body
            chunks.append(
                Chunk(
                    content=content,
                    headers=list(header_stack),
                    chunk_type="text",
                    metadata={
                        "chunk_type": "text",
                        "section": section_str,
                        "pages": sorted_pages,
                        "page_range": page_range,
                        "image_paths": [],
                        "possible_incomplete_table": False,
                        "part": idx,
                        "total_parts": total_parts,
                    },
                )
            )
        return chunks

    def _split_text_body(self, text: str, max_chars: int, overlap_chars: int) -> list[str]:
        """Divide un testo lungo in blocchi `<= max_chars` con `overlap_chars` di sovrapposizione."""
        if len(text) <= max_chars:
            return [text]

        words = text.split()
        if not words:
            return []

        parts: list[str] = []
        start_w = 0
        n_words = len(words)

        while start_w < n_words:
            curr_len = 0
            end_w = start_w
            while end_w < n_words:
                w_len = len(words[end_w]) + (1 if end_w > start_w else 0)
                if curr_len + w_len > max_chars and end_w > start_w:
                    break
                curr_len += w_len
                end_w += 1

            parts.append(" ".join(words[start_w:end_w]))
            if end_w >= n_words:
                break

            overlap_len = 0
            back_w = end_w
            while back_w > start_w + 1:
                prev_len = len(words[back_w - 1]) + 1
                if overlap_len + prev_len > overlap_chars:
                    break
                overlap_len += prev_len
                back_w -= 1

            start_w = back_w

        return parts

    def _create_table_chunk(self, table: TableBlock, header_stack: list[str]) -> Chunk:
        """Crea un chunk per una tabella (o porzione di righe di una tabella), ripetendo
        gerarchia di intestazioni, titolo della tabella e header delle colonne.
        """
        table_md = table.to_markdown(include_title=True, include_footnotes=True)
        header_prefix = "\n".join(header_stack)
        content = f"{header_prefix}\n\n{table_md}" if header_prefix else table_md

        sorted_pages = sorted(set(table.pages))
        page_range = _format_page_range(sorted_pages)
        section_str = _format_section_path(header_stack)
        incomplete = table.is_possibly_incomplete(self.empty_cells_threshold)

        return Chunk(
            content=content,
            headers=list(header_stack),
            chunk_type="table",
            metadata={
                "chunk_type": "table",
                "section": section_str,
                "pages": sorted_pages,
                "page_range": page_range,
                "image_paths": list(table.image_paths),
                "possible_incomplete_table": incomplete,
                "empty_cells_ratio": table.empty_cells_ratio,
                "table_index": table.table_index,
                "table_title": table.title,
                "row_count": len(table.rows),
                "total_table_rows": table.total_rows,
                "row_range": f"{table.row_start}-{table.row_end}" if table.rows else "0-0",
                "column_count": len(table.headers),
                "has_caption": table.title is not None,
                "part": table.part,
                "total_parts": table.total_parts,
            },
        )

    # ------------------------------------------------------------------
    # Modalità 1: Chunking diretto da DoclingDocument (Step 4 & 5)
    # ------------------------------------------------------------------

    def chunk_document(self, doc: DoclingDocument) -> list[Chunk]:
        """Esegue il chunking nativo su un `DoclingDocument` (dopo fusione tabelle e salvataggio immagini).

        - Mantiene lo stack gerarchico delle intestazioni (`SectionHeaderItem`, `TitleItem`).
        - Raggruppa i paragrafi e le liste per sezione, dividendoli se superano `chunk_size`.
        - Per ogni `TableItem`, estrae titolo, header colonne, pagine e `image_paths`, e se
          la tabella supera `max_table_rows` righe la divide in più chunk ripetendo titolo e
          header delle colonne in ciascun chunk.
        """
        from docling_core.types.doc import DocItemLabel, SectionHeaderItem, TableItem

        owned_by_tables: set[str] = set()
        table_idx_by_ref: dict[str, int] = {}
        for idx, t in enumerate(doc.tables):
            table_idx_by_ref[t.self_ref] = idx
            for ref in list(getattr(t, "captions", [])) + list(getattr(t, "footnotes", [])):
                if hasattr(ref, "cref"):
                    owned_by_tables.add(ref.cref)

        header_by_level: dict[int, str] = {}
        current_paragraphs: list[str] = []
        current_pages: list[int] = []
        last_short_paragraph: str | None = None
        all_chunks: list[Chunk] = []

        def current_header_stack() -> list[str]:
            return [header_by_level[k] for k in sorted(header_by_level)]

        def flush_text() -> None:
            nonlocal current_paragraphs, current_pages
            if current_paragraphs:
                all_chunks.extend(
                    self._build_text_chunks(current_paragraphs, current_header_stack(), current_pages)
                )
                current_paragraphs = []
                current_pages = []

        for item, _ in doc.iterate_items():
            ref = getattr(item, "self_ref", None)
            if ref in owned_by_tables:
                continue

            label = getattr(item, "label", None)
            if label in {DocItemLabel.PAGE_HEADER, DocItemLabel.PAGE_FOOTER}:
                continue

            item_pages = [p.page_no for p in getattr(item, "prov", []) if hasattr(p, "page_no")]

            # 1. Intestazione di sezione o titolo documento
            if isinstance(item, SectionHeaderItem) or label in {
                DocItemLabel.TITLE,
                DocItemLabel.SECTION_HEADER,
            }:
                text = getattr(item, "text", "").strip()
                if not text:
                    continue
                flush_text()
                level = getattr(item, "level", None)
                if level is None:
                    level = 1 if label == DocItemLabel.TITLE else 2
                else:
                    level = int(level)
                level = max(1, min(level, 6))
                for deeper in [k for k in header_by_level if k >= level]:
                    del header_by_level[deeper]
                header_by_level[level] = f"{'#' * level} {text}"
                last_short_paragraph = None
                continue

            # 2. Tabella
            if isinstance(item, TableItem):
                t_idx = table_idx_by_ref.get(item.self_ref)
                fallback_title = last_short_paragraph or (
                    current_header_stack()[-1] if current_header_stack() else None
                )
                flush_text()
                t_block = extract_table_block_from_docling(
                    item,
                    doc,
                    table_index=t_idx,
                    fallback_title=fallback_title,
                )
                for row_chunk_block in split_large_table(t_block, self.max_table_rows):
                    all_chunks.append(self._create_table_chunk(row_chunk_block, current_header_stack()))
                last_short_paragraph = None
                continue

            # 3. Paragrafo di testo / elemento di lista
            text = getattr(item, "text", "").strip()
            if not text:
                continue

            if label == DocItemLabel.LIST_ITEM:
                marker = getattr(item, "marker", "-") or "-"
                formatted = f"{marker} {text}" if not text.startswith(("-", "•", "*")) else text
            else:
                formatted = text

            current_paragraphs.append(formatted)
            current_pages.extend(item_pages)
            last_short_paragraph = text if len(text) <= 160 else None

        flush_text()
        return all_chunks

    # ------------------------------------------------------------------
    # Modalità 2: Chunking da stringa Markdown
    # ------------------------------------------------------------------

    def _split_markdown_by_headers(self, markdown_content: str) -> list[tuple[list[str], str]]:
        """Divide un testo Markdown in sezioni `(header_stack, section_body)`."""
        header_re = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
        header_by_level: dict[int, str] = {}
        current_lines: list[str] = []
        sections: list[tuple[list[str], str]] = []
        in_code_fence = False

        def current_stack() -> list[str]:
            return [header_by_level[k] for k in sorted(header_by_level)]

        def flush_section() -> None:
            nonlocal current_lines
            body = "\n".join(current_lines).strip()
            if body:
                sections.append((current_stack(), body))
            current_lines = []

        for line in markdown_content.splitlines():
            stripped = line.strip()
            if stripped.startswith("```"):
                in_code_fence = not in_code_fence
                current_lines.append(line)
                continue

            if not in_code_fence:
                m = header_re.match(stripped)
                if m and len(m.group(1)) in self._allowed_levels:
                    flush_section()
                    lvl = len(m.group(1))
                    title = m.group(2).strip()
                    for deeper in [k for k in header_by_level if k >= lvl]:
                        del header_by_level[deeper]
                    header_by_level[lvl] = f"{'#' * lvl} {title}"
                    continue

            current_lines.append(line)

        flush_section()
        return sections

    def chunk(self, markdown_content: str) -> list[Chunk]:
        """Divide una stringa Markdown in chunk per intestazioni e tabelle."""
        sections = self._split_markdown_by_headers(markdown_content)
        all_chunks: list[Chunk] = []

        for header_stack, section_body in sections:
            if not self.preserve_tables:
                all_chunks.extend(self._build_text_chunks([section_body], header_stack, pages=[]))
                continue
            all_chunks.extend(self._process_tables_in_markdown_section(section_body, header_stack))

        return all_chunks

    def _process_tables_in_markdown_section(
        self,
        content: str,
        header_stack: list[str],
    ) -> list[Chunk]:
        """Estrae testo e tabelle da una sezione Markdown, dividendo le tabelle lunghe per righe."""
        tables = find_all_tables(content)
        if not tables:
            return self._build_text_chunks([content], header_stack, pages=[])

        chunks: list[Chunk] = []
        lines = content.split("\n")
        last_end = 0

        for table in tables:
            text_lines = lines[last_end : table.start_line]
            # Se l'ultima riga non vuota prima della tabella è stata usata come caption, non duplicarla
            if table.title:
                while text_lines and not text_lines[-1].strip():
                    text_lines.pop()
                if text_lines:
                    candidate = re.sub(r"^\*\*|\*\*$", "", text_lines[-1].strip()).strip()
                    candidate = re.sub(r"^Tabella:\s*", "", candidate, flags=re.IGNORECASE).strip()
                    if candidate and candidate in table.title:
                        text_lines.pop()

            text_block = "\n".join(line for line in text_lines if line.strip())
            if text_block.strip():
                chunks.extend(self._build_text_chunks([text_block], header_stack, pages=[]))

            if not table.title and header_stack:
                table.title = re.sub(r"^#+\s*", "", header_stack[-1]).strip()

            for tc in split_large_table(table, self.max_table_rows):
                chunks.append(self._create_table_chunk(tc, header_stack))

            last_end = table.end_line + 1

        tail_lines = lines[last_end:]
        tail_block = "\n".join(line for line in tail_lines if line.strip())
        if tail_block.strip():
            chunks.extend(self._build_text_chunks([tail_block], header_stack, pages=[]))

        return chunks
