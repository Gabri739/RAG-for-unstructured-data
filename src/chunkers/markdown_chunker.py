"""Chunker strutturato per documenti Docling e Markdown:
  - Chunking per intestazioni (mantenendo la gerarchia `#` -> `##` -> `###` in ogni chunk).
  - Separazione atomica delle tabelle dal testo discorsivo.
  - Per le tabelle lunghe (`rows > max_table_rows`): suddivisione per blocchi di righe con
    ripetizione dell'intestazione di sezione, del titolo della tabella e dell'header delle
    colonne in ciascun chunk. Ogni chunk di tabella resta inoltre sotto `max_table_tokens`.
  - Testate, piè di pagina e numeri di pagina vengono esclusi (riconosciuti per ripetizione).
  - Codici di riferimento opzionali (`reference_patterns`, es. `[GRI 205-1]`): diventano il campo
    `reference_codes` dei chunk che seguono, fino al marcatore o al titolo successivo.
  - Metadati completi per chunk:
      * `doc_id`, `source_file`, `chunk_id` (leggibile) e `point_id` (UUIDv5 deterministico per Qdrant)
      * `pages`, `page_start`, `page_end`, `page_range` (pagine del singolo chunk)
      * `section`, `reference_codes`, `chunk_type` (`text` / `table`), `image_paths`
      * `possible_incomplete_table`, `empty_cells_ratio` (intera tabella),
        `chunk_empty_cells_ratio` (singolo blocco) e `empty_columns` (colonne >= 80% vuote, min 3 righe).
"""

from __future__ import annotations

import json
import re
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Sequence

from src.extractors.running_text import find_running_text_refs

from .reference_codes import ReferencePattern, format_marker, split_by_markers
from .table_utils import (
    TableBlock,
    approx_token_count,
    extract_table_block_from_docling,
    find_all_tables,
    split_large_table,
)

if TYPE_CHECKING:
    from docling_core.types.doc import DoclingDocument

# Testo composto solo da un'etichetta tra parentesi quadre (es. "[Rif. 12]"): non è un titolo
_BRACKETED_RE = re.compile(r"^\[[^\[\]]{1,80}\]$")

CHUNK_UUID_NAMESPACE = uuid.UUID("6ba7b812-9dad-11d1-80b4-00c04fd430c8")  # NAMESPACE_OID


def _format_page_range(pages: list[int]) -> str | None:
    """Restituisce `'43'` per pagina singola o `'43-44'` per intervallo di pagine."""
    if not pages:
        return None
    lo, hi = min(pages), max(pages)
    return str(lo) if lo == hi else f"{lo}-{hi}"


def _format_section_path(headers: list[str]) -> str:
    """Formatta lo stack di intestazioni come percorso leggibile (es. `'Cap 1 > Sezione 1.2'`)."""
    cleaned = [re.sub(r"^#+\s*", "", h).strip() for h in headers if h.strip()]
    return " > ".join(cleaned)


def make_point_id(chunk_id: str) -> str:
    """Genera un UUIDv5 deterministico da `chunk_id` (richiesto da Qdrant come ID del punto)."""
    return str(uuid.uuid5(CHUNK_UUID_NAMESPACE, chunk_id))


def _load_manifest_image_map(manifest: Path | dict[str, Any] | None) -> dict[int, list[str]]:
    """Carica da `table_images.json` (se disponibile) la mappa `{table_index: [image_paths]}`."""
    if manifest is None:
        return {}
    data: dict[str, Any]
    if isinstance(manifest, Path):
        if not manifest.exists():
            return {}
        data = json.loads(manifest.read_text(encoding="utf-8"))
    elif isinstance(manifest, dict):
        data = manifest
    else:
        return {}

    result: dict[int, list[str]] = {}
    for entry in data.get("tables", []):
        t_idx = entry.get("table_index")
        paths = entry.get("image_paths")
        if isinstance(t_idx, int) and isinstance(paths, list):
            result[t_idx] = [str(p) for p in paths]
    return result


@dataclass
class _Paragraph:
    """Paragrafo di testo con la pagina di ciascuna parola (`word_pages` vuota se non nota)."""

    text: str
    word_pages: list[int] = field(default_factory=list)


def _page_at(prov: list[Any], offset: int) -> int | None:
    """Pagina del carattere `offset`: un paragrafo a cavallo di due pagine ha più `prov`."""
    page: int | None = None
    for p in prov:
        start = p.charspan[0] if getattr(p, "charspan", None) else 0
        if page is None or offset >= start:
            page = p.page_no
    return page


def _build_paragraph(text: str, offset: int, prov: list[Any], prefix: str = "") -> _Paragraph:
    """Crea un `_Paragraph` da un segmento di testo che inizia a `offset` nell'elemento Docling."""
    pages = [_page_at(prov, offset + m.start()) for m in re.finditer(r"\S+", text)]
    full_text = f"{prefix} {text.strip()}" if prefix else text.strip()
    if not pages or any(p is None for p in pages):
        return _Paragraph(full_text)
    return _Paragraph(full_text, [pages[0]] * len(prefix.split()) + pages)


@dataclass
class Chunk:
    """Rappresenta un chunk pronto per l'embedding e il salvataggio nel Vector DB."""

    content: str
    headers: list[str] = field(default_factory=list)
    chunk_type: str = "text"  # "text" oppure "table"
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serializza il chunk in dizionario JSON-compatibile (metadati ricchi completi)."""
        return asdict(self)

    def to_vector_metadata(self, *, flat: bool = False) -> dict[str, Any]:
        """Restituisce i metadati pronti per l'indicizzazione nel Vector DB.

        - `flat=False` (default, per Qdrant): mantiene liste e dizionari nativi.
        - `flat=True` (per ChromaDB): converte le liste in stringhe JSON ed elimina le chiavi `None`.
        """
        if not flat:
            return dict(self.metadata)

        flat_meta: dict[str, str | int | float | bool] = {}
        for k, v in self.metadata.items():
            if v is None:
                continue
            if isinstance(v, (str, int, float, bool)):
                flat_meta[k] = v
            elif isinstance(v, (list, dict)):
                flat_meta[k] = json.dumps(v, ensure_ascii=False)
            else:
                flat_meta[k] = str(v)
        return flat_meta


class MarkdownChunker:
    """Divide un `DoclingDocument` (o un testo Markdown) in chunk per intestazioni e per righe tabella."""

    def __init__(
        self,
        headers_to_split_on: list[str] | None = None,
        chunk_size: int = 1500,
        chunk_overlap: int = 200,
        max_table_rows: int = 25,
        max_table_tokens: int = 900,
        count_tokens: Callable[[str], int] | None = None,
        reference_patterns: Sequence[ReferencePattern] = (),
        empty_cells_threshold: float = 0.35,
        col_empty_threshold: float = 0.80,
        min_rows_for_col_check: int = 3,
        preserve_tables: bool = True,
    ) -> None:
        """Inizializza il chunker.

        Args:
            headers_to_split_on: Marcatori Markdown su cui spezzare le sezioni (default `["#", "##", "###", "####"]`).
            chunk_size: Dimensione massima (in caratteri) di un chunk di testo prima dello split con overlap.
            chunk_overlap: Sovrapposizione (in caratteri) tra chunk di testo consecutivi della stessa sezione.
            max_table_rows: Numero massimo di righe dati per chunk di tabella; oltre questa soglia la tabella
                viene divisa in più chunk per righe, ripetendo header colonne e titolo in ciascuno.
            max_table_tokens: Numero massimo di token per chunk di tabella (default 900, così il chunk
                entra per intero sia nell'embedding sia nei 1024 token di un reranker; `0` = nessun limite).
            count_tokens: Funzione che conta i token (idealmente il tokenizer del modello di embedding);
                se `None` usa una stima prudente basata sui caratteri.
            reference_patterns: Formati di codici di riferimento da riconoscere (default nessuno).
            empty_cells_threshold: Frazione globale di celle vuote (`0.0 - 1.0`) oltre la quale una tabella
                viene marcata con `possible_incomplete_table = True`.
            col_empty_threshold: Frazione di celle vuote (`0.0 - 1.0`) in una singola colonna per segnalarla
                in `empty_columns` (default `0.80`).
            min_rows_for_col_check: Numero minimo di righe dati perché il controllo `empty_columns` per colonna
                sia attivo (default `3`, evita falsi positivi su tabelle di 1-2 righe).
            preserve_tables: Se True (default), estrae le tabelle come chunk dedicati di tipo `"table"`.
        """
        raw_headers = headers_to_split_on or ["#", "##", "###", "####"]
        self._raw_headers = sorted({h.strip() for h in raw_headers if h.strip()}, key=len)
        self._allowed_levels = {len(h) for h in self._raw_headers}
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.max_table_rows = max_table_rows
        self.max_table_tokens = max_table_tokens
        self.count_tokens = count_tokens or approx_token_count
        self.reference_patterns = tuple(reference_patterns)
        self.empty_cells_threshold = empty_cells_threshold
        self.col_empty_threshold = col_empty_threshold
        self.min_rows_for_col_check = min_rows_for_col_check
        self.preserve_tables = preserve_tables

    # ------------------------------------------------------------------
    # Costruzione dei chunk (testo e tabella)
    # ------------------------------------------------------------------

    def _assign_ids(
        self,
        chunks: list[Chunk],
        doc_id: str,
        source_file: str | None,
    ) -> list[Chunk]:
        """Assegna a ogni chunk `doc_id`, `source_file`, `chunk_index`, `chunk_id` e `point_id` (UUIDv5)."""
        for idx, c in enumerate(chunks):
            chunk_id = f"{doc_id}__chunk_{idx:04d}"
            c.metadata["doc_id"] = doc_id
            c.metadata["source_file"] = source_file
            c.metadata["chunk_index"] = idx
            c.metadata["chunk_id"] = chunk_id
            c.metadata["point_id"] = make_point_id(chunk_id)
        return chunks

    @staticmethod
    def _header_prefix(header_stack: list[str], reference_codes: list[str]) -> str:
        """Intestazioni di sezione (ed eventuali codici) ripetute all'inizio di ogni chunk."""
        lines = list(header_stack)
        if reference_codes:
            lines.append(format_marker(reference_codes))
        return "\n".join(lines)

    def _build_text_chunks(
        self,
        paragraphs: list[_Paragraph],
        header_stack: list[str],
        reference_codes: list[str] | None = None,
    ) -> list[Chunk]:
        """Costruisce uno o più chunk di testo per una sezione, ripetendo le intestazioni in ogni parte.

        Ogni parte riporta solo le pagine delle parole che contiene.
        """
        codes = list(reference_codes or [])
        clean_paras = [p for p in paragraphs if p.text.strip()]
        if not clean_paras:
            return []

        header_prefix = self._header_prefix(header_stack, codes)
        full_body = "\n\n".join(p.text.strip() for p in clean_paras)
        header_budget = len(header_prefix) + 2 if header_prefix else 0
        max_body_chars = max(200, self.chunk_size - header_budget)

        words = full_body.split()
        word_pages: list[int] = []
        for p in clean_paras:
            word_pages.extend(p.word_pages)
        if len(word_pages) != len(words):
            # Pagine non note per tutte le parole: attribuisci a ogni parte tutte quelle note
            word_pages = []
        known_pages = {pg for p in clean_paras for pg in p.word_pages}

        if len(full_body) <= max_body_chars:
            ranges = [(0, len(words))]
            body_parts = [full_body]
        else:
            ranges = self._split_word_ranges(words, max_body_chars, self.chunk_overlap)
            body_parts = [" ".join(words[a:b]) for a, b in ranges]
        total_parts = len(body_parts)
        section_str = _format_section_path(header_stack)

        chunks: list[Chunk] = []
        for idx, (body, (start_w, end_w)) in enumerate(zip(body_parts, ranges), start=1):
            content = f"{header_prefix}\n\n{body}" if header_prefix else body
            sorted_pages = sorted(set(word_pages[start_w:end_w]) if word_pages else known_pages)
            page_start = sorted_pages[0] if sorted_pages else None
            page_end = sorted_pages[-1] if sorted_pages else None
            page_range = _format_page_range(sorted_pages)
            chunks.append(
                Chunk(
                    content=content,
                    headers=list(header_stack),
                    chunk_type="text",
                    metadata={
                        "chunk_type": "text",
                        "section": section_str,
                        "reference_codes": list(codes),
                        "pages": sorted_pages,
                        "page_start": page_start,
                        "page_end": page_end,
                        "page_range": page_range,
                        "image_paths": [],
                        "possible_incomplete_table": False,
                        "empty_columns": [],
                        "part": idx,
                        "total_parts": total_parts,
                    },
                )
            )
        return chunks

    def _split_word_ranges(
        self,
        words: list[str],
        max_chars: int,
        overlap_chars: int,
    ) -> list[tuple[int, int]]:
        """Divide le parole in intervalli `<= max_chars` caratteri con `overlap_chars` di sovrapposizione."""
        ranges: list[tuple[int, int]] = []
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

            ranges.append((start_w, end_w))
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

        return ranges

    def _split_table(
        self,
        table: TableBlock,
        header_stack: list[str],
        reference_codes: list[str] | None = None,
    ) -> list[Chunk]:
        """Divide la tabella per righe e token e crea un chunk per ogni parte."""
        codes = list(reference_codes or [])
        # Margine per "(Parte k/N, righe a-b di c)" aggiunto al titolo di ogni parte
        extra = self.count_tokens(self._header_prefix(header_stack, codes)) + 20
        parts = split_large_table(
            table,
            self.max_table_rows,
            max_tokens=self.max_table_tokens,
            count_tokens=self.count_tokens,
            extra_tokens=extra,
        )
        return [self._create_table_chunk(part, header_stack, codes) for part in parts]

    def _create_table_chunk(
        self,
        table: TableBlock,
        header_stack: list[str],
        reference_codes: list[str] | None = None,
    ) -> Chunk:
        """Crea un chunk per una tabella (o porzione di righe di una tabella), ripetendo
        gerarchia di intestazioni, titolo della tabella e header delle colonne.
        """
        codes = list(reference_codes or [])
        table_md = table.to_markdown(include_title=True, include_footnotes=True)
        header_prefix = self._header_prefix(header_stack, codes)
        content = f"{header_prefix}\n\n{table_md}" if header_prefix else table_md

        sorted_pages = sorted(set(table.pages))
        page_start = sorted_pages[0] if sorted_pages else None
        page_end = sorted_pages[-1] if sorted_pages else None
        page_range = _format_page_range(sorted_pages)
        section_str = _format_section_path(header_stack)
        incomplete = table.is_possibly_incomplete(
            empty_ratio_threshold=self.empty_cells_threshold,
            col_empty_threshold=self.col_empty_threshold,
            min_rows_for_col_check=self.min_rows_for_col_check,
        )
        empty_cols = list(table.empty_columns) if table.empty_columns is not None else []

        return Chunk(
            content=content,
            headers=list(header_stack),
            chunk_type="table",
            metadata={
                "chunk_type": "table",
                "section": section_str,
                "reference_codes": codes,
                "pages": sorted_pages,
                "page_start": page_start,
                "page_end": page_end,
                "page_range": page_range,
                "image_paths": list(table.image_paths),
                "possible_incomplete_table": incomplete,
                "empty_cells_ratio": table.empty_cells_ratio,
                "chunk_empty_cells_ratio": table.chunk_empty_cells_ratio,
                "empty_columns": empty_cols,
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

    def chunk_document(
        self,
        doc: DoclingDocument,
        *,
        doc_id: str | None = None,
        source_file: str | None = None,
        table_images_manifest: Path | dict[str, Any] | None = None,
    ) -> list[Chunk]:
        """Esegue il chunking nativo su un `DoclingDocument` (dopo fusione tabelle e salvataggio immagini).

        Args:
            doc: `DoclingDocument` già processato da `fuse_document_tables`.
            doc_id: Identificativo univoco del documento (default: `doc.name` o `"document"`).
            source_file: Nome del file sorgente (es. `"source.pdf"`).
            table_images_manifest: Percorso (o dict) di `table_images.json` usato come fonte
                principale per `image_paths`, con fallback a `table.meta["rag__table_images"]`.
        """
        from docling_core.types.doc import DocItemLabel, SectionHeaderItem, TableItem
        from docling_core.types.doc.common.content_layer import ContentLayer

        resolved_doc_id = doc_id or getattr(doc, "name", None) or "document"
        resolved_source = source_file
        if resolved_source is None and getattr(doc, "origin", None) is not None:
            resolved_source = getattr(doc.origin, "filename", None)

        manifest_map = _load_manifest_image_map(table_images_manifest)

        owned_by_tables: set[str] = set()
        table_idx_by_ref: dict[str, int] = {}
        for idx, t in enumerate(doc.tables):
            table_idx_by_ref[t.self_ref] = idx
            for ref in list(getattr(t, "captions", [])) + list(getattr(t, "footnotes", [])):
                if hasattr(ref, "cref"):
                    owned_by_tables.add(ref.cref)

        header_by_level: dict[int, str] = {}
        current_paragraphs: list[_Paragraph] = []
        current_codes: list[str] = []
        codes_have_content = False
        running_refs = find_running_text_refs(doc)
        last_short_paragraph: str | None = None
        all_chunks: list[Chunk] = []

        def current_header_stack() -> list[str]:
            return [header_by_level[k] for k in sorted(header_by_level)]

        def flush_text() -> None:
            nonlocal current_paragraphs
            if current_paragraphs:
                all_chunks.extend(
                    self._build_text_chunks(current_paragraphs, current_header_stack(), current_codes)
                )
                current_paragraphs = []

        def set_codes(codes: list[str]) -> None:
            """Attiva dei codici di riferimento: valgono per il contenuto successivo."""
            nonlocal current_codes, codes_have_content
            flush_text()
            if current_codes and not codes_have_content:
                # Due marcatori consecutivi valgono entrambi per lo stesso contenuto
                current_codes = current_codes + [c for c in codes if c not in current_codes]
            else:
                current_codes = list(codes)
            codes_have_content = False

        # Include anche header/footer di pagina: un marcatore può essere finito lì
        layers = {ContentLayer.BODY, ContentLayer.FURNITURE}
        for item, _ in doc.iterate_items(included_content_layers=layers):
            ref = getattr(item, "self_ref", None)
            if ref in owned_by_tables:
                continue

            label = getattr(item, "label", None)
            segments = split_by_markers(getattr(item, "text", "") or "", self.reference_patterns)

            if segments and all(codes for _, _, codes in segments):
                for _, _, codes in segments:
                    set_codes(codes)
                continue

            if (
                ref in running_refs
                or label in {DocItemLabel.PAGE_HEADER, DocItemLabel.PAGE_FOOTER}
                or getattr(item, "content_layer", ContentLayer.BODY) == ContentLayer.FURNITURE
            ):
                continue

            is_bracketed = bool(_BRACKETED_RE.match((getattr(item, "text", "") or "").strip()))

            # 1. Intestazione di sezione o titolo documento
            if not is_bracketed and (
                isinstance(item, SectionHeaderItem)
                or label in {DocItemLabel.TITLE, DocItemLabel.SECTION_HEADER}
            ):
                text = getattr(item, "text", "").strip()
                if len(re.sub(r"\W", "", text)) < 2:
                    continue  # rumore (es. un titolo composto dalla sola lettera "x")
                flush_text()
                if codes_have_content:
                    current_codes = []
                    codes_have_content = False
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
                    manifest_image_paths=manifest_map.get(t_idx) if t_idx is not None else None,
                    empty_ratio_threshold=self.empty_cells_threshold,
                    col_empty_threshold=self.col_empty_threshold,
                    min_rows_for_col_check=self.min_rows_for_col_check,
                )
                all_chunks.extend(self._split_table(t_block, current_header_stack(), current_codes))
                codes_have_content = True
                last_short_paragraph = None
                continue

            # 3. Paragrafo di testo / elemento di lista
            prov = [p for p in getattr(item, "prov", []) if hasattr(p, "page_no")]
            for seg_idx, (offset, seg_text, codes) in enumerate(segments):
                if codes:
                    set_codes(codes)
                    continue
                text = seg_text.strip()
                prefix = ""
                if (
                    label == DocItemLabel.LIST_ITEM
                    and seg_idx == 0
                    and not text.startswith(("-", "•", "*"))
                ):
                    prefix = getattr(item, "marker", "-") or "-"
                current_paragraphs.append(_build_paragraph(seg_text, offset, prov, prefix))
                codes_have_content = True
                # Una nota o un'etichetta tra parentesi non è il titolo della tabella che segue
                if label != DocItemLabel.FOOTNOTE and not is_bracketed:
                    last_short_paragraph = text if len(text) <= 160 else None

        flush_text()
        return self._assign_ids(all_chunks, doc_id=resolved_doc_id, source_file=resolved_source)

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

    def chunk(
        self,
        markdown_content: str,
        *,
        doc_id: str = "markdown_doc",
        source_file: str | None = None,
    ) -> list[Chunk]:
        """Divide una stringa Markdown in chunk per intestazioni e tabelle."""
        sections = self._split_markdown_by_headers(markdown_content)
        all_chunks: list[Chunk] = []

        for header_stack, section_body in sections:
            if not self.preserve_tables:
                all_chunks.extend(self._build_text_chunks([_Paragraph(section_body)], header_stack))
                continue
            all_chunks.extend(self._process_tables_in_markdown_section(section_body, header_stack))

        return self._assign_ids(all_chunks, doc_id=doc_id, source_file=source_file)

    def _process_tables_in_markdown_section(
        self,
        content: str,
        header_stack: list[str],
    ) -> list[Chunk]:
        """Estrae testo e tabelle da una sezione Markdown, dividendo le tabelle lunghe per righe."""
        from .table_utils import find_empty_columns

        tables = find_all_tables(content)
        if not tables:
            return self._build_text_chunks([_Paragraph(content)], header_stack)

        chunks: list[Chunk] = []
        lines = content.split("\n")
        last_end = 0

        for table in tables:
            text_lines = lines[last_end : table.start_line]
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
                chunks.extend(self._build_text_chunks([_Paragraph(text_block)], header_stack))

            if not table.title and header_stack:
                table.title = re.sub(r"^#+\s*", "", header_stack[-1]).strip()

            # Ricalcola empty_columns e flag sull'intera tabella prima dello split per righe
            table.empty_columns = find_empty_columns(
                table.headers,
                table.value_rows,
                col_empty_threshold=self.col_empty_threshold,
                min_rows=self.min_rows_for_col_check,
            )
            table.table_possibly_incomplete = table.is_possibly_incomplete(
                empty_ratio_threshold=self.empty_cells_threshold,
                col_empty_threshold=self.col_empty_threshold,
                min_rows_for_col_check=self.min_rows_for_col_check,
            )

            chunks.extend(self._split_table(table, header_stack))

            last_end = table.end_line + 1

        tail_lines = lines[last_end:]
        tail_block = "\n".join(line for line in tail_lines if line.strip())
        if tail_block.strip():
            chunks.extend(self._build_text_chunks([_Paragraph(tail_block)], header_stack))

        return chunks
