"""Utilities per l'estrazione, normalizzazione e suddivisione per righe delle tabelle.

Ordine garantito nella pipeline:
  1. Fusione tabelle multi-pagina e ricucitura righe spezzate (`table_fusion.py`)
  2. Estrazione di `TableBlock` e calcolo dei flag di completezza sull'INTERA tabella:
     - `table_empty_cells_ratio` (frazione globale di celle vuote)
     - `empty_columns` (colonne con >= 80% di celle vuote, calcolato solo se la tabella
       ha almeno `min_rows_for_col_check = 3` righe dati e ignorando colonne opzionali come "Note")
     - `possible_incomplete_table` (`table_empty_cells_ratio >= 0.35` oppure `bool(empty_columns)`)
  3. Divisione per righe e per token (`split_large_table`), che propaga a tutte le parti della
     tabella i flag calcolati sull'intera tabella e ripete titolo e header colonne in ogni parte.
     Il taglio cade, se possibile, all'inizio di un gruppo di righe; se cade dentro un gruppo,
     la parte successiva ripete le righe "madre" (`context_rows`) che danno senso alle altre.

Header, righe-sezione e gruppi di righe sono riconosciuti dalla sola struttura della tabella
(`table_structure.py`), senza parole chiave: le regole valgono per qualunque lingua e documento.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable

from src.extractors.table_images import get_table_images, get_table_row_pages
from src.extractors.table_structure import (
    LEVEL_MAIN_SECTION,
    LEVEL_ROW,
    blank_repeated_cells,
    count_header_rows,
    is_banner_row,
    is_numeric_value,
    is_section_row,
    is_span_row,
    looks_like_data_row,
    row_levels,
    section_style,
    split_merged_header,
)

if TYPE_CHECKING:
    from docling_core.types.doc import DoclingDocument, TableItem

# Intestazioni di colonne opzionali che possono essere legittimamente quasi vuote senza indicare errori di estrazione
_OPTIONAL_COLUMN_NAMES_RE = re.compile(
    r"^(?:notes?|annotazioni|osservazioni|commenti|rif\.?|riferimenti|remarks?|comments?|refs?\.?|references?)$",
    re.IGNORECASE,
)


def approx_token_count(text: str) -> int:
    """Stima prudente dei token (usata se non viene fornito il tokenizer del modello)."""
    return len(text) // 3 + 1


def find_empty_columns(
    headers: list[str],
    rows: list[list[str]],
    *,
    col_empty_threshold: float = 0.80,
    min_rows: int = 3,
) -> list[str]:
    """Individua le colonne con una frazione di celle vuote >= `col_empty_threshold`.

    - Richiede almeno `min_rows` righe dati (default 3) per evitare falsi positivi su tabelle
      di 1-2 righe dove una singola cella vuota supera il 50-100%.
    - Esclude colonne puramente opzionali (es. "Note", "Osservazioni").
    """
    n_rows = len(rows)
    if not headers or n_rows < min_rows:
        return []

    empty_cols: list[str] = []
    for j, h in enumerate(headers):
        h_clean = h.strip()
        if _OPTIONAL_COLUMN_NAMES_RE.match(h_clean):
            continue
        empty_in_col = sum(
            1 for r in rows if (r[j].strip() if j < len(r) else "") == ""
        )
        if (empty_in_col / n_rows) >= col_empty_threshold:
            empty_cols.append(h_clean or f"Colonna {j + 1}")

    return empty_cols


@dataclass
class TableBlock:
    """Rappresenta una tabella (o una porzione di righe di una tabella lunga) pronta per il chunking."""

    headers: list[str]
    rows: list[list[str]]
    title: str | None = None
    footnotes: list[str] = field(default_factory=list)
    pages: list[int] = field(default_factory=list)
    image_paths: list[str] = field(default_factory=list)
    table_index: int | None = None
    self_ref: str | None = None
    start_line: int = 0
    end_line: int = 0
    # Righe-sezione (celle unite a tutta larghezza): stessa lunghezza di `rows`
    section_flags: list[bool] = field(default_factory=list)
    # Righe ripetute dalla parte precedente per dare contesto (non contate in `rows`)
    context_rows: list[list[str]] = field(default_factory=list)
    # Pagina di ciascuna riga di `rows` (vuota se non nota)
    row_pages: list[int] = field(default_factory=list)
    row_start: int = 1
    row_end: int = 0
    total_rows: int = 0
    part: int = 1
    total_parts: int = 1
    # Metriche a livello di intera tabella (calcolate prima dello split per righe)
    table_empty_cells_ratio: float | None = None
    empty_columns: list[str] | None = None
    table_possibly_incomplete: bool | None = None

    def __post_init__(self) -> None:
        if len(self.section_flags) != len(self.rows):
            self.section_flags = [is_span_row(r) for r in self.rows]
        if self.total_rows == 0:
            self.total_rows = len(self.rows)
        if self.row_end == 0:
            self.row_end = len(self.rows)
        if self.table_empty_cells_ratio is None:
            self.table_empty_cells_ratio = self.chunk_empty_cells_ratio
        if self.empty_columns is None:
            self.empty_columns = find_empty_columns(self.headers, self.value_rows)

    @property
    def value_rows(self) -> list[list[str]]:
        """Righe dati escluse le righe-sezione (le loro celle vuote non sono dati mancanti)."""
        return [r for r, is_section in zip(self.rows, self.section_flags) if not is_section]

    @property
    def caption(self) -> str | None:
        """Alias retro-compatibile per `title`."""
        return self.title

    @property
    def is_valid(self) -> bool:
        """Verifica se la tabella contiene almeno un'intestazione e una riga."""
        return len(self.headers) > 0 and len(self.rows) > 0

    @property
    def total_cells_count(self) -> int:
        """Numero totale di celle dati nel blocco corrente."""
        return len(self.value_rows) * len(self.headers) if self.headers else 0

    @property
    def empty_cells_count(self) -> int:
        """Numero di celle dati vuote nel blocco corrente."""
        if not self.headers:
            return 0
        n_cols = len(self.headers)
        empty = 0
        for row in self.value_rows:
            for c_idx in range(n_cols):
                val = row[c_idx].strip() if c_idx < len(row) else ""
                if val == "":
                    empty += 1
        return empty

    @property
    def chunk_empty_cells_ratio(self) -> float:
        """Frazione di celle dati vuote sulle righe di questo singolo blocco (0.0 - 1.0)."""
        tot = self.total_cells_count
        if tot == 0:
            return 1.0
        return round(self.empty_cells_count / tot, 4)

    @property
    def empty_cells_ratio(self) -> float:
        """Frazione di celle vuote a livello di intera tabella (coerente tra tutte le parti)."""
        if self.table_empty_cells_ratio is not None:
            return self.table_empty_cells_ratio
        return self.chunk_empty_cells_ratio

    def is_possibly_incomplete(
        self,
        empty_ratio_threshold: float = 0.35,
        col_empty_threshold: float = 0.80,
        min_rows_for_col_check: int = 3,
    ) -> bool:
        """Determina se la tabella è potenzialmente incompleta (a livello di intera tabella).

        Scatta se:
          - non ci sono righe dati o header, oppure
          - il rapporto globale di celle vuote della tabella è >= `empty_ratio_threshold` (0.35), oppure
          - esiste almeno una colonna non opzionale con >= `col_empty_threshold` (80%) di celle vuote
            su una tabella con almeno `min_rows_for_col_check` (3) righe dati.
        """
        if self.table_possibly_incomplete is not None:
            return self.table_possibly_incomplete
        if not self.rows or not self.headers:
            return True
        cols = (
            self.empty_columns
            if self.empty_columns is not None
            else find_empty_columns(
                self.headers,
                self.value_rows,
                col_empty_threshold=col_empty_threshold,
                min_rows=min_rows_for_col_check,
            )
        )
        return self.empty_cells_ratio >= empty_ratio_threshold or len(cols) > 0

    def format_title_line(self) -> str | None:
        """Formatta la riga del titolo della tabella (includendo eventuale Parte k/N)."""
        part_info = (
            f" (Parte {self.part}/{self.total_parts}, righe {self.row_start}-{self.row_end} di {self.total_rows})"
            if self.total_parts > 1
            else ""
        )
        if self.title:
            return f"**Tabella: {self.title}{part_info}**"
        if part_info:
            return f"**Tabella{part_info}**"
        return None

    def to_markdown_table(self) -> str:
        """Restituisce solo la griglia Markdown (`| header |` + `| --- |` + righe dati)."""
        if not self.headers:
            return ""
        n_cols = len(self.headers)
        clean_headers = [h.replace("\n", " ").strip() for h in self.headers]
        header_line = "| " + " | ".join(clean_headers) + " |"
        separator_line = "| " + " | ".join("---" for _ in range(n_cols)) + " |"

        row_lines: list[str] = []
        for row in self.context_rows + self.rows:
            padded = [(row[j] if j < len(row) else "").replace("\n", " ").strip() for j in range(n_cols)]
            row_lines.append("| " + " | ".join(padded) + " |")

        return "\n".join([header_line, separator_line] + row_lines)

    def to_markdown(self, *, include_title: bool = True, include_footnotes: bool = True) -> str:
        """Converte il blocco tabella in Markdown completo di titolo e note."""
        parts: list[str] = []
        if include_title:
            title_line = self.format_title_line()
            if title_line:
                parts.append(title_line)

        md_grid = self.to_markdown_table()
        if md_grid:
            parts.append(md_grid)

        if include_footnotes and self.footnotes:
            parts.append("\n".join(f"> Nota: {fn}" for fn in self.footnotes))

        return "\n\n".join(parts)


def _strip_common_banner_prefix(headers: list[str]) -> tuple[str | None, list[str]]:
    """Se tutte le colonne iniziano con lo stesso prefisso `'BANNER - Colonna'` (tipico di
    Docling quando la riga 0 è un titolo unificato su tutte le colonne), separa il banner
    dall'intestazione delle singole colonne.
    """
    if len(headers) < 2:
        return None, headers

    split_parts = [h.split(" - ", 1) for h in headers]
    if not all(len(sp) == 2 and sp[0].strip() and sp[1].strip() for sp in split_parts):
        return None, headers

    first_prefix = split_parts[0][0].strip()
    if all(sp[0].strip() == first_prefix for sp in split_parts):
        return first_prefix, [sp[1].strip() for sp in split_parts]

    return None, headers


def _combine_titles(caption: str | None, banner: str | None, fallback: str | None = None) -> str | None:
    """Unisce caption esterna e banner interno senza duplicazioni."""
    pieces: list[str] = []
    for candidate in (caption, banner):
        if not candidate:
            continue
        c_clean = re.sub(r"\s+", " ", candidate).strip()
        if not c_clean:
            continue
        if not any(c_clean.lower() in existing.lower() or existing.lower() in c_clean.lower() for existing in pieces):
            pieces.append(c_clean)

    if pieces:
        return " — ".join(pieces)

    if fallback:
        fb_clean = re.sub(r"^#+\s*", "", fallback).strip()
        return fb_clean or None

    return None


def _split_header_and_data(
    rows_text: list[list[str]],
    row_is_header: list[bool],
) -> tuple[str | None, list[list[str]], list[tuple[int, list[str], bool]]]:
    """Separa titolo interno, righe di header e righe dati.

    Restituisce `(banner_title, header_rows, data_rows)`, dove ogni riga dati è
    `(indice_nella_griglia, celle, is_section)`.
    """
    rows, flags, origin = split_merged_header(rows_text, row_is_header)
    n_header = count_header_rows(rows, flags)
    indexed = list(zip(origin, rows))
    header_part = indexed[:n_header]
    data_part = [(i, r, is_section_row(r)) for i, r in indexed[n_header:]]

    # Riga 0 a tutta larghezza: è il titolo della tabella, oppure la prima di più righe-sezione
    banner_title: str | None = None
    if len(rows) >= 2 and is_banner_row(rows[0]) and sum(1 for c in rows[1] if c) >= 2:
        text = next(c for c in rows[0] if c)
        style = section_style(text)
        is_section = any(
            sec and section_style(r[0]) == style for _, r, sec in data_part if r is not rows[0]
        )
        if n_header > 0:
            header_part = header_part[1:]
            if is_section:
                data_part.insert(0, (origin[0], rows[0], True))
        elif not is_section:
            data_part = data_part[1:]
        if not is_section:
            banner_title = text

    # Se Docling non segnala alcun header, la prima riga lo diventa solo se non sembra un dato
    header_rows = [r for _, r in header_part]
    if not header_rows and not any(flags) and data_part:
        _, first, first_is_section = data_part[0]
        below = [r for _, r, _ in data_part[1:]]
        has_values = any(is_numeric_value(c) for c in first[1:])
        if not first_is_section and not (has_values and looks_like_data_row(first, below)):
            header_rows = [first]
            data_part = data_part[1:]

    data_rows = [
        (i, blank_repeated_cells(r) if sec and is_span_row(r) else r, sec)
        for i, r, sec in data_part
    ]
    return banner_title, header_rows, data_rows


def extract_table_block_from_docling(
    table: TableItem,
    doc: DoclingDocument,
    *,
    table_index: int | None = None,
    fallback_title: str | None = None,
    manifest_image_paths: list[str] | None = None,
    empty_ratio_threshold: float = 0.35,
    col_empty_threshold: float = 0.80,
    min_rows_for_col_check: int = 3,
) -> TableBlock:
    """Estrae un `TableBlock` strutturato da un `TableItem` di Docling.

    Usa `manifest_image_paths` (da `table_images.json`) come fonte principale dei percorsi
    immagine se fornito, con fallback automatico a `table.meta["rag__table_images"]`.
    """
    grid = table.data.grid
    rows_text: list[list[str]] = []
    row_is_header: list[bool] = []

    for row in grid:
        rows_text.append([re.sub(r"\s+", " ", c.text).strip() for c in row])
        row_is_header.append(any(c.column_header for c in row))

    # 1. Separa titolo interno, header e righe dati (correggendo i flag di Docling)
    num_cols = table.data.num_cols or (len(rows_text[0]) if rows_text else 0)
    banner_title, header_rows, indexed_rows = _split_header_and_data(rows_text, row_is_header)
    data_rows = [r for _, r, _ in indexed_rows]
    section_flags = [sec for _, _, sec in indexed_rows]

    # 2. Costruisci l'intestazione delle colonne

    headers: list[str] = []
    for col_idx in range(num_cols):
        col_parts: list[str] = []
        for h_row in header_rows:
            val = h_row[col_idx].strip() if col_idx < len(h_row) else ""
            if val and (not col_parts or val != col_parts[-1]):
                col_parts.append(val)
        headers.append(" - ".join(col_parts) if col_parts else f"Colonna {col_idx + 1}")

    prefix_banner, headers = _strip_common_banner_prefix(headers)
    if prefix_banner and not banner_title:
        banner_title = prefix_banner

    # 3. Estrai caption, footnotes, pagine e percorsi immagini
    caption_str = table.caption_text(doc).strip() if hasattr(table, "caption_text") else ""
    title = _combine_titles(caption_str, banner_title, fallback_title)

    footnotes: list[str] = []
    for fn_ref in getattr(table, "footnotes", []):
        fn_item = fn_ref.resolve(doc) if hasattr(fn_ref, "resolve") else None
        fn_text = getattr(fn_item, "text", "").strip() if fn_item is not None else ""
        if fn_text:
            footnotes.append(fn_text)

    pages = sorted({p.page_no for p in getattr(table, "prov", []) if hasattr(p, "page_no")})
    if manifest_image_paths is not None:
        image_paths = list(manifest_image_paths)
    else:
        img_entries = get_table_images(table)
        image_paths = [e["image_path"] for e in img_entries if "image_path" in e]

    grid_pages = get_table_row_pages(table)
    if len(grid_pages) == len(grid):
        row_pages = [grid_pages[i] for i, _, _ in indexed_rows]
    else:
        row_pages = pages * len(data_rows) if len(pages) == 1 else []

    block = TableBlock(
        headers=headers,
        rows=data_rows,
        title=title,
        footnotes=footnotes,
        pages=pages,
        image_paths=image_paths,
        table_index=table_index,
        self_ref=getattr(table, "self_ref", None),
        row_start=1 if data_rows else 0,
        row_end=len(data_rows),
        total_rows=len(data_rows),
        part=1,
        total_parts=1,
        section_flags=section_flags,
        row_pages=row_pages,
    )
    block.empty_columns = find_empty_columns(
        headers,
        block.value_rows,
        col_empty_threshold=col_empty_threshold,
        min_rows=min_rows_for_col_check,
    )
    block.table_possibly_incomplete = block.is_possibly_incomplete(
        empty_ratio_threshold=empty_ratio_threshold,
        col_empty_threshold=col_empty_threshold,
        min_rows_for_col_check=min_rows_for_col_check,
    )
    return block


def _context_row_indices(levels: list[int], start: int) -> list[int]:
    """Indici delle righe "madre" della riga `start` (es. sezione > totale > fascia d'età)."""
    needed = levels[start]
    found: list[int] = []
    for i in range(start - 1, -1, -1):
        if levels[i] >= needed:
            continue
        found.append(i)
        needed = levels[i]
        if needed == LEVEL_MAIN_SECTION:
            break
    return found[::-1]


def _plan_row_slices(
    table: TableBlock,
    max_rows: int,
    max_tokens: int,
    count_tokens: Callable[[str], int],
    extra_tokens: int,
) -> list[tuple[int, int, list[int]]]:
    """Decide i tagli `(inizio, fine, righe_di_contesto)` rispettando `max_rows` e `max_tokens`."""
    n = len(table.rows)
    levels = row_levels(table.rows, table.section_flags)
    has_groups = any(lv != LEVEL_ROW for lv in levels)

    row_tokens = [count_tokens("| " + " | ".join(r) + " |") for r in table.rows]
    shell = TableBlock(
        headers=table.headers, rows=[], title=table.title, footnotes=table.footnotes
    ).to_markdown()
    base_tokens = extra_tokens + count_tokens(shell)

    slices: list[tuple[int, int, list[int]]] = []
    start = 0
    while start < n:
        context = _context_row_indices(levels, start) if start > 0 else []
        used = base_tokens + sum(row_tokens[i] for i in context)
        end = start
        while end < n:
            over_rows = max_rows > 0 and end - start >= max_rows
            over_tokens = max_tokens > 0 and used + row_tokens[end] > max_tokens
            if end > start and (over_rows or over_tokens):
                break
            used += row_tokens[end]
            end += 1

        if end < n:
            if has_groups:
                # Preferisci tagliare all'inizio del gruppo più generale nella seconda metà
                half = start + max(1, (end - start) // 2)
                best = min(range(half, end + 1), key=lambda i: (levels[i], -i))
                if levels[best] <= LEVEL_ROW:
                    end = best
            # Una riga-sezione non deve restare in fondo, staccata dalle sue righe
            while end - start > 1 and table.section_flags[end - 1]:
                end -= 1

        slices.append((start, end, context))
        start = end
    return slices


def split_large_table(
    table: TableBlock,
    max_rows: int = 25,
    *,
    max_tokens: int = 0,
    count_tokens: Callable[[str], int] | None = None,
    extra_tokens: int = 0,
) -> list[TableBlock]:
    """Divide una tabella lunga per righe (e, se `max_tokens > 0`, per token), ripetendo titolo e
    header delle colonne in ogni chunk e propagando a tutte le parti i flag di completezza
    calcolati sull'intera tabella.

    `extra_tokens` sono i token del testo che precede la tabella nel chunk (intestazioni di sezione).
    """
    counter = count_tokens or approx_token_count
    full_ratio = table.empty_cells_ratio
    full_empty_cols = list(table.empty_columns) if table.empty_columns is not None else []
    full_incomplete = table.table_possibly_incomplete

    slices = _plan_row_slices(table, max_rows, max_tokens, counter, extra_tokens)
    if len(slices) <= 1:
        table.row_start = 1 if table.rows else 0
        table.row_end = len(table.rows)
        table.total_rows = len(table.rows)
        table.part = 1
        table.total_parts = 1
        table.table_empty_cells_ratio = full_ratio
        table.empty_columns = full_empty_cols
        table.table_possibly_incomplete = full_incomplete
        return [table]

    total_rows = len(table.rows)
    total_parts = len(slices)
    chunks: list[TableBlock] = []

    for part_idx, (start, end, context) in enumerate(slices, start=1):
        slice_rows = table.rows[start:end]
        slice_pages = table.row_pages[start:end]
        chunks.append(
            TableBlock(
                headers=list(table.headers),
                rows=slice_rows,
                title=table.title,
                footnotes=list(table.footnotes),
                pages=sorted(set(slice_pages)) if slice_pages else list(table.pages),
                section_flags=table.section_flags[start:end],
                context_rows=[list(table.rows[i]) for i in context],
                row_pages=slice_pages,
                image_paths=list(table.image_paths),
                table_index=table.table_index,
                self_ref=table.self_ref,
                start_line=table.start_line,
                end_line=table.end_line,
                row_start=start + 1,
                row_end=end,
                total_rows=total_rows,
                part=part_idx,
                total_parts=total_parts,
                table_empty_cells_ratio=full_ratio,
                empty_columns=list(full_empty_cols),
                table_possibly_incomplete=full_incomplete,
            )
        )

    return chunks


def _parse_md_row(line: str) -> list[str]:
    """Estrae le celle da una riga di tabella Markdown `| a | b | c |`."""
    stripped = line.strip()
    if stripped.startswith("|"):
        stripped = stripped[1:]
    if stripped.endswith("|"):
        stripped = stripped[:-1]
    return [c.strip() for c in stripped.split("|")]


def _is_separator_row(cells: list[str]) -> bool:
    """Verifica se una riga di tabella Markdown è il separatore `| --- | :---: |`."""
    return bool(cells) and all(bool(re.match(r"^:?-{2,}:?$", c.strip())) for c in cells if c.strip())


def extract_caption(lines: list[str], table_start: int) -> str | None:
    """Estrae un eventuale titolo/didascalia immediatamente sopra una tabella Markdown."""
    idx = table_start - 1
    while idx >= 0 and not lines[idx].strip():
        idx -= 1
    if idx < 0:
        return None

    prev_line = lines[idx].strip()
    if prev_line.startswith("|") or prev_line.startswith("#"):
        return None
    cleaned = re.sub(r"^\*\*|\*\*$", "", prev_line).strip()
    cleaned = re.sub(r"^Tabella:\s*", "", cleaned, flags=re.IGNORECASE).strip()
    if 0 < len(cleaned) <= 180:
        return cleaned
    return None


def parse_table(lines: list[str], start_idx: int) -> TableBlock | None:
    """Analizza una tabella Markdown a partire dalla riga `start_idx`."""
    table_lines: list[str] = []
    end_idx = start_idx

    for i in range(start_idx, len(lines)):
        line = lines[i].strip()
        if line.startswith("|"):
            table_lines.append(line)
            end_idx = i
        else:
            break

    if len(table_lines) < 2:
        return None

    raw_headers = _parse_md_row(table_lines[0])
    if not raw_headers or not any(raw_headers):
        return None

    data_start_idx = 2 if _is_separator_row(_parse_md_row(table_lines[1])) else 1
    banner_title, headers = _strip_common_banner_prefix(raw_headers)
    n_cols = len(headers)

    rows: list[list[str]] = []
    for line in table_lines[data_start_idx:]:
        cells = _parse_md_row(line)
        if _is_separator_row(cells):
            continue
        if len(cells) < n_cols:
            cells.extend([""] * (n_cols - len(cells)))
        rows.append(cells[:n_cols])

    caption = extract_caption(lines, start_idx)
    title = _combine_titles(caption, banner_title)

    return TableBlock(
        headers=headers,
        rows=rows,
        title=title,
        start_line=start_idx,
        end_line=end_idx,
        row_start=1 if rows else 0,
        row_end=len(rows),
        total_rows=len(rows),
    )


def find_all_tables(content: str) -> list[TableBlock]:
    """Individua tutte le tabelle in una stringa Markdown."""
    lines = content.split("\n")
    tables: list[TableBlock] = []
    i = 0

    while i < len(lines):
        if lines[i].strip().startswith("|"):
            table = parse_table(lines, i)
            if table is not None:
                tables.append(table)
                i = table.end_line + 1
                continue
        i += 1

    return tables
