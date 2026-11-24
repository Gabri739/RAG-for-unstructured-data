"""Modulo di fusione delle tabelle multi-pagina su DoclingDocument.

Secondo blocco della pipeline di ingestione:
  PDF -> Docling -> Fusione (tabelle e righe) -> Immagini -> Chunking -> ...

Nessuna regola dipende dal contenuto o dalla lingua del documento, né dal formato di pagina:
  1. Rimozione tabelle vuote/fittizie (0 righe o 0 colonne).
  2. Rilevamento catene di tabelle spezzate su pagine consecutive (p -> p+1). Due tabelle si
     fondono solo se:
     - la prima arriva in fondo alla pagina e la seconda inizia in cima (frazioni di pagina);
     - sono allineate orizzontalmente e hanno un numero di colonne compatibile;
     - tra loro non c'è testo reale né una figura (testate e piè di pagina sono riconosciuti
       per ripetizione, vedi `running_text.py`);
     - la seconda non è una tabella nuova (didascalia, titolo o intestazione propri).
  3. Unificazione della griglia (TableData): allineamento delle colonne mancanti, rimozione degli
     header ripetuti, ricucitura delle righe spezzate a cavallo di pagina, header normalizzati
     con le regole strutturali di `table_structure.py`.
  4. Aggiornamento diretto di DoclingDocument:
     - la tabella capofila conserva la lista completa di ProvenanceItem (pagine e bbox), così da
       permettere il ritaglio immagine per ciascuna pagina, e la pagina di ogni riga;
     - le tabelle di continuazione e le testate di pagina intruse vengono eliminate.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from docling_core.types.doc import (
    DocItemLabel,
    DoclingDocument,
    NodeItem,
    PictureItem,
    TableCell,
    TableData,
    TableItem,
)

from .running_text import find_running_text_refs, page_box
from .table_images import set_table_row_pages
from .table_structure import (
    count_header_rows,
    has_header_contrast,
    is_banner_row,
    is_section_row,
    section_style,
    split_merged_header,
)

if TYPE_CHECKING:
    pass

# Soglie in frazioni di pagina, indipendenti dal formato (A4, Letter, orizzontale, ...)
BOTTOM_ZONE: float = 0.20      # t1 deve finire nell'ultimo 20% della pagina
TOP_ZONE: float = 0.18         # t2 deve iniziare nel primo 18% della pagina successiva
HORIZ_TOLERANCE: float = 0.025 # tolleranza allineamento margini sinistro/destro
MIN_PICTURE_AREA: float = 0.03 # figure più piccole (loghi, icone) non separano due tabelle


@dataclass
class MergedTableInfo:
    """Metadati diagnostici di una catena di tabelle fusa."""

    head_index: int
    merged_indices: list[int]
    pages: list[int]
    bboxes: list[dict]
    num_rows: int
    num_cols: int
    stitched_rows: int
    caption: str


@dataclass
class FusionReport:
    """Riepilogo dell'operazione di fusione tabelle su un DoclingDocument."""

    original_table_count: int
    empty_tables_removed: list[int]
    fused_table_count: int
    merged_chains: list[MergedTableInfo] = field(default_factory=list)


def _is_empty_table(table: TableItem) -> bool:
    """Verifica se una tabella è vuota (0x0 o priva di testo in tutte le celle)."""
    if table.data.num_rows == 0 or table.data.num_cols == 0:
        return True
    return not any(c.text.strip() for c in table.data.table_cells)


def _collect_intervening_items(
    doc: DoclingDocument,
) -> dict[tuple[str, str], list[NodeItem]]:
    """Mappa ogni coppia di TableItem consecutive (ref1, ref2) agli elementi intermedi nel body."""
    items_in_order: list[NodeItem] = [item for item, _ in doc.iterate_items()]
    table_positions: list[tuple[int, str]] = [
        (idx, item.self_ref)
        for idx, item in enumerate(items_in_order)
        if isinstance(item, TableItem)
    ]
    between_map: dict[tuple[str, str], list[NodeItem]] = {}
    for (pos1, ref1), (pos2, ref2) in zip(table_positions, table_positions[1:]):
        between_map[(ref1, ref2)] = items_in_order[pos1 + 1 : pos2]
    return between_map


def _is_separating_picture(
    item: PictureItem,
    doc: DoclingDocument,
    tables: tuple[TableItem, TableItem],
) -> bool:
    """True per una figura di dimensioni significative fuori dalle due tabelle."""
    if not item.prov:
        return False
    box = page_box(doc, item.prov[0])
    if box is None or (box[2] - box[0]) * (box[3] - box[1]) < MIN_PICTURE_AREA:
        return False
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    for table in tables:
        for prov in table.prov:
            tb = page_box(doc, prov)
            if prov.page_no == item.prov[0].page_no and tb is not None:
                if tb[0] <= cx <= tb[2] and tb[1] <= cy <= tb[3]:
                    return False
    return True


def _classify_intervening_items(
    t1: TableItem,
    t2: TableItem,
    between_items: list[NodeItem],
    doc: DoclingDocument,
    running_refs: set[str],
) -> tuple[bool, list[NodeItem]]:
    """Controlla se tra t1 e t2 c'è contenuto reale del documento oppure solo rumore/didascalie.

    Restituisce:
      (has_real_content, noise_items_to_delete)
    """
    owned_refs = {r.cref for t in (t1, t2) for r in (t.captions + t.footnotes)}
    p1 = t1.prov[-1].page_no
    box1 = page_box(doc, t1.prov[-1])

    noise_to_delete: list[NodeItem] = []

    for item in between_items:
        ref = getattr(item, "self_ref", None)
        if ref in owned_refs:
            continue

        if isinstance(item, PictureItem):
            if _is_separating_picture(item, doc, (t1, t2)):
                return True, []
            continue

        label = getattr(item, "label", None)
        if label in {
            DocItemLabel.PAGE_HEADER,
            DocItemLabel.PAGE_FOOTER,
            DocItemLabel.FOOTNOTE,
            DocItemLabel.CAPTION,
        }:
            continue

        if not getattr(item, "text", "").strip():
            continue

        # Testata, piè di pagina o numero di pagina classificati come testo o titolo
        if ref in running_refs:
            noise_to_delete.append(item)
            continue

        # Nota a piè di pagina posizionata fisicamente sotto la tabella t1
        prov = getattr(item, "prov", None)
        if prov and prov[0].page_no == p1 and box1 is not None:
            box = page_box(doc, prov[0])
            if box is not None and box[1] >= box1[3] - 0.005:
                continue

        return True, []

    return False, noise_to_delete


def _starts_new_table(t1: TableItem, t2: TableItem, doc: DoclingDocument) -> bool:
    """True se t2 ha titolo o intestazione propri, quindi non è la continuazione di t1."""
    caption1 = t1.caption_text(doc).strip().lower()
    caption2 = t2.caption_text(doc).strip().lower()
    if caption2 and caption2 != caption1 and not (caption1 and caption2.startswith(caption1)):
        return True

    rows1, flags1 = _extract_grid_and_headers(t1)
    rows2, flags2 = _extract_grid_and_headers(t2)
    header1 = rows1[: max(count_header_rows(rows1, flags1), 1)]

    if not rows2 or _is_repeated_header(rows2[0], header1):
        return False

    first = rows2[0]
    if is_section_row(first) or is_banner_row(first):
        # Se t1 contiene già righe-sezione dello stesso tipo, t2 ne prosegue semplicemente l'elenco
        style = section_style(first[0])
        if any(is_section_row(r) and section_style(r[0]) == style for r in rows1):
            return False
    if is_banner_row(first):
        # Un titolo a tutta larghezza seguito da un'intestazione apre una tabella nuova
        has_header = len(rows2) >= 3 and has_header_contrast(rows2[1], rows2[2:])
        return has_header and not _is_repeated_header(rows2[1], header1)

    return has_header_contrast(first, rows2[1:]) and bool(first[0].strip())


def should_merge_tables(
    t1: TableItem,
    t2: TableItem,
    doc: DoclingDocument,
    between_items: list[NodeItem],
    running_refs: set[str] | None = None,
) -> tuple[bool, list[NodeItem]]:
    """Determina se t2 è la continuazione diretta di t1 sulla pagina successiva."""
    if _is_empty_table(t1) or _is_empty_table(t2):
        return False, []

    if not t1.prov or not t2.prov:
        return False, []

    if t2.prov[0].page_no != t1.prov[-1].page_no + 1:
        return False, []

    box1 = page_box(doc, t1.prov[-1])
    box2 = page_box(doc, t2.prov[0])
    if box1 is None or box2 is None:
        return False, []

    # 1. Verifica vicinanza ai margini inferiore (p1) e superiore (p2)
    if box1[3] < 1 - BOTTOM_ZONE or box2[1] > TOP_ZONE:
        return False, []

    # 2. Verifica allineamento orizzontale
    c1 = t1.data.num_cols
    c2 = t2.data.num_cols
    left_aligned = abs(box1[0] - box2[0]) <= HORIZ_TOLERANCE
    right_aligned = abs(box1[2] - box2[2]) <= HORIZ_TOLERANCE
    if not left_aligned:
        return False, []
    if not right_aligned and c2 != 1:
        return False, []

    # 3. Verifica compatibilità colonne:
    #    - Stesso numero di colonne (c1 == c2)
    #    - Una colonna in meno (colonna di icone o vuota persa nella continuazione)
    #    - Continuazione di una sola colonna
    cols_compatible = (
        c1 == c2
        or (c1 > c2 and c1 - c2 == 1)
        or (c2 == 1 and c1 <= 6)
    )
    if not cols_compatible:
        return False, []

    # 4. Verifica assenza di contenuto intermedio nel flusso del documento
    refs = find_running_text_refs(doc) if running_refs is None else running_refs
    has_real_content, noise_to_delete = _classify_intervening_items(
        t1, t2, between_items, doc, refs
    )
    if has_real_content:
        return False, []

    # 5. Verifica che t2 non sia una tabella nuova
    if _starts_new_table(t1, t2, doc):
        return False, []

    return True, noise_to_delete


def _extract_grid_and_headers(table: TableItem) -> tuple[list[list[str]], list[bool]]:
    """Estrae la matrice di stringhe (righe x colonne) e il flag column_header per ogni riga."""
    grid = table.data.grid
    rows_text: list[list[str]] = []
    row_is_header: list[bool] = []
    for row in grid:
        rows_text.append([re.sub(r"\s+", " ", c.text).strip() for c in row])
        row_is_header.append(any(c.column_header for c in row))
    return rows_text, row_is_header


def _find_empty_interior_col(rows: list[list[str]], header_count: int, num_cols: int) -> int | None:
    """Cerca se esiste una colonna interna (0 < col < num_cols-1) sempre vuota nelle righe dati."""
    data_rows = rows[header_count:] if len(rows) > header_count else rows
    if not data_rows:
        return None
    for col_idx in range(1, num_cols - 1):
        if all(r[col_idx] == "" for r in data_rows):
            return col_idx
    return None


def _pad_row_to_cols(
    row: list[str],
    target_cols: int,
    empty_interior_col: int | None,
) -> list[str]:
    """Allinea una riga con meno colonne al numero di colonne target."""
    if len(row) == target_cols:
        return list(row)
    if len(row) == target_cols - 1 and empty_interior_col is not None:
        return row[:empty_interior_col] + [""] + row[empty_interior_col:]
    if len(row) < target_cols:
        return list(row) + [""] * (target_cols - len(row))
    return list(row[:target_cols])


def _is_repeated_header(row: list[str], header_rows: list[list[str]]) -> bool:
    """Verifica se una riga iniziale di continuazione ripete uno degli header della prima tabella."""
    non_empty = [c.lower() for c in row if c]
    if not non_empty:
        return False
    for h_row in header_rows:
        if [c.lower() for c in row] == [c.lower() for c in h_row]:
            return True
        h_non_empty = [c.lower() for c in h_row if c]
        if len(non_empty) >= 2 and non_empty == h_non_empty:
            return True
    return False


def _is_split_row_continuation(prev_row: list[str], next_row: list[str]) -> bool:
    """Rileva se la prima riga della tabella successiva è il frammento spezzato dell'ultima riga precedente.

    Esempi: ['', '', 'resto della descrizione...'], ['', 'fine della voce precedente', '', '']
    """
    num_cols = len(next_row)
    if num_cols < 2:
        return False

    # La prima colonna (ID riga / Categoria / Tema) deve essere vuota nella riga spezzata
    if next_row[0] != "":
        return False

    non_empty_indices = [j for j, val in enumerate(next_row) if val != ""]
    if not non_empty_indices:
        return False

    # Nelle righe spezzate solo 1 (o al massimo metà nelle tabelle larghe) colonna contiene il testo residuo,
    # mentre tutte le altre colonne (es. valori numerici, stakeholder, impatti) sono vuote.
    max_split_cols = 1 if num_cols <= 4 else 2
    if len(non_empty_indices) > max_split_cols:
        return False

    # Ogni cella non vuota in next_row deve continuare una cella non vuota nella stessa colonna di prev_row
    for j in non_empty_indices:
        if prev_row[j] == "":
            return False
        nxt = next_row[j]
        prv = prev_row[j]
        # Se entrambe iniziano con un codice distinto (es. "405-1" e "405-2"), sono righe diverse
        if re.match(r"^\d+[-.]\d+", prv) and re.match(r"^\d+[-.]\d+", nxt):
            return False
        # Verifica segnale linguistico di continuazione: inizia con minuscola / parentesi,
        # oppure la riga precedente ha la colonna chiave (col 0) valorizzata e le altre colonne piene.
        starts_as_continuation = nxt[0].islower() or nxt[0] in "(-•●"
        prev_has_other_cols = any(
            prev_row[k] != "" for k in range(num_cols) if k not in non_empty_indices
        )
        if not (starts_as_continuation or prev_has_other_cols):
            return False

    return True


def _normalize_header_flags(rows: list[list[str]], is_header: list[bool]) -> list[bool]:
    """Normalizza i flag column_header con le regole strutturali di `count_header_rows`.

    Un titolo a tutta larghezza seguito dall'intestazione viene ridotto alla prima cella, così da
    non essere ripetuto su tutte le colonne nell'header combinato.
    """
    if not rows:
        return is_header
    n_header = count_header_rows(rows, is_header)
    if n_header >= 2 and is_banner_row(rows[0]):
        rows[0] = [next(c for c in rows[0] if c)] + [""] * (len(rows[0]) - 1)
    return [i < n_header for i in range(len(rows))]


def _build_table_data(rows: list[list[str]], is_header: list[bool], num_cols: int) -> TableData:
    """Costruisce un nuovo oggetto TableData a partire dalla griglia di testo e dai flag header."""
    cells: list[TableCell] = []
    for i, (row, hdr) in enumerate(zip(rows, is_header)):
        for j, text in enumerate(row):
            cells.append(
                TableCell(
                    text=text,
                    start_row_offset_idx=i,
                    end_row_offset_idx=i + 1,
                    start_col_offset_idx=j,
                    end_col_offset_idx=j + 1,
                    row_span=1,
                    col_span=1,
                    column_header=bool(hdr),
                    row_header=False,
                    row_section=False,
                )
            )
    return TableData(
        table_cells=cells,
        num_rows=len(rows),
        num_cols=num_cols,
    )


def _merge_table_chain(chain: list[TableItem]) -> tuple[TableData, int, list[int]]:
    """Unisce una catena di TableItem [T_0, T_1, ..., T_k] in un unico TableData.

    Restituisce:
      (merged_table_data, num_stitched_rows, pagina_di_ogni_riga)
    Una riga ricucita a cavallo di due pagine mantiene la pagina in cui inizia.
    """
    target_cols = max(t.data.num_cols for t in chain)

    # Estrai griglia della prima tabella
    head_rows, head_hdr_flags = _extract_grid_and_headers(chain[0])
    head_hdr_flags = _normalize_header_flags(head_rows, head_hdr_flags)
    header_count = sum(1 for f in head_hdr_flags if f)

    # Verifica se la prima tabella ha una colonna interna sempre vuota nei dati (es. icone)
    empty_interior_col = _find_empty_interior_col(head_rows, header_count, chain[0].data.num_cols)

    merged_rows: list[list[str]] = [
        _pad_row_to_cols(r, target_cols, empty_interior_col) for r in head_rows
    ]
    merged_hdr_flags: list[bool] = list(head_hdr_flags)
    row_pages: list[int] = [chain[0].prov[0].page_no] * len(merged_rows)
    header_rows_ref = merged_rows[: max(header_count, 1)]

    stitched_count = 0

    for cont_table in chain[1:]:
        cont_rows, _ = _extract_grid_and_headers(cont_table)
        if not cont_rows:
            continue

        aligned_rows = [
            _pad_row_to_cols(r, target_cols, empty_interior_col) for r in cont_rows
        ]

        # 1. Salta eventuale header ripetuto in cima alla pagina di continuazione
        start_idx = 0
        while start_idx < len(aligned_rows) and _is_repeated_header(
            aligned_rows[start_idx], header_rows_ref
        ):
            start_idx += 1

        remaining_rows = aligned_rows[start_idx:]
        if not remaining_rows:
            continue

        # 2. Controlla se la prima riga di continuazione è una riga spezzata (split-row)
        #    che continua l'ultima riga dati della pagina precedente
        if (
            merged_rows
            and not merged_hdr_flags[-1]
            and _is_split_row_continuation(merged_rows[-1], remaining_rows[0])
        ):
            prev_last = merged_rows[-1]
            split_first = remaining_rows[0]
            for j in range(target_cols):
                if split_first[j]:
                    if prev_last[j]:
                        # Se la riga precedente termina con trattino di sillabazione, unisci senza spazio
                        if prev_last[j].endswith("-") and not prev_last[j].endswith(" -"):
                            prev_last[j] = prev_last[j][:-1] + split_first[j]
                        else:
                            prev_last[j] = f"{prev_last[j]} {split_first[j]}"
                    else:
                        prev_last[j] = split_first[j]
            remaining_rows = remaining_rows[1:]
            stitched_count += 1

        # 3. Accoda le restanti righe come righe dati (column_header=False)
        for r in remaining_rows:
            merged_rows.append(r)
            merged_hdr_flags.append(False)
            row_pages.append(cont_table.prov[0].page_no)

    # Riapplica la normalizzazione degli header (utile quando la prima tabella aveva solo l'header)
    merged_hdr_flags = _normalize_header_flags(merged_rows, merged_hdr_flags)
    merged_rows, merged_hdr_flags, origin = split_merged_header(merged_rows, merged_hdr_flags)
    row_pages = [row_pages[i] for i in origin]

    merged = _build_table_data(merged_rows, merged_hdr_flags, target_cols)
    return merged, stitched_count, row_pages


def fuse_document_tables(doc: DoclingDocument) -> FusionReport:
    """Esegue la pulizia e la fusione in-place delle tabelle multi-pagina su DoclingDocument.

    Operazioni effettuate sul documento:
      1. Elimina le tabelle vuote (0x0).
      2. Individua le catene di tabelle spezzate su pagine consecutive.
      3. Per ciascuna catena, aggiorna `head_table.data` con la griglia unificata e
         concatena `prov`, `captions` e `footnotes` delle tabelle successive.
      4. Normalizza anche le tabelle singole con titolo a tutta larghezza in riga 0.
      5. Elimina dal documento le tabelle di continuazione e le testate di pagina classificate
         per errore come testo o titolo.
    """
    original_count = len(doc.tables)
    empty_indices: list[int] = []
    items_to_delete: list[NodeItem] = []

    # 1. Individua tabelle vuote
    non_empty_tables: list[ tuple[int, TableItem] ] = []
    for idx, table in enumerate(doc.tables):
        if _is_empty_table(table):
            empty_indices.append(idx)
            items_to_delete.append(table)
        else:
            non_empty_tables.append((idx, table))

    # 2. Costruisci mappa degli elementi intermedi nel documento e trova le testate di pagina
    between_map = _collect_intervening_items(doc)
    running_refs = find_running_text_refs(doc)

    # 3. Raggruppa le tabelle non vuote in catene di continuazione
    chains: list[list[tuple[int, TableItem]]] = []
    noise_to_delete: list[NodeItem] = []

    for idx, table in non_empty_tables:
        if not chains:
            chains.append([(idx, table)])
            continue

        prev_idx, prev_table = chains[-1][-1]
        between_items = between_map.get((prev_table.self_ref, table.self_ref), [])
        can_merge, noise_items = should_merge_tables(
            prev_table, table, doc, between_items, running_refs
        )
        if can_merge:
            chains[-1].append((idx, table))
            noise_to_delete.extend(noise_items)
        else:
            chains.append([(idx, table)])

    # 4. Esegui la fusione per ogni catena (e normalizza gli header delle tabelle singole)
    merged_infos: list[MergedTableInfo] = []

    for chain in chains:
        if len(chain) == 1:
            # Tabella singola: normalizza comunque eventuali titolo interno + header
            _, single_table = chain[0]
            rows, hdr_flags = _extract_grid_and_headers(single_table)
            new_flags = _normalize_header_flags(rows, hdr_flags)
            if new_flags != hdr_flags:
                single_table.data = _build_table_data(rows, new_flags, single_table.data.num_cols)
            continue

        head_orig_idx, head_table = chain[0]
        table_objs = [t for _, t in chain]
        orig_indices = [i for i, _ in chain]

        merged_data, stitched_count, row_pages = _merge_table_chain(table_objs)
        head_table.data = merged_data
        set_table_row_pages(head_table, row_pages)

        # Unisci provenance, captions e footnotes delle tabelle di continuazione nella capofila
        existing_caps = {r.cref for r in head_table.captions}
        existing_fns = {r.cref for r in head_table.footnotes}

        for cont_table in table_objs[1:]:
            head_table.prov.extend(cont_table.prov)
            for cap_ref in cont_table.captions:
                if cap_ref.cref not in existing_caps:
                    head_table.captions.append(cap_ref)
                    existing_caps.add(cap_ref.cref)
            for fn_ref in cont_table.footnotes:
                if fn_ref.cref not in existing_fns:
                    head_table.footnotes.append(fn_ref)
                    existing_fns.add(fn_ref.cref)
            items_to_delete.append(cont_table)

        pages = sorted({p.page_no for p in head_table.prov})
        bboxes = [
            {
                "page": p.page_no,
                "l": round(p.bbox.l, 1),
                "t": round(p.bbox.t, 1),
                "r": round(p.bbox.r, 1),
                "b": round(p.bbox.b, 1),
                "origin": p.bbox.coord_origin.value,
            }
            for p in head_table.prov
        ]

        merged_infos.append(
            MergedTableInfo(
                head_index=head_orig_idx,
                merged_indices=orig_indices,
                pages=pages,
                bboxes=bboxes,
                num_rows=merged_data.num_rows,
                num_cols=merged_data.num_cols,
                stitched_rows=stitched_count,
                caption=head_table.caption_text(doc),
            )
        )

    # 5. Rimuovi dal documento le tabelle vuote, le tabelle di continuazione e le testate di pagina
    noise_refs = {item.self_ref for item in noise_to_delete} | running_refs
    all_deletions = items_to_delete + [t for t in doc.texts if t.self_ref in noise_refs]
    if all_deletions:
        doc.delete_items(node_items=all_deletions)

    return FusionReport(
        original_table_count=original_count,
        empty_tables_removed=empty_indices,
        fused_table_count=len(doc.tables),
        merged_chains=merged_infos,
    )
