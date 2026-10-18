"""Modulo di fusione delle tabelle multi-pagina su DoclingDocument.

Secondo blocco della pipeline di ingestione:
  PDF nativi -> Docling -> Fusione (tabelle e righe) -> Immagini -> Chunking -> ...

Calibrato sui dati reali DNF 2023 (85 tabelle estratte da TableFormer ACCURATE):
  1. Rimozione tabelle vuote/fittizie (0 righe o 0 colonne, es. T5 e T6).
  2. Rilevamento catene di tabelle spezzate su pagine consecutive (p -> p+1):
     - Coordinate verticali sul taglio pagina (BOTTOMLEFT: b1.b <= 145 pt, b2.t >= 740 pt).
     - Allineamento orizzontale (|l1 - l2| <= 15 pt e |r1 - r2| <= 15 pt, o stub a 1 colonna).
     - Assenza di testo discorsivo intermedio nel body (ignorando header/footer di pagina
       come "Sara Assicurazioni SpA DNF 2023", numeri di pagina e note a piè di pagina).
     - Compatibilità del numero di colonne (c1 == c2, oppure |c1 - c2| == 1 per colonne
       icone/vuote come T2->T3->T4 e T50->T51, oppure continuazione stub c2 == 1 come T57->T58).
  3. Unificazione della griglia (TableData):
     - Allineamento automatico delle colonne mancanti (interna vuota vs finale vuota).
     - Normalizzazione degli header multi-livello (es. Modelli Tassonomia UE a 3 righe di header)
       e dei banner di titolo a tutta larghezza (es. T29, T46).
     - Rimozione di eventuali header ripetuti nelle pagine successive.
     - Ricucitura delle righe spezzate a cavallo di pagina (split-row stitching: es. T3->T4,
       T9->T10, T62->T63, T66->T67, T72->T73, T74->T75).
  4. Aggiornamento diretto di DoclingDocument:
     - Conserva nella tabella capofila la lista completa di ProvenanceItem (pagine e bbox)
       così da permettere il ritaglio immagine per ciascuna pagina (t.get_image(doc, prov_index=k)).
     - Elimina dal documento le tabelle di continuazione e gli header di pagina intrusi.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from docling_core.types.doc import (
    CoordOrigin,
    DocItemLabel,
    DoclingDocument,
    NodeItem,
    TableCell,
    TableData,
    TableItem,
)

if TYPE_CHECKING:
    pass

# Soglie spaziali calibrate su A4 (595 x 842 pt) in coordinate BOTTOMLEFT
BOTTOM_MARGIN_MAX_PT: float = 145.0  # t1 finisce vicino al fondo pagina (osservato: 52.8 - 140.3; falso positivo T35 = 172.5)
TOP_MARGIN_MIN_PT: float = 740.0     # t2 inizia in cima alla pagina successiva (osservato: 753.8 - 768.4)
HORIZ_TOLERANCE_PT: float = 15.0     # tolleranza allineamento margini sinistro/destro

# Pattern di rumore tipografico (running headers/footers classificati come testo o section_header)
_PAGE_NOISE_RE = re.compile(
    r"^(?:"
    r"sara\s+assicurazioni.*dnf.*|"
    r"dnf\s*\d{4}|"
    r"\d{1,3}"
    r")$",
    re.IGNORECASE,
)


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


def _classify_intervening_items(
    t1: TableItem,
    t2: TableItem,
    between_items: list[NodeItem],
) -> tuple[bool, list[NodeItem]]:
    """Controlla se tra t1 e t2 c'è testo reale del documento oppure solo rumore/didascalie.

    Restituisce:
      (has_real_text, noise_items_to_delete)
    """
    t1_owned_refs = {r.cref for r in (t1.captions + t1.footnotes)}
    t2_owned_refs = {r.cref for r in (t2.captions + t2.footnotes)}
    p1 = t1.prov[-1].page_no
    b1 = t1.prov[-1].bbox
    p2 = t2.prov[0].page_no

    noise_to_delete: list[NodeItem] = []

    for item in between_items:
        ref = getattr(item, "self_ref", None)
        if ref in t1_owned_refs or ref in t2_owned_refs:
            continue

        label = getattr(item, "label", None)
        if label in {
            DocItemLabel.PAGE_HEADER,
            DocItemLabel.PAGE_FOOTER,
            DocItemLabel.FOOTNOTE,
            DocItemLabel.CAPTION,
        }:
            continue

        text = getattr(item, "text", "").strip()
        if not text:
            continue

        # Header di pagina intruso (es. "## Sara Assicurazioni SpA DNF 2023" a pag. 44, 56, 65)
        if _PAGE_NOISE_RE.match(text):
            noise_to_delete.append(item)
            continue

        prov = getattr(item, "prov", None)
        if prov:
            ip = prov[0].page_no
            ib = prov[0].bbox
            if ib.coord_origin == CoordOrigin.BOTTOMLEFT:
                # Nota a piè di pagina sulla pagina p1 posizionata fisicamente sotto la tabella t1
                if ip == p1 and ib.t <= b1.b + 4.0:
                    continue
                # Running header nel margine superiore della pagina p2 (sopra y = 780 pt)
                if ip == p2 and ib.b >= 780.0 and len(text) <= 80:
                    noise_to_delete.append(item)
                    continue

        # Se arriviamo qui, c'è un vero paragrafo o titolo di sezione tra le due tabelle
        return True, []

    return False, noise_to_delete


def should_merge_tables(
    t1: TableItem,
    t2: TableItem,
    doc: DoclingDocument,
    between_items: list[NodeItem],
) -> tuple[bool, list[NodeItem]]:
    """Determina se t2 è la continuazione diretta di t1 sulla pagina successiva."""
    if _is_empty_table(t1) or _is_empty_table(t2):
        return False, []

    if not t1.prov or not t2.prov:
        return False, []

    p1 = t1.prov[-1].page_no
    p2 = t2.prov[0].page_no
    if p2 != p1 + 1:
        return False, []

    b1 = t1.prov[-1].bbox
    b2 = t2.prov[0].bbox

    # 1. Verifica vicinanza ai margini inferiore (p1) e superiore (p2)
    if b1.coord_origin == CoordOrigin.BOTTOMLEFT:
        if b1.b > BOTTOM_MARGIN_MAX_PT:
            return False, []
        if b2.t < TOP_MARGIN_MIN_PT:
            return False, []
    else:
        page_h = doc.pages[p1].size.height if p1 in doc.pages and doc.pages[p1].size else 842.0
        if (page_h - b1.b) > BOTTOM_MARGIN_MAX_PT:
            return False, []
        if (page_h - b2.t) < TOP_MARGIN_MIN_PT and b2.t > (page_h - TOP_MARGIN_MIN_PT):
            return False, []

    # 2. Verifica allineamento orizzontale
    c1 = t1.data.num_cols
    c2 = t2.data.num_cols
    left_aligned = abs(b1.l - b2.l) <= HORIZ_TOLERANCE_PT
    right_aligned = abs(b1.r - b2.r) <= HORIZ_TOLERANCE_PT
    if not left_aligned:
        return False, []
    if not right_aligned and c2 != 1:
        return False, []

    # 3. Verifica compatibilità colonne:
    #    - Stesso numero di colonne (c1 == c2)
    #    - Differenza di 1 colonna per colonna icone/vuota (es. T2 3col -> T3 2col, T50 5col -> T51 4col)
    #    - Continuazione stub a 1 colonna (es. T57 4col -> T58 1col)
    cols_compatible = (
        c1 == c2
        or (c1 > c2 and c1 - c2 == 1)
        or (c2 == 1 and c1 <= 6)
    )
    if not cols_compatible:
        return False, []

    # 4. Verifica assenza di testo intermedio nel flusso del documento
    has_real_text, noise_to_delete = _classify_intervening_items(t1, t2, between_items)
    if has_real_text:
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
    """Cerca se esiste una colonna interna (0 < col < num_cols-1) sempre vuota nelle righe dati (es. T2 col 1)."""
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

    Casi reali calibrati su DNF 2023:
      - T3 -> T4:   ['', '', 'di sicurezza ovvero trattamento dei dati...']
      - T9 -> T10:  ['', '', '', '', '', 'prodotti/tariffe Sara Assicurazioni...']
      - T62 -> T63: ['', '(UE) 2021/2139 al denominatore del KPI applicabile', '', ...]
      - T66 -> T67: ['', 'non allineata alla tassonomia...', '', ...]
      - T72 -> T73: ['', '(UE) 2021/2139 al numeratore del KPI applicabile', '', ...]
      - T74 -> T75: ['', 'denominatore del KPI applicabile', '', ...]
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
        # Se entrambe iniziano con un codice numerico distinto (es. "405-1" vs "405-2"), sono righe diverse
        if re.match(r"^\d{1,3}-\d{1,2}\b", prv) and re.match(r"^\d{1,3}-\d{1,2}\b", nxt):
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
    """Normalizza i flag column_header sulle prime righe della tabella.

    Gestisce tre casi frequenti nei report DNF/bilanci:
      1. Banner di titolo in riga 0 (tutte le celle uguali o solo col 0 piena) seguito
         dalla vera intestazione colonne in riga 1 (es. T29 'SUBAGENTI...', T46 'EMISSIONI...').
      2. Intestazioni multi-livello a 2 o 3 righe (es. T70, T72 Modelli Tassonomia UE dove
         le righe 0..2 compongono l'header con 'Importo' e '%').
      3. Tabelle di dati che hanno la riga 0 con intestazioni esplicite ma column_header=False
         (es. T76 'Riga | Attività economiche | Importo | Percentuale').
    """
    n_rows = len(rows)
    if n_rows == 0:
        return is_header

    flags = list(is_header)

    # Caso 3: Riga 0 è chiaramente un header ("Riga | Attività economiche..." o "Tema | ...")
    r0_lower = [c.lower() for c in rows[0]]
    if not flags[0] and n_rows >= 2:
        if r0_lower[0] in {"riga", "rig a", "tema", "tipologia", "categoria", "città", "gri standard"}:
            flags[0] = True

    # Caso 1: Riga 0 è un banner di titolo a tutta larghezza e Riga 1 è l'intestazione colonne
    if n_rows >= 2:
        r0_non_empty = [c for c in rows[0] if c]
        r1_non_empty = [c for c in rows[1] if c]
        is_r0_banner = (
            len(r0_non_empty) == 1 and rows[0][0] != "" and len(rows[0]) > 1
        ) or (
            len(r0_non_empty) > 1 and len(set(r0_non_empty)) == 1
        )
        if is_r0_banner and len(r1_non_empty) >= 2:
            # Controlliamo se riga 1 contiene anni (2021, 2022, 2023) o parole chiave di header
            r1_joined = " ".join(r1_non_empty).lower()
            if any(k in r1_joined for k in ("2021", "2022", "2023", "tipologia", "categoria", "uomini", "donne", "totale", "importo")):
                # Evitiamo che il banner venga ripetuto su tutte le colonne nell'header combinato
                banner_text = r0_non_empty[0]
                rows[0] = [banner_text] + [""] * (len(rows[0]) - 1)
                flags[0] = True
                flags[1] = True

    # Caso 2: Intestazioni multi-livello Tassonomia UE (3 righe di header: Riga/Attività + CCM/CCA + Importo/%)
    if n_rows >= 3 and flags[0]:
        for r_idx in (1, 2):
            r_cells = [c.lower() for c in rows[r_idx] if c]
            if r_cells and all(
                any(tok in c for tok in ("importo", "%", "ccm", "cca", "mitigazione", "adattamento", "quota", "riga", "attività"))
                for c in r_cells
            ):
                flags[r_idx] = True

    # Garantisce che i flag column_header=True siano solo un prefisso contiguo all'inizio della tabella
    seen_data = False
    for i in range(n_rows):
        if flags[i] and not seen_data:
            continue
        seen_data = True
        flags[i] = False

    return flags


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


def _merge_table_chain(chain: list[TableItem]) -> tuple[TableData, int]:
    """Unisce una catena di TableItem [T_0, T_1, ..., T_k] in un unico TableData.

    Restituisce:
      (merged_table_data, num_stitched_rows)
    """
    target_cols = max(t.data.num_cols for t in chain)

    # Estrai griglia della prima tabella
    head_rows, head_hdr_flags = _extract_grid_and_headers(chain[0])
    head_hdr_flags = _normalize_header_flags(head_rows, head_hdr_flags)
    header_count = sum(1 for f in head_hdr_flags if f)

    # Verifica se la prima tabella ha una colonna interna sempre vuota nei dati (es. T2 col 1)
    empty_interior_col = _find_empty_interior_col(head_rows, header_count, chain[0].data.num_cols)

    merged_rows: list[list[str]] = [
        _pad_row_to_cols(r, target_cols, empty_interior_col) for r in head_rows
    ]
    merged_hdr_flags: list[bool] = list(head_hdr_flags)
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

    # Riapplica la normalizzazione degli header (utile quando T_0 aveva solo l'header, es. T29 o T76)
    merged_hdr_flags = _normalize_header_flags(merged_rows, merged_hdr_flags)

    return _build_table_data(merged_rows, merged_hdr_flags, target_cols), stitched_count


def fuse_document_tables(doc: DoclingDocument) -> FusionReport:
    """Esegue la pulizia e la fusione in-place delle tabelle multi-pagina su DoclingDocument.

    Operazioni effettuate sul documento:
      1. Elimina le tabelle vuote (0x0).
      2. Individua le catene di tabelle spezzate su pagine consecutive.
      3. Per ciascuna catena, aggiorna `head_table.data` con la griglia unificata e
         concatena `prov`, `captions` e `footnotes` delle tabelle successive.
      4. Normalizza anche le tabelle singole con banner di titolo in riga 0 (es. T41).
      5. Elimina dal documento le tabelle di continuazione e gli header di pagina intrusi.
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

    # 2. Costruisci mappa degli elementi intermedi nel documento
    between_map = _collect_intervening_items(doc)

    # 3. Raggruppa le tabelle non vuote in catene di continuazione
    chains: list[list[tuple[int, TableItem]]] = []
    noise_to_delete: list[NodeItem] = []

    for idx, table in non_empty_tables:
        if not chains:
            chains.append([(idx, table)])
            continue

        prev_idx, prev_table = chains[-1][-1]
        between_items = between_map.get((prev_table.self_ref, table.self_ref), [])
        can_merge, noise_items = should_merge_tables(prev_table, table, doc, between_items)
        if can_merge:
            chains[-1].append((idx, table))
            noise_to_delete.extend(noise_items)
        else:
            chains.append([(idx, table)])

    # 4. Esegui la fusione per ogni catena (e normalizza gli header delle tabelle singole)
    merged_infos: list[MergedTableInfo] = []

    for chain in chains:
        if len(chain) == 1:
            # Tabella singola: normalizza comunque eventuali banner + header (es. T33, T41)
            _, single_table = chain[0]
            rows, hdr_flags = _extract_grid_and_headers(single_table)
            new_flags = _normalize_header_flags(rows, hdr_flags)
            if new_flags != hdr_flags:
                single_table.data = _build_table_data(rows, new_flags, single_table.data.num_cols)
            continue

        head_orig_idx, head_table = chain[0]
        table_objs = [t for _, t in chain]
        orig_indices = [i for i, _ in chain]

        merged_data, stitched_count = _merge_table_chain(table_objs)
        head_table.data = merged_data

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

    # 5. Rimuovi dal documento le tabelle vuote, le tabelle di continuazione e il rumore intermedio
    all_deletions = items_to_delete + noise_to_delete
    if all_deletions:
        doc.delete_items(node_items=all_deletions)

    return FusionReport(
        original_table_count=original_count,
        empty_tables_removed=empty_indices,
        fused_table_count=len(doc.tables),
        merged_chains=merged_infos,
    )
