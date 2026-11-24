#!/usr/bin/env python3
"""Test del chunking su `DoclingDocument` sintetici (non richiedono la cartella `data/`):
intestazioni di tabella, divisione per gruppi e token, codici di riferimento, pagine per chunk.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from docling_core.types.doc import (  # noqa: E402
    BoundingBox,
    CoordOrigin,
    DocItemLabel,
    DoclingDocument,
    ProvenanceItem,
    Size,
    TableCell,
    TableData,
)
from docling_core.types.doc.common.content_layer import ContentLayer  # noqa: E402

from src.chunkers import GRI_PATTERN, MarkdownChunker, split_by_markers  # noqa: E402
from src.chunkers.table_utils import TableBlock, split_large_table  # noqa: E402
from src.extractors import (  # noqa: E402
    export_table_markdown,
    find_running_text_refs,
    fuse_document_tables,
    get_table_row_pages,
)


def _prov(page: int, start: int = 0, end: int = 0, top: float = 700.0, bottom: float = 600.0):
    bbox = BoundingBox(l=50, t=top, r=540, b=bottom, coord_origin=CoordOrigin.BOTTOMLEFT)
    return ProvenanceItem(page_no=page, bbox=bbox, charspan=(start, end))


def _table_data(rows: list[list[str]], n_header: int) -> TableData:
    cells = [
        TableCell(
            text=text,
            start_row_offset_idx=i,
            end_row_offset_idx=i + 1,
            start_col_offset_idx=j,
            end_col_offset_idx=j + 1,
            column_header=i < n_header,
        )
        for i, row in enumerate(rows)
        for j, text in enumerate(row)
    ]
    return TableData(table_cells=cells, num_rows=len(rows), num_cols=len(rows[0]))


def _new_doc(n_pages: int = 3) -> DoclingDocument:
    doc = DoclingDocument(name="sintetico")
    for p in range(1, n_pages + 1):
        doc.add_page(page_no=p, size=Size(width=595, height=842))
    return doc


def _chunk_single_table(rows: list[list[str]], n_header: int, **kwargs) -> list:
    doc = _new_doc(1)
    doc.add_heading("Personale", level=1, prov=_prov(1))
    doc.add_table(_table_data(rows, n_header), prov=_prov(1))
    return MarkdownChunker(**kwargs).chunk_document(doc, doc_id="t")


def _md_rows(chunk) -> list[str]:
    return [ln for ln in chunk.content.splitlines() if ln.startswith("|")]


def test_data_row_marked_as_header_is_restored():
    """Una riga dati marcata per errore come header torna tra i dati."""
    rows = [
        ["DIPENDENTI NEL TRIENNIO", "", "", "", ""],
        ["Inquadramento", "2022", "2022", "2023", "2023"],
        ["Numero totale di dirigenti", "21", "100,0%", "23", "100,0%"],
        ["- Totale Uomini", "19", "90,5%", "20", "87,0%"],
    ]
    (c,) = _chunk_single_table(rows, n_header=3)
    md = _md_rows(c)
    assert md[0] == "| Inquadramento | 2022 | 2022 | 2023 | 2023 |"
    assert md[2] == "| Numero totale di dirigenti | 21 | 100,0% | 23 | 100,0% |"
    assert c.metadata["table_title"] == "DIPENDENTI NEL TRIENNIO"
    assert c.metadata["total_table_rows"] == 2
    print("[OK] test_data_row_marked_as_header_is_restored")


def test_header_merged_with_first_row_is_split():
    """Header e prima riga dati nelle stesse celle (`2021 613`) vengono separati."""
    rows = [
        ["Categorie contrattuali Dipendenti a tempo indeterminato", "2021 613", "2022 624"],
        ["Di cui Uomini", "362", "370"],
    ]
    (c,) = _chunk_single_table(rows, n_header=1)
    md = _md_rows(c)
    assert md[0] == "| Categorie contrattuali | 2021 | 2022 |"
    assert md[2] == "| Dipendenti a tempo indeterminato | 613 | 624 |"

    # Lo stesso se la tabella passa prima dalla fusione (titolo a tutta larghezza in riga 0)
    doc = _new_doc(1)
    doc.add_table(_table_data([["DIPENDENTI", "", ""]] + rows, 2), prov=_prov(1))
    fuse_document_tables(doc)
    (c,) = MarkdownChunker().chunk_document(doc, doc_id="t")
    assert _md_rows(c)[0] == "| Categorie contrattuali | 2021 | 2022 |"
    print("[OK] test_header_merged_with_first_row_is_split")


def test_section_rows_are_not_titles_nor_missing_data():
    """Le righe a celle unite numerate non sono titolo né celle mancanti."""
    rows = [
        ["1) Mitigazione"] * 4,
        ["- fatturato: %", "1,10%", "Attività di transizione: %", "0,01%"],
        ["2) Adattamento"] * 4,
        ["- fatturato: %", "0,01%", "Attività di transizione: %", "0,00%"],
    ]
    (c,) = _chunk_single_table(rows, n_header=2)
    md = _md_rows(c)
    assert md[0] == "| Colonna 1 | Colonna 2 | Colonna 3 | Colonna 4 |"
    assert md[2] == "| 1) Mitigazione |  |  |  |"
    assert md[4] == "| 2) Adattamento |  |  |  |"
    assert c.metadata["table_title"] == "Personale"
    assert c.metadata["possible_incomplete_table"] is False

    # Righe-sezione a due blocchi, senza header segnalato da Docling
    pairs = [["Quota A", "Quota A", "Valore A", "Valore A"], ["- voce", "1,1%", "- voce", "20,5"]]
    pairs += [["Quota B", "Quota B", "Valore B", "Valore B"], ["- voce", "2,2%", "- voce", "30,5"]]
    (c,) = _chunk_single_table(pairs, n_header=0)
    assert _md_rows(c)[0] == "| Colonna 1 | Colonna 2 | Colonna 3 | Colonna 4 |"
    assert c.metadata["total_table_rows"] == 4
    print("[OK] test_section_rows_are_not_titles_nor_missing_data")


def test_regular_headers_are_unchanged():
    """Banner + anni, header su due livelli e prima riga non marcata restano come prima."""
    (c,) = _chunk_single_table(
        [["CLIENTI DEL GRUPPO"] * 3, ["Categoria", "2022", "2023"], ["Vita", "47.155", "47.985"]], 2
    )
    assert _md_rows(c)[0] == "| Categoria | 2022 | 2023 |"
    assert c.metadata["table_title"] == "CLIENTI DEL GRUPPO"

    (c,) = _chunk_single_table(
        [["Tipologia", "2023", "2023"], ["Reclami", "Sara Ass", "Totale"], ["Accolti", "370", "372"]], 2
    )
    assert _md_rows(c)[0] == "| Tipologia - Reclami | 2023 - Sara Ass | 2023 - Totale |"

    (c,) = _chunk_single_table([["Voce", "Testo"], ["A", "descrizione"]], 0)
    assert _md_rows(c)[0] == "| Voce | Testo |"
    print("[OK] test_regular_headers_are_unchanged")


def test_numeric_column_names_are_kept():
    """Nomi di colonna numerici (taglie, trimestri, periodi) non vanno scambiati per dati."""
    for header in (
        ["Size", "36", "38", "40"],
        ["Voce", "1", "2", "3"],
        ["Item", "Q1", "Q2", "Q3"],
        ["Item", "FY2022", "FY2023", "FY2024"],
        ["Periodo", "2021/22", "2022/23", "2023/24"],
    ):
        (c,) = _chunk_single_table([header, ["A", "10", "12", "9"], ["B", "3", "4", "5"]], 1)
        assert _md_rows(c)[0] == "| " + " | ".join(header) + " |", header
    # Anche sotto un titolo a tutta larghezza: un intero semplice è ambiguo, quindi non si tocca
    rows = [["STOCK BY SIZE"] * 4, ["Size", "36", "38", "40"], ["Shoes", "10", "12", "9"]]
    (c,) = _chunk_single_table(rows, 2)
    assert _md_rows(c)[0] == "| Size | 36 | 38 | 40 |"
    print("[OK] test_numeric_column_names_are_kept")


def test_table_without_header_keeps_all_rows():
    """Se Docling non segnala header e la prima riga è un dato, nessuna riga diventa header."""
    (c,) = _chunk_single_table([["Width", "10 mm"], ["Height", "20 mm"], ["Depth", "5 mm"]], 0)
    assert _md_rows(c)[0] == "| Colonna 1 | Colonna 2 |"
    assert c.metadata["total_table_rows"] == 3
    print("[OK] test_table_without_header_keeps_all_rows")


def test_groups_in_any_language():
    """I gruppi si riconoscono dalle etichette che si ripetono, non dalla lingua."""
    rows = [["Item", "2023"]]
    for name in ("managers", "clerks"):
        rows += [[f"Total {name}", "10"], ["of which men", "6"], ["of which women", "4"],
                 ["Under 30", "2"], ["Over 50", "3"]]
    chunks = _chunk_single_table(rows, 1, max_table_rows=7)
    assert [c.metadata["row_range"] for c in chunks] == ["1-5", "6-10"]
    assert _md_rows(chunks[1])[2] == "| Total clerks | 10 |"
    print("[OK] test_groups_in_any_language")


def test_flat_list_with_repeated_names_is_not_grouped():
    """In un elenco in cui un nome ricorre per caso non si inventano righe madre."""
    cities = ["Roma", "Milano", "Torino", "Roma", "Napoli", "Milano", "Bari", "Roma"]
    rows = [["Città", "Importo"]] + [[c, str(i)] for i, c in enumerate(cities)]
    chunks = _chunk_single_table(rows, 1, max_table_rows=5)
    assert [c.metadata["row_range"] for c in chunks] == ["1-5", "6-8"]
    assert len(_md_rows(chunks[1])) == 2 + 3
    print("[OK] test_flat_list_with_repeated_names_is_not_grouped")


def test_split_at_group_start():
    """Il taglio cade all'inizio di un gruppo, non in mezzo alle sue righe di dettaglio."""
    rows = [["Voce", "2023"]]
    for name in ("dirigenti", "funzionari", "impiegati"):
        rows.append([f"Numero totale di {name}", "10"])
        rows += [["Sotto i 30 anni di età", "1"], ["Di cui Uomini", "1"], ["Di cui Donne", "0"]]
    chunks = _chunk_single_table(rows, n_header=1, max_table_rows=10)
    assert [c.metadata["row_range"] for c in chunks] == ["1-8", "9-12"]
    assert _md_rows(chunks[1])[2] == "| Numero totale di impiegati | 10 |"
    print("[OK] test_split_at_group_start")


def test_context_rows_when_group_is_cut():
    """Se il gruppo è più lungo del limite, la parte successiva ripete le righe madre."""
    rows = [["Total staff", "9"]] + [[f"- group {i}", str(i)] for i in range(7)]
    parts = split_large_table(TableBlock(headers=["Item", "2023"], rows=rows), max_rows=5)
    assert len(parts) == 2
    assert parts[1].context_rows == [["Total staff", "9"]]
    assert (parts[1].row_start, parts[1].row_end, len(parts[1].rows)) == (6, 8, 3)
    assert "| Total staff | 9 |" in parts[1].to_markdown()
    assert sum(len(p.rows) for p in parts) == len(rows)
    print("[OK] test_context_rows_when_group_is_cut")


def test_table_token_limit():
    """Nessun chunk di tabella supera `max_table_tokens` e nessuna riga va persa."""
    rows = [["Tema", "Descrizione"]] + [[f"Tema {i}", "parola " * 30] for i in range(12)]

    def count(text: str) -> int:
        return len(text.split())

    chunks = _chunk_single_table(rows, n_header=1, max_table_tokens=150, count_tokens=count)
    assert len(chunks) > 1
    assert all(count(c.content) <= 150 for c in chunks)
    assert sum(c.metadata["row_count"] for c in chunks) == 12
    print("[OK] test_table_token_limit")


def test_parse_reference_codes():
    assert GRI_PATTERN.parse("207-1, 207.2, 207-1") == ["GRI 207-1", "GRI 207-2"]
    assert GRI_PATTERN.parse("2021") == []
    assert split_by_markers("Testo [GRI 401-3]", [GRI_PATTERN]) == [
        (0, "Testo ", []),
        (6, "", ["GRI 401-3"]),
    ]
    assert split_by_markers("[GRI 2021] nota", [GRI_PATTERN]) == [(0, "[GRI 2021] nota", [])]
    assert split_by_markers("[GRI 401-3] nota", []) == [(0, "[GRI 401-3] nota", [])]
    print("[OK] test_parse_reference_codes")


def test_reference_codes_are_opt_in():
    """Senza configurazione i codici restano testo e non diventano titoli di sezione."""
    doc = _new_doc(1)
    doc.add_heading("Fornitori", level=1, prov=_prov(1))
    doc.add_heading("[GRI 204-1]", level=1, prov=_prov(1))
    doc.add_text(DocItemLabel.TEXT, "Elenco dei fornitori.", prov=_prov(1, 0, 21))
    (c,) = MarkdownChunker().chunk_document(doc, doc_id="t")
    assert (c.metadata["section"], c.metadata["reference_codes"]) == ("Fornitori", [])
    assert "[GRI 204-1]" in c.content
    (c,) = MarkdownChunker(reference_patterns=[GRI_PATTERN]).chunk_document(doc, doc_id="t")
    assert (c.metadata["section"], c.metadata["reference_codes"]) == ("Fornitori", ["GRI 204-1"])
    print("[OK] test_reference_codes_are_opt_in")


def test_reference_markers_become_metadata():
    """Titolo, testo, coda di paragrafo o piè di pagina: il marcatore vale per ciò che segue."""
    doc = _new_doc(2)
    doc.add_heading("La gestione fiscale", level=1, prov=_prov(1))
    doc.add_heading("[GRI 207-1, 207.2]", level=1, prov=_prov(1))
    doc.add_text(DocItemLabel.TEXT, "Politica fiscale del Gruppo.", prov=_prov(1, 0, 28))
    doc.add_text(
        DocItemLabel.PAGE_FOOTER,
        "[GRI 205-3]",
        prov=_prov(1, 0, 11),
        content_layer=ContentLayer.FURNITURE,
    )
    doc.add_text(DocItemLabel.TEXT, "Nessun episodio.", prov=_prov(2, 0, 16))
    doc.add_text(DocItemLabel.TEXT, "Segue la tabella [GRI 207-4]", prov=_prov(2, 0, 28))
    doc.add_table(_table_data([["Voce", "2023"], ["Imposte", "18"]], 1), prov=_prov(2))
    doc.add_text(
        DocItemLabel.PAGE_FOOTER, "Pagina 2", prov=_prov(2, 0, 8), content_layer=ContentLayer.FURNITURE
    )
    doc.add_heading("x", level=1, prov=_prov(2))
    doc.add_heading("La privacy", level=1, prov=_prov(2))
    doc.add_text(DocItemLabel.TEXT, "Protezione dei dati.", prov=_prov(2, 0, 20))

    chunks = MarkdownChunker(reference_patterns=[GRI_PATTERN]).chunk_document(doc, doc_id="t")
    got = [(c.chunk_type, c.metadata["section"], c.metadata["reference_codes"]) for c in chunks]
    assert got == [
        ("text", "La gestione fiscale", ["GRI 207-1", "GRI 207-2"]),
        ("text", "La gestione fiscale", ["GRI 205-3"]),
        ("table", "La gestione fiscale", ["GRI 207-4"]),
        ("text", "La privacy", []),
    ], got
    assert "[GRI 207-4]" in chunks[2].content
    assert "Pagina 2" not in "".join(c.content for c in chunks)
    print("[OK] test_reference_markers_become_metadata")


def test_reference_marker_before_heading_is_kept():
    """Un marcatore seguito subito da un titolo vale per il contenuto sotto quel titolo."""
    doc = _new_doc(1)
    doc.add_heading("Capitolo", level=1, prov=_prov(1))
    doc.add_text(DocItemLabel.TEXT, "[GRI 2-23]", prov=_prov(1, 0, 10))
    doc.add_heading("Le politiche", level=1, prov=_prov(1))
    doc.add_text(DocItemLabel.TEXT, "Elenco delle politiche.", prov=_prov(1, 0, 23))
    (c,) = MarkdownChunker(reference_patterns=[GRI_PATTERN]).chunk_document(doc, doc_id="t")
    assert (c.metadata["section"], c.metadata["reference_codes"]) == ("Le politiche", ["GRI 2-23"])
    print("[OK] test_reference_marker_before_heading_is_kept")


def test_pages_are_per_chunk():
    """Ogni parte di una sezione lunga riporta solo le proprie pagine, anche dentro un paragrafo."""
    part1, part2 = "alfa " * 100, "beta " * 100
    text = (part1 + part2).strip()
    doc = _new_doc(3)
    doc.add_heading("Sezione", level=1, prov=_prov(1))
    item = doc.add_text(DocItemLabel.TEXT, text, prov=_prov(1, 0, len(part1) - 1))
    item.prov.append(_prov(2, len(part1), len(text)))
    doc.add_text(DocItemLabel.TEXT, ("gamma " * 100).strip(), prov=_prov(3, 0, 599))

    chunks = MarkdownChunker(chunk_size=500, chunk_overlap=0).chunk_document(doc, doc_id="t")
    for c in chunks:
        expected = {p for w, p in (("alfa", 1), ("beta", 2), ("gamma", 3)) if w in c.content}
        assert set(c.metadata["pages"]) == expected, (c.metadata["pages"], expected)
    assert chunks[0].metadata["pages"] == [1]
    assert chunks[-1].metadata["pages"] == [3]
    print("[OK] test_pages_are_per_chunk")


def test_row_pages_after_fusion():
    """La fusione registra la pagina di ogni riga e il chunker la usa per le parti di tabella."""
    doc = _new_doc(2)
    doc.add_heading("Agenzie", level=1, prov=_prov(1))
    head = [["Regione", "2023"]] + [[f"Regione {i}", str(i)] for i in range(4)]
    cont = [[f"Regione {i}", str(i)] for i in range(4, 8)]
    doc.add_table(_table_data(head, 1), prov=_prov(1, top=400, bottom=60))
    doc.add_table(_table_data(cont, 0), prov=_prov(2, top=760, bottom=500))

    report = fuse_document_tables(doc)
    assert report.fused_table_count == 1
    assert get_table_row_pages(doc.tables[0]) == [1] * 5 + [2] * 4
    assert "rag__" not in export_table_markdown(doc.tables[0], doc)

    chunks = MarkdownChunker(max_table_rows=4).chunk_document(doc, doc_id="t")
    assert [c.metadata["pages"] for c in chunks] == [[1], [2]]
    print("[OK] test_row_pages_after_fusion")


def _is_fused(
    width=595, height=842, *, top=0.10, bottom=0.07, head=None, cont=None, cont_header=0,
    cont_caption=None, between=None,
) -> bool:
    """Crea due tabelle su pagine consecutive e verifica se vengono fuse."""
    doc = _new_doc(0)
    for p in (1, 2, 3, 4):
        doc.add_page(page_no=p, size=Size(width=width, height=height))

    def prov(page, t, b):
        bbox = BoundingBox(
            l=width * 0.1, t=height * (1 - t), r=width * 0.9, b=height * (1 - b),
            coord_origin=CoordOrigin.BOTTOMLEFT,
        )
        return ProvenanceItem(page_no=page, bbox=bbox, charspan=(0, 0))

    doc.add_table(
        _table_data(head or [["Region", "2023"], ["A", "1"], ["B", "2"]], 1),
        prov=prov(1, 0.5, 1 - bottom),
    )
    for label, text, t, b in between or []:
        doc.add_text(label, text, prov=prov(2, t, b))
    caption = None
    if cont_caption:
        caption = doc.add_text(DocItemLabel.CAPTION, cont_caption, prov=prov(2, 0.08, 0.09))
    doc.add_table(
        _table_data(cont or [["C", "3"], ["D", "4"]], cont_header),
        prov=prov(2, top, 0.5),
        caption=caption,
    )
    # La stessa testata compare anche sulle pagine successive
    for label, text, t, b in between or []:
        if b <= 0.12:
            doc.add_text(label, text, prov=prov(3, t, b))
            doc.add_text(label, text, prov=prov(4, t, b))
    return fuse_document_tables(doc).fused_table_count == 1


def test_fusion_works_on_any_page_format():
    for name, w, h in (("A4", 595, 842), ("Letter", 612, 792), ("A4 orizz.", 842, 595),
                       ("A3", 842, 1191), ("slide 16:9", 960, 540)):
        assert _is_fused(w, h), name
        assert not _is_fused(w, h, bottom=0.35), f"{name}: la prima finisce a metà pagina"
        assert not _is_fused(w, h, top=0.30), f"{name}: la seconda non inizia in cima"
    print("[OK] test_fusion_works_on_any_page_format")


def test_different_tables_are_not_fused():
    """Una tabella con intestazione, titolo o didascalia propri non è una continuazione."""
    new_header = [["Product", "Price"], ["C", "3"], ["D", "4"]]
    assert not _is_fused(cont=new_header, cont_header=1)
    assert not _is_fused(cont=new_header, cont_header=0)
    assert not _is_fused(cont=[["PRICE LIST", "PRICE LIST"], ["Product", "2024"], ["C", "3"]], cont_header=2)
    assert not _is_fused(cont_caption="Table 7 - Price list")
    # Una riga-sezione dello stesso tipo di quelle già presenti è una continuazione
    sections = [["Quota A", "Quota A", "Valore A", "Valore A"], ["- voce", "1,1%", "- voce", "20"]]
    cont = [["Quota B", "Quota B", "Valore B", "Valore B"], ["- voce", "2,2%", "- voce", "30"]]
    assert _is_fused(head=sections, cont=cont)
    # Header ripetuto o prima riga dati marcata per errore come header: è una continuazione
    assert _is_fused(cont=[["Region", "2023"], ["C", "3"], ["D", "4"]], cont_header=1)
    assert _is_fused(cont=[["C", "3"], ["D", "4"]], cont_header=1)
    print("[OK] test_different_tables_are_not_fused")


def test_running_headers_are_found_by_repetition():
    """Testate di qualunque testo e lingua non bloccano la fusione; un vero paragrafo sì."""
    header = (DocItemLabel.SECTION_HEADER, "ACME Corp - Annual Report", 0.03, 0.05)
    assert _is_fused(between=[header])
    assert not _is_fused(between=[(DocItemLabel.TEXT, "A real paragraph.", 0.20, 0.25)])

    doc = _new_doc(12)

    def at(page, t, b):
        bbox = BoundingBox(l=50, t=842 * (1 - t), r=500, b=842 * (1 - b), coord_origin=CoordOrigin.BOTTOMLEFT)
        return ProvenanceItem(page_no=page, bbox=bbox, charspan=(0, 0))

    running = [doc.add_text(DocItemLabel.TEXT, f"Handbuch - Seite {p}", prov=at(p, 0.95, 0.97)) for p in (2, 3, 4)]
    number = doc.add_text(DocItemLabel.TEXT, "7", prov=at(7, 0.95, 0.97))
    # Codici diversi in cima a pagine vicine non sono una testata, anche se differiscono solo nelle cifre
    codes = [doc.add_text(DocItemLabel.TEXT, f"[REF 20{n}-1]", prov=at(p, 0.05, 0.07)) for n, p in ((1, 8), (4, 9), (9, 10))]
    chapters = [doc.add_heading(f"Chapter {n}", prov=at(p, 0.05, 0.08)) for n, p in ((1, 5), (2, 6), (3, 7))]
    far = [doc.add_heading("Summary", prov=at(p, 0.05, 0.08)) for p in (3, 7, 11)]
    refs = find_running_text_refs(doc)
    assert {i.self_ref for i in running} | {number.self_ref} == refs
    assert not {i.self_ref for i in chapters + far + codes} & refs

    # Documento corto: bastano due pagine con lo stesso testo alla stessa altezza
    doc = _new_doc(2)
    short = [doc.add_text(DocItemLabel.TEXT, "Contratto n. 88/2025", prov=at(p, 0.04, 0.06)) for p in (1, 2)]
    doc.add_text(DocItemLabel.TEXT, "Ultima riga.", prov=at(1, 0.93, 0.95))
    assert {i.self_ref for i in short} == find_running_text_refs(doc)
    print("[OK] test_running_headers_are_found_by_repetition")


if __name__ == "__main__":
    test_data_row_marked_as_header_is_restored()
    test_header_merged_with_first_row_is_split()
    test_section_rows_are_not_titles_nor_missing_data()
    test_regular_headers_are_unchanged()
    test_numeric_column_names_are_kept()
    test_table_without_header_keeps_all_rows()
    test_groups_in_any_language()
    test_flat_list_with_repeated_names_is_not_grouped()
    test_split_at_group_start()
    test_context_rows_when_group_is_cut()
    test_table_token_limit()
    test_parse_reference_codes()
    test_reference_codes_are_opt_in()
    test_reference_markers_become_metadata()
    test_reference_marker_before_heading_is_kept()
    test_pages_are_per_chunk()
    test_row_pages_after_fusion()
    test_fusion_works_on_any_page_format()
    test_different_tables_are_not_fused()
    test_running_headers_are_found_by_repetition()
    print("\nTUTTI I TEST SUPERATI!")
