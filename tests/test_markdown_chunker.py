#!/usr/bin/env python3
"""Test completi del MarkdownChunker (Markdown + DoclingDocument + metadati avanzati)."""

import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.chunkers import MarkdownChunker  # noqa: E402


def test_header_hierarchy():
    """Verifica che lo stack delle intestazioni (# -> ## -> ###) venga mantenuto in ogni chunk."""
    md = """# Bilancio di Sostenibilità 2023

## Capitolo 1: Governance
Testo introduttivo sulla governance aziendale.

### 1.1 Consiglio di Amministrazione
Dettagli sul CdA.

## Capitolo 2: Fornitori
Analisi della catena di fornitura.
"""
    chunker = MarkdownChunker(headers_to_split_on=["#", "##", "###"])
    chunks = chunker.chunk(md, doc_id="dnf2023", source_file="source.pdf")

    assert len(chunks) == 3
    assert chunks[0].headers == ["# Bilancio di Sostenibilità 2023", "## Capitolo 1: Governance"]
    assert chunks[1].headers == [
        "# Bilancio di Sostenibilità 2023",
        "## Capitolo 1: Governance",
        "### 1.1 Consiglio di Amministrazione",
    ]
    assert chunks[2].headers == ["# Bilancio di Sostenibilità 2023", "## Capitolo 2: Fornitori"]
    assert chunks[0].metadata["doc_id"] == "dnf2023"
    assert chunks[0].metadata["chunk_id"] == "dnf2023__chunk_0000"
    # Verifica che point_id sia un UUID valido per Qdrant
    uuid.UUID(chunks[0].metadata["point_id"])
    print("[OK] test_header_hierarchy")


def test_long_table_row_splitting_repeats_header_and_title():
    """Verifica che una tabella lunga venga divisa per righe con header e titolo ripetuti in ogni chunk."""
    rows_md = "\n".join(f"| Fornitore {i} | Italia | {i * 100} |" for i in range(1, 13))
    md = f"""# Approvvigionamenti

## [GRI 204-1]

Elenco fornitori principali 2023

| Ragione Sociale | Paese | Importo (Euro) |
|---|---|---|
{rows_md}
"""
    chunker = MarkdownChunker(max_table_rows=5)
    chunks = chunker.chunk(md)

    table_chunks = [c for c in chunks if c.chunk_type == "table"]
    assert len(table_chunks) == 3

    expected_ranges = ["1-5", "6-10", "11-12"]
    for idx, tc in enumerate(table_chunks, start=1):
        assert "# Approvvigionamenti" in tc.content
        assert "## [GRI 204-1]" in tc.content
        assert "Elenco fornitori principali 2023" in tc.content
        assert f"Parte {idx}/3" in tc.content
        assert "| Ragione Sociale | Paese | Importo (Euro) |" in tc.content
        assert tc.metadata["part"] == idx
        assert tc.metadata["total_parts"] == 3
        assert tc.metadata["row_range"] == expected_ranges[idx - 1]
        assert tc.metadata["total_table_rows"] == 12
        assert tc.metadata["possible_incomplete_table"] is False
        assert tc.metadata["empty_columns"] == []

    print("[OK] test_long_table_row_splitting_repeats_header_and_title")


def test_empty_columns_and_table_level_flag():
    """Verifica:
    1) Tabella a 3 colonne con 1 colonna icone vuota (33.3% vuote globali < 35%) -> scatta grazie a empty_columns!
    2) Tabella corta (2 righe) -> empty_columns non scatta (min_rows=3).
    3) Colonna 'Note' vuota -> ignorata, non fa scattare falsi positivi.
    4) Coerenza del flag a livello di tabella quando viene divisa in più parti.
    """
    # Caso 1: 3 colonne, colonna centrale "Icona" vuota su 6 righe (33.3% globale < 0.35, ma 100% sulla colonna)
    rows_icon = "\n".join(f"| Tema {i} | | Descrizione {i} |" for i in range(1, 7))
    md_icon = f"""# Materialità

| Tema | Icona | Descrizione |
|---|---|---|
{rows_icon}
"""
    chunker = MarkdownChunker(max_table_rows=3)
    chunks_icon = [c for c in chunker.chunk(md_icon) if c.chunk_type == "table"]
    assert len(chunks_icon) == 2
    for tc in chunks_icon:
        assert tc.metadata["empty_cells_ratio"] == round(1 / 3, 4)  # 0.3333 < 0.35
        assert tc.metadata["empty_columns"] == ["Icona"]
        assert tc.metadata["possible_incomplete_table"] is True

    # Caso 2: Tabella di sole 2 righe con una colonna vuota (25% globale) -> min_rows=3 evita falso positivo
    md_short = """# Breve

| A | B | C | D |
|---|---|---|---|
| 1 | | 3 | 4 |
| 5 | | 7 | 8 |
"""
    tc_short = [c for c in chunker.chunk(md_short) if c.chunk_type == "table"][0]
    assert tc_short.metadata["empty_columns"] == []
    assert tc_short.metadata["possible_incomplete_table"] is False

    # Caso 3: Colonna "Note" vuota su 4 righe -> ignorata come colonna opzionale
    md_notes = """# Con Note

| Voce | 2022 | 2023 | Note |
|---|---|---|---|
| A | 10 | 12 | |
| B | 20 | 22 | |
| C | 30 | 32 | |
| D | 40 | 42 | |
"""
    tc_notes = [c for c in chunker.chunk(md_notes) if c.chunk_type == "table"][0]
    assert tc_notes.metadata["empty_columns"] == []
    assert tc_notes.metadata["possible_incomplete_table"] is False

    print("[OK] test_empty_columns_and_table_level_flag")


def test_docling_document_chunking():
    """Verifica chunk_document() sul documento Docling reale (se presente)."""
    out_dir = Path("data/processed/docling/dnf2023_test_p43_44")
    doc_path = out_dir / "document_images.json"
    if not doc_path.exists():
        print("[SKIP] test_docling_document_chunking (file non presente)")
        return

    from docling_core.types.doc import DoclingDocument

    doc = DoclingDocument.load_from_json(doc_path)
    chunker = MarkdownChunker(max_table_rows=4)
    chunks = chunker.chunk_document(
        doc,
        doc_id="dnf2023",
        source_file="source.pdf",
        table_images_manifest=out_dir / "table_images.json",
    )

    table_chunks = [c for c in chunks if c.chunk_type == "table"]
    assert len(table_chunks) == 4

    t1_chunks = [c for c in table_chunks if c.metadata["table_index"] == 1]
    assert len(t1_chunks) == 2
    for idx, tc in enumerate(t1_chunks, start=1):
        assert tc.metadata["pages"] == [43, 44]
        assert tc.metadata["page_start"] == 43
        assert tc.metadata["page_end"] == 44
        assert tc.metadata["page_range"] == "43-44"
        assert tc.metadata["image_paths"] == ["table_images/t1_p43.png", "table_images/t1_p44.png"]
        assert "% DEL VALORE EROGATO PER TIPOLOGIA DI FORNITURA" in tc.content
        assert "| Tipologia | Valore % 2021 | Valore % 2022 | Valore % 2023 |" in tc.content
        assert f"Parte {idx}/2" in tc.content
        # Verifica anche la conversione flat per ChromaDB
        flat = tc.to_vector_metadata(flat=True)
        assert all(isinstance(v, (str, int, float, bool)) for v in flat.values())

    print("[OK] test_docling_document_chunking")


if __name__ == "__main__":
    test_header_hierarchy()
    test_long_table_row_splitting_repeats_header_and_title()
    test_empty_columns_and_table_level_flag()
    test_docling_document_chunking()
    print("\nTUTTI I TEST SUPERATI!")
