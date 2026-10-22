#!/usr/bin/env python3
"""Test completi del MarkdownChunker (Markdown + DoclingDocument)."""

import sys
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
    chunks = chunker.chunk(md)

    assert len(chunks) == 3
    assert chunks[0].headers == ["# Bilancio di Sostenibilità 2023", "## Capitolo 1: Governance"]
    assert chunks[1].headers == [
        "# Bilancio di Sostenibilità 2023",
        "## Capitolo 1: Governance",
        "### 1.1 Consiglio di Amministrazione",
    ]
    assert chunks[2].headers == ["# Bilancio di Sostenibilità 2023", "## Capitolo 2: Fornitori"]
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
    assert len(table_chunks) == 3, f"Attesi 3 chunk per 12 righe con max_table_rows=5, trovati {len(table_chunks)}"

    expected_ranges = ["1-5", "6-10", "11-12"]
    for idx, tc in enumerate(table_chunks, start=1):
        # 1. Ogni chunk ripete le intestazioni di sezione
        assert "# Approvvigionamenti" in tc.content
        assert "## [GRI 204-1]" in tc.content
        # 2. Ogni chunk ripete il titolo della tabella con Parte k/N
        assert "Elenco fornitori principali 2023" in tc.content
        assert f"Parte {idx}/3" in tc.content
        # 3. Ogni chunk ripete l'header delle colonne
        assert "| Ragione Sociale | Paese | Importo (Euro) |" in tc.content
        assert tc.metadata["part"] == idx
        assert tc.metadata["total_parts"] == 3
        assert tc.metadata["row_range"] == expected_ranges[idx - 1]
        assert tc.metadata["total_table_rows"] == 12
        assert tc.metadata["possible_incomplete_table"] is False

    print("[OK] test_long_table_row_splitting_repeats_header_and_title")


def test_incomplete_table_flag():
    """Verifica il flag possible_incomplete_table quando molte celle sono vuote."""
    md = """# Sezione Test

Tabella con dati mancanti

| Col A | Col B | Col C | Col D |
|---|---|---|---|
| X | | | |
| Y | | | |
"""
    chunker = MarkdownChunker(empty_cells_threshold=0.35)
    chunks = chunker.chunk(md)
    table_chunks = [c for c in chunks if c.chunk_type == "table"]
    assert len(table_chunks) == 1
    assert table_chunks[0].metadata["possible_incomplete_table"] is True
    assert table_chunks[0].metadata["empty_cells_ratio"] == 0.75
    print("[OK] test_incomplete_table_flag")


def test_docling_document_chunking():
    """Verifica chunk_document() sul documento Docling reale (se presente)."""
    doc_path = Path("data/processed/docling/dnf2023_test_p43_44/document_images.json")
    if not doc_path.exists():
        print("[SKIP] test_docling_document_chunking (file non presente)")
        return

    from docling_core.types.doc import DoclingDocument

    doc = DoclingDocument.load_from_json(doc_path)
    # Testiamo sia con max_table_rows=25 (tabelle intere) sia con max_table_rows=4 (split per righe)
    chunker = MarkdownChunker(max_table_rows=4)
    chunks = chunker.chunk_document(doc)

    table_chunks = [c for c in chunks if c.chunk_type == "table"]
    assert len(table_chunks) == 4  # T0 (8 righe -> 2 parti da 4) + T1 (7 righe -> 2 parti: 4 + 3)

    # Verifica che T1 (multi-pagina 43-44) abbia pages=[43, 44] e 2 image_paths in ogni suo sotto-chunk
    t1_chunks = [c for c in table_chunks if c.metadata["table_index"] == 1]
    assert len(t1_chunks) == 2
    for idx, tc in enumerate(t1_chunks, start=1):
        assert tc.metadata["pages"] == [43, 44]
        assert tc.metadata["page_range"] == "43-44"
        assert tc.metadata["image_paths"] == ["table_images/t1_p43.png", "table_images/t1_p44.png"]
        assert "% DEL VALORE EROGATO PER TIPOLOGIA DI FORNITURA" in tc.content
        assert "| Tipologia | Valore % 2021 | Valore % 2022 | Valore % 2023 |" in tc.content
        assert f"Parte {idx}/2" in tc.content

    print("[OK] test_docling_document_chunking")


if __name__ == "__main__":
    test_header_hierarchy()
    test_long_table_row_splitting_repeats_header_and_title()
    test_incomplete_table_flag()
    test_docling_document_chunking()
    print("\nTUTTI I TEST SUPERATI!")
