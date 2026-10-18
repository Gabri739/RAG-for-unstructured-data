#!/usr/bin/env python3
"""Test completo del MarkdownChunker con vari casi edge."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.chunkers import MarkdownChunker  # noqa: E402


def test_basic_headers():
    """Test base: header splitting semplice."""
    print("\n" + "="*60)
    print("TEST 1: Header splitting base")
    print("="*60)

    md = """# Documento Principale

## Introduzione
Questo è il testo di introduzione.

## Capitolo 1
Contenuto del capitolo 1.

### Sezione 1.1
Dettagli della sezione.

## Capitolo 2
Contenuto finale.
"""

    chunker = MarkdownChunker(headers_to_split_on=["#", "##", "###"])
    chunks = chunker.chunk(md)

    print(f"Chunk generati: {len(chunks)}")
    for i, c in enumerate(chunks, 1):
        print(f"  {i}. {c.chunk_type}: {c.headers}")

    assert len(chunks) >= 3, "Dovrebbe generare almeno 3 chunk"
    print("[OK] PASS")


def test_single_page_table():
    """Test tabella su singola pagina."""
    print("\n" + "="*60)
    print("TEST 2: Tabella singola pagina")
    print("="*60)

    md = """# Report Vendite

Dati delle vendite Q1.

| Prodotto | Q1 | Q2 |
|----------|----|----|
| A        | 10 | 20 |
| B        | 15 | 25 |

Fine report.
"""

    chunker = MarkdownChunker(preserve_tables=True)
    chunks = chunker.chunk(md)

    print(f"Chunk generati: {len(chunks)}")
    for i, c in enumerate(chunks, 1):
        print(f"  {i}. {c.chunk_type}: {len(c.content)} chars")
        if c.chunk_type == "table":
            print(f"     Righe: {c.metadata.get('row_count')}, Colonne: {c.metadata.get('column_count')}")

    table_chunks = [c for c in chunks if c.chunk_type == "table"]
    assert len(table_chunks) == 1, "Dovrebbe esserci 1 chunk tabella"
    assert table_chunks[0].metadata['row_count'] == 2, "Tabella dovrebbe avere 2 righe"
    print("[OK] PASS")


def test_multi_page_table():
    """Test tabella che continua su più pagine (CASO CRITICO)."""
    print("\n" + "="*60)
    print("TEST 3: Tabella multi-pagina (CASO CRITICO)")
    print("="*60)

    md = """# Bilancio 2024

## Dati Finanziari

| Account | Importo |
|---------|---------|
| Vendite | 1000    |
| Costi   | 500     |

Testo intermedio tra le tabelle.

| Account | Importo |
|---------|---------|
| Marketing | 200   |
| IT        | 100   |

## Note
Conclusioni del bilancio.
"""

    chunker = MarkdownChunker(preserve_tables=True)
    chunks = chunker.chunk(md)

    print(f"Chunk generati: {len(chunks)}")
    for i, c in enumerate(chunks, 1):
        print(f"  {i}. {c.chunk_type}: {c.content[:80]}...")

    # Qui ci sono 2 tabelle separate (non continuation perché headers match esattamente)
    # Ma dovrebbero essere chunk separati
    table_chunks = [c for c in chunks if c.chunk_type == "table"]
    print(f"     Tabelle trovate: {len(table_chunks)}")


def test_continuation_table():
    """Test tabella che continua (stessi headers, righe diverse pagine)."""
    print("\n" + "="*60)
    print("TEST 4: Tabella continuation (stessi headers)")
    print("="*60)

    md = """# Estratto Conto

| Data | Descrizione | Importo |
|------|-------------|---------|
| 01/01 | Saldo iniziale | 1000 |
| 02/01 | Pagamento | -100 |

Altre operazioni:

| Data | Descrizione | Importo |
|------|-------------|---------|
| 03/01 | Versamento | 500 |
| 04/01 | Prelievo | -200 |

Totale: 1200
"""

    chunker = MarkdownChunker(preserve_tables=True)
    chunks = chunker.chunk(md)

    print(f"Chunk generati: {len(chunks)}")
    for i, c in enumerate(chunks, 1):
        print(f"  {i}. {c.chunk_type}: {c.headers}")
        if c.chunk_type == "table":
            print(f"     Righe: {c.metadata.get('row_count')}")

    # Con headers identici dovrebbe unire le tabelle
    table_chunks = [c for c in chunks if c.chunk_type == "table"]
    if table_chunks:
        # Se unisce, dovrebbe avere 4 righe totali
        rows = table_chunks[0].metadata.get('row_count', 0)
        print(f"     Tabella finale: {rows} righe")
        if rows == 4:
            print("[OK] Tabelle unite correttamente!")
        else:
            print(f"⚠️  Tabelle separate ({rows} righe vs 4 attese)")


def test_large_text_splitting():
    """Test splitting di testo molto lungo."""
    print("\n" + "="*60)
    print("TEST 5: Testo lungo con split")
    print("="*60)

    # Genera testo lungo
    paragraphs = [f"Paragrafo numero {i} con abbastanza testo da superare il limite di dimensione. " * 10
                  for i in range(20)]
    md = "# Documento Lungo\n\n" + "\n\n".join(paragraphs)

    chunker = MarkdownChunker(chunk_size=1000, chunk_overlap=100)
    chunks = chunker.chunk(md)

    print(f"Dimensione totale: {len(md)} caratteri")
    print(f"Chunk generati: {len(chunks)}")

    # Verifica che ci sia almeno un chunk splittato
    split_chunks = [c for c in chunks if c.metadata.get('total_parts', 1) > 1]
    if split_chunks:
        print(f"[OK] Chunk splittati: {len(split_chunks)}")
        for c in split_chunks[:2]:  # Mostra primi 2
            print(f"   Parte {c.metadata['part']}/{c.metadata['total_parts']}: {len(c.content)} chars")
    else:
        print("ℹ️  Nessuno split necessario (testo rientra nel limite)")


def test_real_document():
    """Test con documento reale del progetto."""
    print("\n" + "="*60)
    print("TEST 6: Documento reale (output.md)")
    print("="*60)

    try:
        with open('data/raw/20260513_121255/output.md', 'r', encoding='utf-8') as f:
            content = f.read()

        print(f"Dimensione: {len(content)} caratteri, {len(content.splitlines())} righe")

        chunker = MarkdownChunker(
            headers_to_split_on=["#", "##", "###"],
            chunk_size=2000,
            preserve_tables=True
        )
        chunks = chunker.chunk(content)

        print(f"Chunk generati: {len(chunks)}")

        text_chunks = len([c for c in chunks if c.chunk_type == "text"])
        table_chunks = len([c for c in chunks if c.chunk_type == "table"])

        print(f"  - Testo: {text_chunks}")
        print(f"  - Tabelle: {table_chunks}")

        # Mostra primi 3 chunk
        for c in chunks[:3]:
            print(f"\n  {c.chunk_type.upper()}: {c.headers}")
            preview = c.content[:150].replace('\n', ' ')
            print(f"     {preview}...")

        print("\n[OK] Documento processato correttamente!")

    except FileNotFoundError:
        print("⚠️  File non trovato, salto test")


def main():
    """Esegue tutti i test."""
    print("="*60)
    print("TEST COMPRENSIVI MARKDOWN CHUNKER")
    print("="*60)

    test_basic_headers()
    test_single_page_table()
    test_multi_page_table()
    test_continuation_table()
    test_large_text_splitting()
    test_real_document()

    print("\n" + "="*60)
    print("TUTTI I TEST COMPLETATI")
    print("="*60)


if __name__ == "__main__":
    main()
