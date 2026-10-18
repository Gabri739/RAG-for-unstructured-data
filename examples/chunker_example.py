"""Example usage of MarkdownChunker."""

from src.chunkers import MarkdownChunker

# Esempio di documento Markdown da OCR
sample_md = """# Report Finanziario 2024

## Introduzione

Questo report analizza le performance dell'azienda nel 2024.

## Vendite Trimestrali

Di seguito i dati di vendita per prodotto.

| Prodotto | Q1 | Q2 | Q3 | Q4 |
|----------|----|----|----|----|
| Prodotto A | 100 | 120 | 150 | 180 |
| Prodotto B | 80 | 90 | 110 | 130 |

## Analisi dei Costi

I costi operativi sono aumentati del 15%.

### Dettaglio Costi per Reparto

| Reparto | Budget | Speso | Variazione |
|---------|--------|-------|------------|
| Marketing | 100k | 95k | -5% |
| IT | 200k | 220k | +10% |
| HR | 50k | 48k | -4% |

## Conclusioni

L'anno 2024 ha mostrato risultati positivi.
"""


def main():
    # Inizializza il chunker
    chunker = MarkdownChunker(
        headers_to_split_on=["#", "##", "###"],
        chunk_size=2000,
        chunk_overlap=200,
        max_table_rows=100,
        preserve_tables=True
    )

    # Processa il documento
    chunks = chunker.chunk(sample_md)

    print(f"Generati {len(chunks)} chunk:\n")

    for i, chunk in enumerate(chunks, 1):
        print(f"--- Chunk {i} ---")
        print(f"Tipo: {chunk.chunk_type}")
        print(f"Headers: {chunk.headers}")
        print(f"Lunghezza: {len(chunk.content)} caratteri")
        print(f"Content preview:\n{chunk.content[:300]}...\n")


if __name__ == "__main__":
    main()
