#!/usr/bin/env python3
"""Testa il chunker con il tuo file Markdown."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.chunkers import MarkdownChunker  # noqa: E402


def main():
    # Cerca file .md in data/raw/
    raw_dir = Path("data/raw")
    md_files = list(raw_dir.rglob("*.md"))

    if not md_files:
        print("Nessun file .md trovato in data/raw/")
        print("Metti il tuo file Markdown in data/raw/ o una sottocartella")
        return

    print(f"Trovati {len(md_files)} file Markdown:")
    for i, f in enumerate(md_files, 1):
        print(f"  {i}. {f}")

    # Usa il primo file
    file_path = md_files[0]
    print(f"\nProcesso: {file_path}")

    # Leggi contenuto
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    print(f"Dimensione: {len(content)} caratteri, {len(content.splitlines())} righe")

    # Configura chunker
    chunker = MarkdownChunker(
        headers_to_split_on=["#", "##", "###"],
        chunk_size=2000,
        chunk_overlap=200,
        preserve_tables=True
    )

    # Chunka
    print("\nChunking in corso...")
    chunks = chunker.chunk(content)

    print(f"\n{'='*60}")
    print(f"RISULTATI: {len(chunks)} chunk generati")
    print(f"{'='*60}")

    text_chunks = [c for c in chunks if c.chunk_type == "text"]
    table_chunks = [c for c in chunks if c.chunk_type == "table"]

    print(f"  - Chunk di testo: {len(text_chunks)}")
    print(f"  - Chunk di tabella: {len(table_chunks)}")

    # Dettagli per ogni chunk
    for i, chunk in enumerate(chunks, 1):
        print(f"\n--- Chunk {i}/{len(chunks)} ---")
        print(f"Tipo: {chunk.chunk_type}")
        print(f"Headers: {[h[:50] + '...' if len(h) > 50 else h for h in chunk.headers]}")
        print(f"Lunghezza: {len(chunk.content)} caratteri")
        print(f"Metadata: {chunk.metadata}")

        # Preview
        preview = chunk.content[:300].replace('\n', ' | ')
        print(f"Content: {preview}...")

    print(f"\n{'='*60}")
    print("Test completato!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
