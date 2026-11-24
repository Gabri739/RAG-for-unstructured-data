"""Step 7 - Risposta a una domanda: candidati ibridi, reranker e LLM locale (Ollama).

Uso:
    uv run python scripts/07_ask.py "domanda" [--doc-id ID] [--model qwen3.5:4b]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.rag import (  # noqa: E402
    DEFAULT_CONTEXT_CHUNKS,
    DEFAULT_LLM_MODEL,
    OllamaError,
    OllamaGenerator,
    retrieve,
)
from src.vectorstore import DEFAULT_QDRANT_PATH, QdrantVectorStore  # noqa: E402


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description="Risponde a una domanda sui documenti indicizzati")
    parser.add_argument("question")
    parser.add_argument("--doc-id", default=None, help="limita la ricerca a un documento")
    parser.add_argument("--model", default=DEFAULT_LLM_MODEL)
    parser.add_argument("--context-chunks", type=int, default=DEFAULT_CONTEXT_CHUNKS)
    args = parser.parse_args()

    store = QdrantVectorStore(path=ROOT / DEFAULT_QDRANT_PATH)
    try:
        hits = retrieve(args.question, store, doc_id=args.doc_id, top_k=args.context_chunks)
    finally:
        store.close()
    try:
        answer = OllamaGenerator(args.model).answer(args.question, hits)
    except OllamaError as exc:
        sys.exit(str(exc))

    print(f"\n{answer.text}\n")
    if answer.truncated:
        print("ATTENZIONE: risposta o contesto troncati.\n")
    print("Estratti passati all'LLM:")
    for i, h in enumerate(hits, start=1):
        print(f"  {i}. pag. {h.page_range} | {h.chunk_type} | {h.section or '-'} | {h.chunk_id}")


if __name__ == "__main__":
    main()
