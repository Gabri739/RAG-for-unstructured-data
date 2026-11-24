"""Step 8 - Valutazione delle risposte dell'LLM sul file di domande di test.

Controlli automatici, senza un LLM giudice:
  - domande fuori perimetro: l'LLM dichiara che la risposta manca;
  - domande con risposta: l'LLM non rifiuta, cita una pagina attesa e riporta le parole chiave attese;
  - numeri della risposta che non compaiono negli estratti né nella domanda (da controllare).

Le parole chiave servivano a riconoscere il chunk giusto: qui sono solo un INDIZIO di correttezza.
Il giudizio definitivo richiede di leggere le risposte salvate nel report.

Uso:
    uv run python scripts/08_eval_answers.py domande.json [--doc-id ID] [--model qwen3.5:4b]
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.rag import (  # noqa: E402
    DEFAULT_CONTEXT_CHUNKS,
    DEFAULT_LLM_MODEL,
    Answer,
    OllamaError,
    OllamaGenerator,
    retrieve,
)
from src.vectorstore import (  # noqa: E402
    DEFAULT_QDRANT_PATH,
    QdrantVectorStore,
    get_embedder,
    get_reranker,
)

_NUMBER_RE = re.compile(r"\d+(?:[.,]\d+)*")
_PAGE_RE = re.compile(r"\[[^\]]*pag[^\]]*\]", re.IGNORECASE)


def unsupported_numbers(question: str, answer: Answer) -> list[str]:
    """Numeri della risposta (escluse le citazioni di pagina) assenti da estratti e domanda."""
    known = set(_NUMBER_RE.findall(question))
    for h in answer.context:
        known.update(_NUMBER_RE.findall(h.content))
    found = _NUMBER_RE.findall(_PAGE_RE.sub(" ", answer.text))
    return [n for n in dict.fromkeys(found) if n not in known]


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description="Valutazione delle risposte dell'LLM")
    parser.add_argument("questions", type=Path, help="file JSON delle domande di test")
    parser.add_argument("--doc-id", default=None, help="limita la ricerca a un documento")
    parser.add_argument("--model", default=DEFAULT_LLM_MODEL)
    parser.add_argument("--context-chunks", type=int, default=DEFAULT_CONTEXT_CHUNKS)
    parser.add_argument("--out", type=Path, default=ROOT / "data/qdrant/eval_answers.json")
    args = parser.parse_args()
    if not args.questions.exists():
        raise FileNotFoundError(f"File domande non trovato: {args.questions}")
    questions = json.loads(args.questions.read_text(encoding="utf-8"))

    embedder, reranker = get_embedder(), get_reranker()
    generator = OllamaGenerator(args.model)
    store = QdrantVectorStore(path=ROOT / DEFAULT_QDRANT_PATH)
    results: list[dict[str, Any]] = []
    try:
        for q in questions:
            question = str(q["question"])
            t0 = time.perf_counter()
            hits = retrieve(
                question,
                store,
                embedder=embedder,
                reranker=reranker,
                doc_id=args.doc_id,
                top_k=args.context_chunks,
            )
            t1 = time.perf_counter()
            answer = generator.answer(question, hits)
            t2 = time.perf_counter()

            pages = [int(p) for p in q.get("expected_pages", [])]
            keywords = [str(k) for k in q.get("expected_keywords", [])]
            context_pages = {p for h in hits for p in h.pages}
            low = answer.text.lower()
            results.append(
                {
                    "id": str(q.get("id", "")),
                    "question": question,
                    "answerable": q.get("expected_type") != "none",
                    "expected_answer": q.get("expected_answer", ""),
                    "expected_pages": pages,
                    "answer": answer.text,
                    "refused": answer.refused,
                    "cited_pages": answer.cited_pages,
                    "cites_expected_page": bool(set(answer.cited_pages) & set(pages)),
                    "invalid_citations": sorted(set(answer.cited_pages) - context_pages),
                    "keywords_missing": [k for k in keywords if k.lower() not in low],
                    "unsupported_numbers": []
                    if answer.refused
                    else unsupported_numbers(question, answer),
                    "truncated": answer.truncated,
                    "prompt_tokens": answer.prompt_tokens,
                    "answer_tokens": answer.answer_tokens,
                    "retrieval_s": round(t1 - t0, 2),
                    "generation_s": round(t2 - t1, 2),
                    "context_chunk_ids": [h.chunk_id for h in hits],
                }
            )
            outcome = "RIFIUTO" if answer.refused else "RISPOSTA"
            print(f"[{results[-1]['id']}] {outcome} ({t2 - t1:.1f}s): {answer.text}", flush=True)
    except OllamaError as exc:
        sys.exit(f"Interrotto dopo {len(results)} domande: {exc}")
    finally:
        store.close()

    pos = [r for r in results if r["answerable"]]
    neg = [r for r in results if not r["answerable"]]
    answered = [r for r in pos if not r["refused"]]
    report = {
        "model": args.model,
        "context_chunks": args.context_chunks,
        "answerable_questions": len(pos),
        "answered": len(answered),
        "wrongly_refused": [r["id"] for r in pos if r["refused"]],
        "cites_expected_page": sum(r["cites_expected_page"] for r in answered),
        "all_keywords": sum(not r["keywords_missing"] for r in answered),
        "unanswerable_questions": len(neg),
        "correctly_refused": sum(r["refused"] for r in neg),
        "wrongly_answered": [r["id"] for r in neg if not r["refused"]],
        "with_unsupported_numbers": [r["id"] for r in results if r["unsupported_numbers"]],
        "with_invalid_citations": [r["id"] for r in results if r["invalid_citations"]],
        "truncated": [r["id"] for r in results if r["truncated"]],
        "median_generation_s": round(statistics.median(r["generation_s"] for r in results), 2)
        if results
        else None,
        "results": results,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=" * 100)
    print(
        f"CON RISPOSTA ({len(pos)}): risposte date {len(answered)} | "
        f"pagina attesa citata {report['cites_expected_page']} | "
        f"tutte le parole chiave {report['all_keywords']} | "
        f"rifiutate per errore {report['wrongly_refused']}\n"
        f"FUORI PERIMETRO ({len(neg)}): rifiutate {report['correctly_refused']} | "
        f"risposte per errore {report['wrongly_answered']}\n"
        f"DA CONTROLLARE: numeri non negli estratti {report['with_unsupported_numbers']} | "
        f"citazioni non valide {report['with_invalid_citations']} | troncate {report['truncated']}\n"
        f"Tempo mediano LLM: {report['median_generation_s']}s | Report: {args.out}"
    )


if __name__ == "__main__":
    main()
