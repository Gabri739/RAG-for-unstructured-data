"""Step 6 - Valutazione quantitativa del Retrieval: candidati ibridi + reranker.

Per ogni domanda del file indicato:
  1. Raccoglie i candidati (`QdrantVectorStore.hybrid_candidates`): 15 chunk dall'unione di
     dense e sparse, più i chunk con i codici citati nella domanda (es. GRI 205-1).
  2. Li riordina con il reranker (`Qwen/Qwen3-Reranker-0.6B`), senza soglie di punteggio.
  3. Per le domande con risposta nel documento, verifica in quale posizione cade il chunk corretto:
     `PRIMO (1°)`, `TRA I PRIMI 5`, `TRA I CANDIDATI`, `ASSENTE`.
  4. Per le domande fuori perimetro (`expected_type == "none"`) riporta solo il punteggio più alto:
     il retrieval restituisce sempre qualcosa, e riconoscere che la risposta manca spetta all'LLM.
  5. Estrae dal chunk trovato la SINGOLA FRASE (se `text`) o la SINGOLA RIGA DELLA TABELLA
     (se `table`, con l'header delle colonne) più pertinente alla domanda tramite BGE-M3.

Uso:
    uv run python scripts/06_eval_retrieval.py domande.json [--doc-id ID] [--candidates 15]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.vectorstore import (  # noqa: E402
    DEFAULT_CANDIDATES,
    DEFAULT_COLLECTION_NAME,
    DEFAULT_QDRANT_PATH,
    QdrantVectorStore,
    SearchResult,
    get_embedder,
    get_reranker,
)


def _extract_best_snippet(
    hit: SearchResult,
    question: str,
    embedder: Any,
    query_vec: list[float] | None = None,
) -> str:
    """Estrae dal chunk `hit` la frase (se `text`) o la riga di tabella (se `table`)
    più rilevante per la `question` usando la similarità coseno di BGE-M3.
    """
    q_vec = np.asarray(
        query_vec if query_vec is not None else embedder.embed_query(question),
        dtype=np.float32,
    )

    if hit.chunk_type == "table":
        lines = [ln.strip() for ln in hit.content.splitlines() if ln.strip()]
        table_lines = [ln for ln in lines if ln.startswith("|")]
        if len(table_lines) >= 3:
            header_line = table_lines[0]
            headers = [c.strip() for c in header_line.strip("|").split("|")]
            data_lines = [
                ln for ln in table_lines[1:]
                if not re.match(r"^\|[\s:\-\|]+\|$", ln)
            ]
            if data_lines:
                row_representations: list[str] = []
                for r_ln in data_lines:
                    cells = [c.strip() for c in r_ln.strip("|").split("|")]
                    pairs = [
                        f"{headers[i]}: {cells[i]}"
                        for i in range(min(len(headers), len(cells)))
                        if cells[i]
                    ]
                    row_representations.append(" | ".join(pairs) if pairs else r_ln)

                row_vecs = np.asarray(embedder.embed_texts(row_representations), dtype=np.float32)
                scores = row_vecs @ q_vec
                ranked_idx = np.argsort(-scores)
                best_idx = int(ranked_idx[0])
                selected_rows = [data_lines[best_idx]]
                if len(ranked_idx) > 1 and scores[ranked_idx[1]] >= scores[best_idx] - 0.04:
                    second_idx = int(ranked_idx[1])
                    selected_rows = [data_lines[i] for i in sorted([best_idx, second_idx])]

                return f"{header_line}\n" + "\n".join(selected_rows)

    body_lines = [
        ln.strip()
        for ln in hit.content.splitlines()
        if ln.strip() and not ln.strip().startswith("#")
    ]
    body_text = " ".join(body_lines)
    # Evita di spezzare su abbreviazioni italiane comuni (D.Lgs., art., n., pag., es., ecc.)
    protected = re.sub(
        r"\b(d\.lgs|art|artt|n|nr|pag|pagg|es|ecc|c\.d|s\.p\.a)\.\s+",
        lambda m: m.group(0).replace(". ", "§ "),
        body_text,
        flags=re.IGNORECASE,
    )
    raw_sentences = re.split(r"(?<=[\.\?;])\s+|\s*●\s*", protected)
    sentences = [
        s.replace("§ ", ". ").strip()
        for s in raw_sentences
        if len(s.strip()) >= 20
    ]
    if not sentences:
        return body_text[:300]

    if len(sentences) == 1:
        return sentences[0]

    sent_vecs = np.asarray(embedder.embed_texts(sentences), dtype=np.float32)
    scores = sent_vecs @ q_vec
    best_idx = int(np.argmax(scores))
    return sentences[best_idx]


def _is_matching_chunk(
    hit: SearchResult,
    expected_pages: list[int],
    expected_type: str | None,
    expected_keywords: list[str],
) -> bool:
    """Verifica se un `SearchResult` è il chunk corretto per la domanda di test."""
    if expected_type and expected_type != "any":
        if hit.chunk_type != expected_type:
            return False

    if expected_pages:
        if not set(hit.pages).intersection(expected_pages):
            return False

    if expected_keywords:
        content_lower = hit.content.lower()
        if not all(kw.lower() in content_lower for kw in expected_keywords):
            return False

    return True


def classify_rank(rank: int | None) -> str:
    """Restituisce l'etichetta leggibile della fascia di posizionamento."""
    if rank is None:
        return "ASSENTE (non tra i candidati)"
    if rank == 1:
        return "PRIMO (1°)"
    if rank <= 5:
        return f"TRA I PRIMI 5 (pos. {rank})"
    return f"TRA I CANDIDATI (pos. {rank})"


def evaluate_questions(
    store: QdrantVectorStore,
    questions: list[dict[str, Any]],
    *,
    embedder: Any = None,
    reranker: Any = None,
    doc_id: str | None = None,
    candidates: int = DEFAULT_CANDIDATES,
) -> dict[str, Any]:
    """Esegue la valutazione su tutte le domande (positive e negative/fuori perimetro)."""
    embedder = embedder or get_embedder()
    reranker = reranker or get_reranker()
    details: list[dict[str, Any]] = []
    ranks: list[int | None] = []
    negative_total = 0

    for q_item in questions:
        question = str(q_item["question"])
        exp_type = q_item.get("expected_type")
        exp_pages = [int(p) for p in q_item.get("expected_pages", [])]
        exp_keywords = [str(k) for k in q_item.get("expected_keywords", [])]
        is_unanswerable = exp_type == "none"

        out = embedder.embed_texts_both([question], batch_size=1)
        q_vec = out.dense_vecs[0]
        pool = store.hybrid_candidates(
            question,
            query_vectors=(q_vec, out.sparse_vecs[0]),
            limit=candidates,
            doc_id=doc_id,
        )
        hits = reranker.rerank(question, pool)
        top1 = hits[0] if hits else None

        rank: int | None = None
        if is_unanswerable:
            negative_total += 1
            status = "FUORI PERIMETRO"
        else:
            rank = next(
                (
                    i
                    for i, h in enumerate(hits, start=1)
                    if _is_matching_chunk(h, exp_pages, exp_type, exp_keywords)
                ),
                None,
            )
            ranks.append(rank)
            status = classify_rank(rank)
        matched = hits[rank - 1] if rank else None

        def snippet(hit: SearchResult | None) -> str | None:
            if hit is None:
                return None
            return _extract_best_snippet(hit, question, embedder, query_vec=q_vec)

        details.append(
            {
                "id": str(q_item.get("id", "")),
                "question": question,
                "expected_type": exp_type,
                "expected_pages": exp_pages,
                "expected_answer": str(q_item.get("expected_answer", "")),
                "is_unanswerable": is_unanswerable,
                "candidates": len(hits),
                "rank": rank,
                "status": status,
                "matched_chunk_id": matched.chunk_id if matched else None,
                "matched_score": matched.score if matched else None,
                "matched_sources": matched.sources if matched else [],
                "matched_snippet": snippet(matched),
                "top1_chunk_id": top1.chunk_id if top1 else None,
                "top1_type": top1.chunk_type if top1 else None,
                "top1_pages": top1.pages if top1 else [],
                "top1_score": top1.score if top1 else None,
                "top1_snippet": snippet(top1),
            }
        )

    pos_div = max(len(ranks), 1)
    first = sum(r == 1 for r in ranks)
    top5 = sum(r is not None and r <= 5 for r in ranks)
    found = sum(r is not None for r in ranks)
    return {
        "total_questions": len(questions),
        "positive_questions": len(ranks),
        "negative_questions": negative_total,
        "embedder": embedder.model_name,
        "reranker": reranker.model_name,
        "candidates_limit": candidates,
        "first_count": first,
        "top5_count": top5,
        "found_count": found,
        "missing_count": len(ranks) - found,
        "hit_at_1": round(first / pos_div, 4),
        "hit_at_5": round(top5 / pos_div, 4),
        "hit_in_candidates": round(found / pos_div, 4),
        "mrr": round(sum(1.0 / r for r in ranks if r) / pos_div, 4),
        "results": details,
    }


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description="Valutazione del retrieval")
    parser.add_argument("questions", type=Path, help="file JSON delle domande di test")
    parser.add_argument("--doc-id", default=None, help="limita la ricerca a un documento")
    parser.add_argument(
        "--candidates",
        type=int,
        default=DEFAULT_CANDIDATES,
        help="candidati da dense + sparse passati al reranker",
    )
    args = parser.parse_args()
    questions_path, doc_id_filter = args.questions, args.doc_id

    if not questions_path.exists():
        raise FileNotFoundError(f"File domande non trovato: {questions_path}")

    questions = json.loads(questions_path.read_text(encoding="utf-8"))
    store = QdrantVectorStore(
        collection_name=DEFAULT_COLLECTION_NAME,
        path=ROOT / DEFAULT_QDRANT_PATH,
    )

    n_points = store.count_points(doc_id=doc_id_filter)
    if n_points == 0:
        store.close()
        raise RuntimeError(
            "La collezione Qdrant è vuota! Esegui prima: python scripts/05_index_qdrant.py <cartella_chunk>"
        )

    embedder = get_embedder()
    reranker = get_reranker()
    print(
        f"Punti indicizzati in Qdrant: {n_points} | Domande di test: {len(questions)} | "
        f"Candidati: {args.candidates} + codici citati | Reranker: {reranker.model_name}"
    )
    print("=" * 105)

    report = evaluate_questions(
        store,
        questions,
        embedder=embedder,
        reranker=reranker,
        doc_id=doc_id_filter,
        candidates=args.candidates,
    )

    for item in report["results"]:
        print(
            f"[{item['id']}] ({item['expected_type']:5s} | pag. {item['expected_pages']}) "
            f"-> {item['status']} | score={item['matched_score']} | chunk={item['matched_chunk_id']}"
        )
        print(f"  Domanda         : {item['question']}")
        print(f"  Risposta attesa : {item['expected_answer']}")
        if item["matched_snippet"]:
            indented = "\n                    ".join(item["matched_snippet"].splitlines())
            print(f"  Frase/Riga estr.: {indented}")
        elif item.get("is_unanswerable") and item["top1_chunk_id"]:
            top1_ind = "\n                    ".join((item["top1_snippet"] or "").splitlines())
            print(
                f"  [1° posto]      : {item['top1_chunk_id']} (score={item['top1_score']}): {top1_ind}"
            )
        elif item["rank"] != 1 and item["top1_chunk_id"]:
            top1_ind = "\n                    ".join((item["top1_snippet"] or "").splitlines())
            print(
                f"  [1° posto era]  : {item['top1_chunk_id']} ({item['top1_type']}, pag. {item['top1_pages']}, "
                f"score={item['top1_score']}): {top1_ind}"
            )
        print("-" * 105)

    print(
        f"RIEPILOGO DOMANDE NEL DOCUMENTO ({report['positive_questions']}): "
        f"1° posto: {report['first_count']}/{report['positive_questions']} ({report['hit_at_1']*100:.1f}%) | "
        f"Top-5: {report['top5_count']}/{report['positive_questions']} ({report['hit_at_5']*100:.1f}%) | "
        f"Tra i candidati: {report['found_count']}/{report['positive_questions']} "
        f"({report['hit_in_candidates']*100:.1f}%) | "
        f"MRR: {report['mrr']:.4f}\n"
        f"DOMANDE FUORI PERIMETRO ({report['negative_questions']}): non scartate in questa fase, "
        f"spetta all'LLM riconoscere che la risposta manca"
    )

    out_report = ROOT / "data/qdrant/eval_report.json"
    out_report.parent.mkdir(parents=True, exist_ok=True)
    out_report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Report salvato in: {out_report}")
    store.close()


if __name__ == "__main__":
    main()
