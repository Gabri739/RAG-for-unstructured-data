"""Step 6 - Valutazione quantitativa del Retrieval su Qdrant (con estrazione della frase/riga e domande negative).

Legge il file delle domande di test (`tests/eval_questions_dnf2023.json`), interroga Qdrant
con i primi `top_k=20` risultati (e soglia di rilevanza `DEFAULT_SCORE_THRESHOLD = 0.55`) e:
  1. Per le domande con risposta nel documento, verifica in quale posizione cade il chunk corretto:
     `PRIMO (1°)`, `TRA I PRIMI 5`, `TRA I PRIMI 20`, `ASSENTE (>20)`.
  2. Per le domande fuori perimetro (`expected_type == "none"` / `expected_pages == []`, es. criptovalute),
     verifica che nessun chunk superi la soglia di similarità (`score_threshold = 0.55`), mostrando
     anche qual era lo score massimo grezzo sotto soglia.
  3. Estrae dal chunk trovato la SINGOLA FRASE (se `text`) o la SINGOLA RIGA DELLA TABELLA
     (se `table`, con l'header delle colonne) più pertinente alla domanda tramite BGE-M3.

Uso:
    uv run python scripts/06_eval_retrieval.py [file_domande.json] [doc_id_filtro] [score_threshold]
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.vectorstore import (  # noqa: E402
    DEFAULT_COLLECTION_NAME,
    DEFAULT_QDRANT_PATH,
    QdrantVectorStore,
    SearchResult,
    get_embedder,
)

DEFAULT_SCORE_THRESHOLD: float = 0.55


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
        return "ASSENTE (>20)"
    if rank == 1:
        return "PRIMO (1°)"
    if rank <= 5:
        return f"TRA I PRIMI 5 (pos. {rank})"
    return f"TRA I PRIMI 20 (pos. {rank})"


def evaluate_questions(
    store: QdrantVectorStore,
    questions: list[dict[str, Any]],
    *,
    embedder: Any = None,
    doc_id: str | None = None,
    top_k: int = 20,
    score_threshold: float = DEFAULT_SCORE_THRESHOLD,
) -> dict[str, Any]:
    """Esegue la valutazione su tutte le domande (positive e negative/fuori perimetro)."""
    details: list[dict[str, Any]] = []
    counts = {
        "first": 0,
        "top5": 0,
        "top20": 0,
        "missing": 0,
        "negative_total": 0,
        "negative_correct": 0,
    }
    reciprocal_ranks: list[float] = []
    positive_count = 0

    for q_item in questions:
        q_id = str(q_item.get("id", ""))
        question = str(q_item["question"])
        exp_type = q_item.get("expected_type")
        exp_pages = [int(p) for p in q_item.get("expected_pages", [])]
        exp_keywords = [str(k) for k in q_item.get("expected_keywords", [])]
        exp_answer = str(q_item.get("expected_answer", ""))
        is_unanswerable = exp_type == "none" or (not exp_pages and exp_type == "none")

        q_vec = embedder.embed_query(question)
        # Cerca sia con soglia sia senza soglia (per mostrare lo score grezzo più alto anche quando viene scartato)
        raw_hits = store.search(query_vector=q_vec, top_k=top_k, doc_id=doc_id)
        hits = [h for h in raw_hits if h.score >= score_threshold]

        if is_unanswerable:
            counts["negative_total"] += 1
            rejected = len(hits) == 0
            if rejected:
                counts["negative_correct"] += 1
                bucket_label = f"ASSENTE CORRETTO (max score={raw_hits[0].score:.4f} < {score_threshold})"
            else:
                bucket_label = f"FALSO POSITIVO (score={hits[0].score:.4f} >= {score_threshold})"

            raw_top1_snippet = (
                _extract_best_snippet(raw_hits[0], question, embedder, query_vec=q_vec)
                if raw_hits
                else None
            )
            details.append(
                {
                    "id": q_id,
                    "question": question,
                    "expected_type": exp_type,
                    "expected_pages": exp_pages,
                    "expected_answer": exp_answer,
                    "is_unanswerable": True,
                    "rank": None,
                    "status": bucket_label,
                    "matched_chunk_id": None,
                    "matched_score": None,
                    "matched_snippet": None,
                    "top1_chunk_id": raw_hits[0].chunk_id if raw_hits else None,
                    "top1_type": raw_hits[0].chunk_type if raw_hits else None,
                    "top1_pages": raw_hits[0].pages if raw_hits else [],
                    "top1_score": raw_hits[0].score if raw_hits else None,
                    "top1_snippet": raw_top1_snippet,
                }
            )
            continue

        positive_count += 1
        found_rank: int | None = None
        matched_hit: SearchResult | None = None
        for idx, h in enumerate(hits, start=1):
            if _is_matching_chunk(h, exp_pages, exp_type, exp_keywords):
                found_rank = idx
                matched_hit = h
                break

        bucket_label = classify_rank(found_rank)
        if found_rank == 1:
            counts["first"] += 1
            counts["top5"] += 1
            counts["top20"] += 1
            reciprocal_ranks.append(1.0)
        elif found_rank is not None and found_rank <= 5:
            counts["top5"] += 1
            counts["top20"] += 1
            reciprocal_ranks.append(1.0 / found_rank)
        elif found_rank is not None:
            counts["top20"] += 1
            reciprocal_ranks.append(1.0 / found_rank)
        else:
            counts["missing"] += 1
            reciprocal_ranks.append(0.0)

        best_overall = hits[0] if hits else (raw_hits[0] if raw_hits else None)
        top1_snippet = (
            _extract_best_snippet(best_overall, question, embedder, query_vec=q_vec)
            if best_overall is not None
            else None
        )
        matched_snippet = (
            _extract_best_snippet(matched_hit, question, embedder, query_vec=q_vec)
            if matched_hit is not None
            else None
        )

        details.append(
            {
                "id": q_id,
                "question": question,
                "expected_type": exp_type,
                "expected_pages": exp_pages,
                "expected_answer": exp_answer,
                "is_unanswerable": False,
                "rank": found_rank,
                "status": bucket_label,
                "matched_chunk_id": matched_hit.chunk_id if matched_hit else None,
                "matched_score": matched_hit.score if matched_hit else None,
                "matched_snippet": matched_snippet,
                "top1_chunk_id": best_overall.chunk_id if best_overall else None,
                "top1_type": best_overall.chunk_type if best_overall else None,
                "top1_pages": best_overall.pages if best_overall else [],
                "top1_score": best_overall.score if best_overall else None,
                "top1_snippet": top1_snippet,
            }
        )

    pos_div = max(positive_count, 1)
    summary = {
        "total_questions": len(questions),
        "positive_questions": positive_count,
        "negative_questions": counts["negative_total"],
        "negative_correct": counts["negative_correct"],
        "score_threshold": score_threshold,
        "first_count": counts["first"],
        "top5_count": counts["top5"],
        "top20_count": counts["top20"],
        "missing_count": counts["missing"],
        "hit_at_1": round(counts["first"] / pos_div, 4),
        "hit_at_5": round(counts["top5"] / pos_div, 4),
        "hit_at_20": round(counts["top20"] / pos_div, 4),
        "mrr": round(sum(reciprocal_ranks) / pos_div, 4),
        "results": details,
    }
    return summary


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    questions_path = (
        Path(sys.argv[1])
        if len(sys.argv) > 1
        else ROOT / "tests/eval_questions_dnf2023.json"
    )
    doc_id_filter = sys.argv[2] if len(sys.argv) > 2 else None
    threshold = float(sys.argv[3]) if len(sys.argv) > 3 else DEFAULT_SCORE_THRESHOLD

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
    print(
        f"Punti indicizzati in Qdrant: {n_points} | Domande di test: {len(questions)} | "
        f"Soglia rilevanza (score_threshold): {threshold}"
    )
    print("=" * 105)

    report = evaluate_questions(
        store,
        questions,
        embedder=embedder,
        doc_id=doc_id_filter,
        top_k=20,
        score_threshold=threshold,
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
                f"  [Scartato < {threshold}]: {item['top1_chunk_id']} (score={item['top1_score']}): {top1_ind}"
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
        f"Top-20: {report['top20_count']}/{report['positive_questions']} ({report['hit_at_20']*100:.1f}%) | "
        f"MRR: {report['mrr']:.4f}\n"
        f"RIEPILOGO DOMANDE FUORI PERIMETRO ({report['negative_questions']}): "
        f"Scartate correttamente sotto soglia ({threshold}): {report['negative_correct']}/{report['negative_questions']}"
    )

    out_report = ROOT / "data/qdrant/eval_report.json"
    out_report.parent.mkdir(parents=True, exist_ok=True)
    out_report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Report salvato in: {out_report}")
    store.close()


if __name__ == "__main__":
    main()
