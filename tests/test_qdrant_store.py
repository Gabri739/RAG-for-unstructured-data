#!/usr/bin/env python3
"""Test dello store vettoriale Qdrant (salvataggio dense + sparse, ricerca densa, candidati ibridi, filtri)."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.chunkers import MarkdownChunker  # noqa: E402
from src.chunkers.reference_codes import find_mentioned_codes  # noqa: E402
from src.vectorstore import (  # noqa: E402
    BGE_M3_DIM,
    DENSE_VECTOR_NAME,
    SPARSE_VECTOR_NAME,
    QdrantVectorStore,
    Reranker,
    SearchResult,
    SparseEmbedding,
)


def _fake_dense(seed: int, dim: int = BGE_M3_DIM) -> list[float]:
    vec = [0.0] * dim
    vec[seed % dim] = 0.8
    vec[(seed + 1) % dim] = 0.6
    return vec


def _fake_sparse(seed: int) -> SparseEmbedding:
    return SparseEmbedding(indices=[100 + seed, 200 + seed], values=[0.45, 0.82])


def test_qdrant_dual_vectors_and_dense_search():
    """Verifica:
    1) Ogni punto salvato su Qdrant contiene SIA il vettore `dense` (1024-d) SIA il vettore `sparse`.
    2) La ricerca (`search()`) interroga il vettore `dense` (ricerca classica).
    3) La re-indicizzazione cancella i vecchi punti dello stesso `doc_id` prima di inserire i nuovi.
    """
    md = """# Bilancio Sostenibilità

## [GRI 204-1] Catena di fornitura
Nel 2023 il Gruppo si è avvalso di 694 fornitori.

Spesa fornitori per area geografica

| Area | 2023 (Euro) |
|---|---|
| Nord | 25.000.000 |
| Centro | 14.000.000 |
| Sud | 600.000 |

## Matrice incompleta

| Tema | Icona | Descrizione |
|---|---|---|
| A | | Testo A |
| B | | Testo B |
| C | | Testo C |
"""
    chunker = MarkdownChunker()
    chunks = chunker.chunk(md, doc_id="dnf_test", source_file="test.pdf")
    for idx, c in enumerate(chunks):
        c.metadata["pages"] = [43] if idx < 2 else [44]
        c.metadata["page_start"] = c.metadata["pages"][0]
        c.metadata["page_end"] = c.metadata["pages"][-1]
        c.metadata["page_range"] = str(c.metadata["page_start"])

    store = QdrantVectorStore(collection_name="test_chunks", in_memory=True)
    d_vecs = [_fake_dense(i) for i in range(len(chunks))]
    s_vecs = [_fake_sparse(i) for i in range(len(chunks))]

    rep1 = store.index_chunks(
        chunks,
        dense_vectors=d_vecs,
        sparse_vectors=s_vecs,
        replace_existing_doc=True,
    )
    assert rep1.indexed_points == 3
    assert rep1.stored_sparse_vectors is True

    # Verifica che il punto su Qdrant abbia effettivamente sia "dense" sia "sparse"
    pts = store.client.retrieve(
        collection_name="test_chunks",
        ids=[chunks[0].metadata["point_id"]],
        with_vectors=True,
    )
    assert len(pts) == 1
    assert DENSE_VECTOR_NAME in pts[0].vector
    assert SPARSE_VECTOR_NAME in pts[0].vector
    assert list(pts[0].vector[SPARSE_VECTOR_NAME].indices) == [100, 200]

    # Ricerca classica sul vettore denso
    hits = store.search(query_vector=_fake_dense(1), top_k=2, doc_id="dnf_test")
    assert len(hits) == 2
    assert hits[0].chunk_id == "dnf_test__chunk_0001"
    assert hits[0].chunk_type == "table"

    # Re-indicizzazione con 2 chunk: deve eliminare prima i 3 precedenti
    rep2 = store.index_chunks(
        chunks[:2],
        dense_vectors=d_vecs[:2],
        sparse_vectors=s_vecs[:2],
        replace_existing_doc=True,
    )
    assert rep2.deleted_existing_points == 3
    assert rep2.indexed_points == 2
    assert store.count_points(doc_id="dnf_test") == 2

    store.close()
    print("[OK] test_qdrant_dual_vectors_and_dense_search")


def test_find_mentioned_codes():
    """I codici citati in una domanda vengono normalizzati come nei metadati dei chunk."""
    assert find_mentioned_codes("Cosa dichiara il Gruppo in relazione al GRI 205-1?") == ["GRI 205-1"]
    assert find_mentioned_codes("Vedi GRI 205-1, 205-2 e GRI 2.7.") == ["GRI 205-1", "GRI 205-2", "GRI 2-7"]
    assert find_mentioned_codes("gri 2-22 e ancora GRI 2-22") == ["GRI 2-22"]
    # Numeri che non sono codici GRI
    assert find_mentioned_codes("Nel periodo 2021-2023 la norma ISO 14001 e il 205-1") == []
    assert find_mentioned_codes("GRI 2-2023") == []
    print("[OK] test_find_mentioned_codes")


def test_hybrid_candidates():
    """Verifica unione alternata dense/sparse, chunk con codice citato, limiti e filtro documento."""
    def chunk(doc_id: str, i: int, codes: list[str]) -> dict:
        return {
            "content": f"testo {i}",
            "headers": [],
            "chunk_type": "text",
            "metadata": {"doc_id": doc_id, "chunk_id": f"{doc_id}__{i}", "reference_codes": codes},
        }

    def dense(x: float, y: float, z: float = 0.0) -> list[float]:
        return [x, y, z] + [0.0] * (BGE_M3_DIM - 3)

    # Dense: 0 > 1 > 2 > 3 > 4 > 5. Sparse: 2 > 4 (gli altri non hanno il termine 7)
    codes = [[], [], [], [], [], ["GRI 205-1", "GRI 205-2"]]
    d_vecs = [dense(1, 0.1 * i) for i in range(5)] + [dense(0, 0, 1)]
    s_vecs = [SparseEmbedding([100 + i], [1.0]) for i in range(6)]
    s_vecs[2] = SparseEmbedding([7], [0.9])
    s_vecs[4] = SparseEmbedding([7], [0.5])

    store = QdrantVectorStore(collection_name="test_hybrid", in_memory=True)
    store.index_chunks(
        [chunk("a", i, codes[i]) for i in range(6)], dense_vectors=d_vecs, sparse_vectors=s_vecs
    )
    store.index_chunks(
        [chunk("b", 0, ["GRI 205-1"])],
        dense_vectors=[dense(1, 0)],
        sparse_vectors=[SparseEmbedding([7], [1.0])],
    )
    vectors = (dense(1, 0), SparseEmbedding([7], [1.0]))

    def ids(query: str, **kw) -> list[str]:
        return [h.chunk_id for h in store.hybrid_candidates(query, query_vectors=vectors, doc_id="a", **kw)]

    # Alternanza senza doppioni: dense 0, sparse 2, dense 1, sparse 4
    assert ids("domanda", limit=4) == ["a__0", "a__2", "a__1", "a__4"]
    # Quando la lista sparse finisce si continua con la dense
    assert ids("domanda", limit=5) == ["a__0", "a__2", "a__1", "a__4", "a__3"]
    # Il chunk con il codice citato si AGGIUNGE al limite, e solo per il documento richiesto
    assert ids("Cosa dice il GRI 205-1?", limit=2) == ["a__0", "a__2", "a__5"]
    assert ids("Cosa dice il GRI 205-1?", limit=2, max_code_chunks=0) == ["a__0", "a__2"]
    assert ids("Cosa dice il GRI 999-9?", limit=2) == ["a__0", "a__2"]

    by_id = {
        h.chunk_id: h.sources
        for h in store.hybrid_candidates("GRI 205-1", query_vectors=vectors, doc_id="a", limit=6)
    }
    assert by_id["a__0"] == ["dense"]
    assert sorted(by_id["a__2"]) == ["dense", "sparse"]
    assert "code" in by_id["a__5"]
    # Senza filtro documento si vedono anche gli altri
    everywhere = store.hybrid_candidates("GRI 205-1", query_vectors=vectors, limit=2)
    assert {"a__5", "b__0"} <= {h.chunk_id for h in everywhere}

    store.close()
    print("[OK] test_hybrid_candidates")


def test_reranker_order():
    """Il reranker ordina per punteggio, non modifica i candidati e gestisce la lista vuota."""
    class FakeModel:
        def predict(self, pairs, **_):
            return [{"basso": -2.0, "alto": 5.0, "medio": 1.0}[text] for _, text in pairs]

    reranker = Reranker.__new__(Reranker)
    reranker.model, reranker.batch_size = FakeModel(), 8
    hits = [
        SearchResult(0.5, t, t, "d", "text", t, [], "", None, [], [], False, []) for t in ("basso", "alto", "medio")
    ]
    assert [h.content for h in reranker.rerank("q", hits)] == ["alto", "medio", "basso"]
    assert [h.score for h in reranker.rerank("q", hits, top_k=2)] == [5.0, 1.0]
    assert [h.score for h in hits] == [0.5, 0.5, 0.5]
    assert reranker.rerank("q", []) == []
    print("[OK] test_reranker_order")


def test_qdrant_with_real_chunks_json():
    """Verifica il caricamento diretto di `chunks.json` generato su Sara Assicurazioni."""
    chunks_path = Path("data/processed/docling/dnf2023_test_p43_44/chunks.json")
    if not chunks_path.exists():
        print("[SKIP] test_qdrant_with_real_chunks_json")
        return

    chunks_data = json.loads(chunks_path.read_text(encoding="utf-8"))
    store = QdrantVectorStore(collection_name="test_sara", in_memory=True)
    d_vecs = [_fake_dense(i * 10) for i in range(len(chunks_data))]
    s_vecs = [_fake_sparse(i * 10) for i in range(len(chunks_data))]

    rep = store.index_chunks(chunks_data, dense_vectors=d_vecs, sparse_vectors=s_vecs)
    assert rep.indexed_points == len(chunks_data)

    hits_p44 = store.search(
        query_vector=d_vecs[-1],
        top_k=5,
        chunk_type="table",
        page_min=44,
        page_max=44,
    )
    assert len(hits_p44) == 2
    for h in hits_p44:
        assert h.page_range == "43-44"
        assert h.image_paths == ["table_images/t1_p43.png", "table_images/t1_p44.png"]

    store.close()
    print("[OK] test_qdrant_with_real_chunks_json")


if __name__ == "__main__":
    test_qdrant_dual_vectors_and_dense_search()
    test_find_mentioned_codes()
    test_hybrid_candidates()
    test_reranker_order()
    test_qdrant_with_real_chunks_json()
    print("\nTUTTI I TEST QDRANT (DENSE + SPARSE) SUPERATI!")
