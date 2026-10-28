#!/usr/bin/env python3
"""Test dello store vettoriale Qdrant (salvataggio dense + sparse, ricerca classica sul denso, filtri)."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.chunkers import MarkdownChunker  # noqa: E402
from src.vectorstore import (  # noqa: E402
    BGE_M3_DIM,
    DENSE_VECTOR_NAME,
    SPARSE_VECTOR_NAME,
    QdrantVectorStore,
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
    test_qdrant_with_real_chunks_json()
    print("\nTUTTI I TEST QDRANT (DENSE + SPARSE) SUPERATI!")
