"""Storage e ricerca vettoriale su Qdrant per i chunk estratti (testo e tabelle).

Architettura di storage su Qdrant:
  - Ogni punto salva ENTRAMBI i vettori generati da BAAI/bge-m3:
      * `"dense"`: vettore denso a 1024 dimensioni (Cosine similarity)
      * `"sparse"`: vettore sparso `SparseVector(indices=..., values=...)` (lexical weights)
  - La ricerca (`search()`) svolge la ricerca classica sul vettore `"dense"`, mentre i
    vettori `"sparse"` sono già archiviati su Qdrant e pronti per attivare in qualsiasi
    momento la ricerca ibrida senza dover re-indicizzare i documenti.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

from src.chunkers.markdown_chunker import Chunk, make_point_id

from .bge_m3 import BGE_M3_DIM, BGEM3Embedder, SparseEmbedding, get_embedder

DEFAULT_COLLECTION_NAME: str = "rag_chunks"
DEFAULT_QDRANT_PATH: Path = Path("data/qdrant")
DENSE_VECTOR_NAME: str = "dense"
SPARSE_VECTOR_NAME: str = "sparse"


@dataclass
class IndexingReport:
    """Riepilogo di un'operazione di indicizzazione su Qdrant."""

    collection_name: str
    doc_ids: list[str]
    deleted_existing_points: int
    indexed_points: int
    vector_dimension: int
    stored_sparse_vectors: bool = True


@dataclass
class SearchResult:
    """Singolo risultato restituito da una ricerca vettoriale su Qdrant."""

    score: float
    point_id: str
    chunk_id: str
    doc_id: str
    chunk_type: str
    content: str
    headers: list[str]
    section: str
    page_range: str | None
    pages: list[int]
    image_paths: list[str]
    possible_incomplete_table: bool
    empty_columns: list[str]
    metadata: dict[str, Any] = field(default_factory=dict)


def _normalize_chunk_dict(item: Chunk | dict[str, Any]) -> tuple[str, list[str], dict[str, Any]]:
    """Estrae `(content, headers, metadata)` sia da un oggetto `Chunk` sia da un dict di `chunks.json`."""
    if isinstance(item, Chunk):
        meta = item.to_vector_metadata(flat=False)
        meta.setdefault("chunk_type", item.chunk_type)
        return item.content, list(item.headers), meta

    content = str(item.get("content", ""))
    headers = list(item.get("headers", []))
    meta = dict(item.get("metadata", {}))
    meta.setdefault("chunk_type", item.get("chunk_type", "text"))
    return content, headers, meta


class QdrantVectorStore:
    """Gestisce la collezione Qdrant (con vettori `dense` + `sparse`) e la ricerca classica sul denso."""

    def __init__(
        self,
        collection_name: str = DEFAULT_COLLECTION_NAME,
        *,
        path: Path | str | None = DEFAULT_QDRANT_PATH,
        url: str | None = None,
        api_key: str | None = None,
        in_memory: bool = False,
        vector_size: int = BGE_M3_DIM,
    ) -> None:
        from qdrant_client import QdrantClient

        self.collection_name = collection_name
        self.vector_size = vector_size

        if in_memory:
            self.client = QdrantClient(location=":memory:")
        elif url:
            self.client = QdrantClient(url=url, api_key=api_key)
        else:
            resolved_path = Path(path) if path is not None else DEFAULT_QDRANT_PATH
            resolved_path.mkdir(parents=True, exist_ok=True)
            self.client = QdrantClient(path=str(resolved_path))

        self.ensure_collection()

    def ensure_collection(self) -> None:
        """Crea la collezione su Qdrant configurando sia il vettore `dense` (1024-d Cosine)
        sia il vettore `sparse` (`SparseVectorParams`).
        """
        from qdrant_client.models import Distance, SparseVectorParams, VectorParams

        if self.client.collection_exists(self.collection_name):
            info = self.client.get_collection(self.collection_name)
            vectors_cfg = info.config.params.vectors
            # Se la collezione esistente non usa ancora il named vector "dense", ricreala
            if not isinstance(vectors_cfg, dict) or DENSE_VECTOR_NAME not in vectors_cfg:
                self.client.delete_collection(self.collection_name)
            else:
                return

        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config={
                DENSE_VECTOR_NAME: VectorParams(
                    size=self.vector_size,
                    distance=Distance.COSINE,
                )
            },
            sparse_vectors_config={
                SPARSE_VECTOR_NAME: SparseVectorParams(),
            },
        )

    def count_points(self, doc_id: str | None = None) -> int:
        """Conta i punti presenti nella collezione (opzionalmente filtrati per `doc_id`)."""
        from qdrant_client.models import FieldCondition, Filter, MatchValue

        count_filter = (
            Filter(must=[FieldCondition(key="doc_id", match=MatchValue(value=doc_id))])
            if doc_id is not None
            else None
        )
        res = self.client.count(
            collection_name=self.collection_name,
            count_filter=count_filter,
            exact=True,
        )
        return int(res.count)

    def delete_document(self, doc_id: str) -> int:
        """Elimina tutti i chunk associati a `doc_id` prima di una re-indicizzazione."""
        from qdrant_client.models import FieldCondition, Filter, FilterSelector, MatchValue

        before = self.count_points(doc_id=doc_id)
        if before == 0:
            return 0

        doc_filter = Filter(
            must=[FieldCondition(key="doc_id", match=MatchValue(value=doc_id))]
        )
        self.client.delete(
            collection_name=self.collection_name,
            points_selector=FilterSelector(filter=doc_filter),
            wait=True,
        )
        return before

    def index_chunks(
        self,
        chunks: Sequence[Chunk | dict[str, Any]],
        *,
        embedder: BGEM3Embedder | None = None,
        dense_vectors: Sequence[Sequence[float]] | None = None,
        sparse_vectors: Sequence[SparseEmbedding | tuple[Sequence[int], Sequence[float]]] | None = None,
        replace_existing_doc: bool = True,
        batch_size: int = 16,
    ) -> IndexingReport:
        """Calcola gli embedding BGE-M3 (sia `dense` sia `sparse`) e salva i chunk su Qdrant.

        Args:
            chunks: Lista di oggetti `Chunk` o dizionari caricati da `chunks.json`.
            embedder: Istanza di `BGEM3Embedder` (se `None`, usa `get_embedder()`).
            dense_vectors: Vettori densi pre-calcolati opzionali (utile nei test).
            sparse_vectors: Vettori sparsi pre-calcolati opzionali (utile nei test).
            replace_existing_doc: Se `True` (default), cancella tutti i punti esistenti con lo
                stesso `doc_id` prima di inserire i nuovi chunk.
            batch_size: Dimensione del batch per embedding e upsert su Qdrant.
        """
        from qdrant_client.models import PointStruct, SparseVector

        if not chunks:
            return IndexingReport(
                collection_name=self.collection_name,
                doc_ids=[],
                deleted_existing_points=0,
                indexed_points=0,
                vector_dimension=self.vector_size,
                stored_sparse_vectors=True,
            )

        normalized_items = [_normalize_chunk_dict(c) for c in chunks]
        doc_ids = sorted({str(meta.get("doc_id") or "document") for _, _, meta in normalized_items})

        deleted_total = 0
        if replace_existing_doc:
            for d_id in doc_ids:
                deleted_total += self.delete_document(d_id)

        if dense_vectors is None or sparse_vectors is None:
            model = embedder or get_embedder()
            texts = [content for content, _, _ in normalized_items]
            bge_out = model.embed_texts_both(texts, batch_size=batch_size)
            computed_dense = bge_out.dense_vecs
            computed_sparse = bge_out.sparse_vecs
        else:
            computed_dense = [list(v) for v in dense_vectors]
            computed_sparse = [
                sp if isinstance(sp, SparseEmbedding) else SparseEmbedding(list(sp[0]), list(sp[1]))
                for sp in sparse_vectors
            ]

        points: list[PointStruct] = []
        for idx, ((content, headers, meta), d_vec, s_vec) in enumerate(
            zip(normalized_items, computed_dense, computed_sparse)
        ):
            doc_id = str(meta.get("doc_id") or "document")
            chunk_id = str(meta.get("chunk_id") or f"{doc_id}__chunk_{idx:04d}")
            point_id = str(meta.get("point_id") or make_point_id(chunk_id))

            payload: dict[str, Any] = {
                **meta,
                "doc_id": doc_id,
                "chunk_id": chunk_id,
                "point_id": point_id,
                "content": content,
                "headers": headers,
            }
            points.append(
                PointStruct(
                    id=point_id,
                    vector={
                        DENSE_VECTOR_NAME: d_vec,
                        SPARSE_VECTOR_NAME: SparseVector(
                            indices=s_vec.indices,
                            values=s_vec.values,
                        ),
                    },
                    payload=payload,
                )
            )

        for start in range(0, len(points), batch_size):
            self.client.upsert(
                collection_name=self.collection_name,
                points=points[start : start + batch_size],
                wait=True,
            )

        return IndexingReport(
            collection_name=self.collection_name,
            doc_ids=doc_ids,
            deleted_existing_points=deleted_total,
            indexed_points=len(points),
            vector_dimension=len(computed_dense[0]) if computed_dense else self.vector_size,
            stored_sparse_vectors=True,
        )

    def search(
        self,
        query: str | None = None,
        *,
        embedder: BGEM3Embedder | None = None,
        query_vector: Sequence[float] | None = None,
        top_k: int = 5,
        doc_id: str | None = None,
        chunk_type: str | None = None,
        page_min: int | None = None,
        page_max: int | None = None,
        exclude_incomplete_tables: bool = False,
        score_threshold: float | None = None,
    ) -> list[SearchResult]:
        """Svolge la ricerca vettoriale classica sul vettore `dense` in Qdrant."""
        from qdrant_client.models import FieldCondition, Filter, MatchValue, Range

        if query_vector is None:
            if not query:
                raise ValueError("Specificare `query` oppure `query_vector`.")
            model = embedder or get_embedder()
            q_vec = model.embed_query(query)
        else:
            q_vec = list(query_vector)

        must_conditions: list[Any] = []
        if doc_id is not None:
            must_conditions.append(FieldCondition(key="doc_id", match=MatchValue(value=doc_id)))
        if chunk_type is not None:
            must_conditions.append(FieldCondition(key="chunk_type", match=MatchValue(value=chunk_type)))
        if page_min is not None:
            must_conditions.append(FieldCondition(key="page_end", range=Range(gte=page_min)))
        if page_max is not None:
            must_conditions.append(FieldCondition(key="page_start", range=Range(lte=page_max)))
        if exclude_incomplete_tables:
            must_conditions.append(
                FieldCondition(key="possible_incomplete_table", match=MatchValue(value=False))
            )

        query_filter = Filter(must=must_conditions) if must_conditions else None

        response = self.client.query_points(
            collection_name=self.collection_name,
            query=q_vec,
            using=DENSE_VECTOR_NAME,
            query_filter=query_filter,
            limit=top_k,
            score_threshold=score_threshold,
            with_payload=True,
        )

        results: list[SearchResult] = []
        for sp in response.points:
            payload = dict(sp.payload or {})
            content = str(payload.pop("content", ""))
            headers = list(payload.pop("headers", []))
            results.append(
                SearchResult(
                    score=round(float(sp.score), 4),
                    point_id=str(sp.id),
                    chunk_id=str(payload.get("chunk_id", "")),
                    doc_id=str(payload.get("doc_id", "")),
                    chunk_type=str(payload.get("chunk_type", "text")),
                    content=content,
                    headers=headers,
                    section=str(payload.get("section", "")),
                    page_range=payload.get("page_range"),
                    pages=list(payload.get("pages", [])),
                    image_paths=list(payload.get("image_paths", [])),
                    possible_incomplete_table=bool(payload.get("possible_incomplete_table", False)),
                    empty_columns=list(payload.get("empty_columns", [])),
                    metadata=payload,
                )
            )
        return results

    def close(self) -> None:
        """Chiude la connessione al client Qdrant."""
        self.client.close()
