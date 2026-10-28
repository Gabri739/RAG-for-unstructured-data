"""Step 5 - Embedding con BAAI/bge-m3 e salvataggio su Qdrant.

Carica `chunks.json` prodotto dallo Step 4 (`scripts/04_chunk_document.py`),
calcola i vettori densi (1024-d, normalizzati L2) con `BAAI/bge-m3` e li indicizza
su Qdrant (di default su disco locale in `data/qdrant`, collezione `rag_chunks`).

Prima dell'inserimento elimina automaticamente eventuali chunk preesistenti con lo
stesso `doc_id`, evitando duplicati o chunk orfani dopo modifiche al chunking.

Uso:
    uv run python scripts/05_index_qdrant.py [cartella_chunk] [query_di_prova]
"""

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.vectorstore import (  # noqa: E402
    DEFAULT_COLLECTION_NAME,
    DEFAULT_QDRANT_PATH,
    QdrantVectorStore,
    get_embedder,
)

OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "data/processed/docling/dnf2023"
TEST_QUERY = (
    sys.argv[2]
    if len(sys.argv) > 2
    else "Qual è la ripartizione geografica della spesa per i fornitori nel 2023?"
)

CHUNKS_JSON = OUT / "chunks.json"
if not CHUNKS_JSON.exists():
    raise FileNotFoundError(
        f"File {CHUNKS_JSON} non trovato. Esegui prima: uv run python scripts/04_chunk_document.py {OUT}"
    )

chunks_data = json.loads(CHUNKS_JSON.read_text(encoding="utf-8"))
print(f"Caricati {len(chunks_data)} chunk da {CHUNKS_JSON}")

t0 = time.perf_counter()
embedder = get_embedder()
load_model_s = time.perf_counter() - t0
print(f"Modello {embedder.model_name} pronto su {embedder.device} ({load_model_s:.2f}s)")

store = QdrantVectorStore(
    collection_name=DEFAULT_COLLECTION_NAME,
    path=ROOT / DEFAULT_QDRANT_PATH,
)

t1 = time.perf_counter()
report = store.index_chunks(chunks_data, embedder=embedder, replace_existing_doc=True)
index_s = time.perf_counter() - t1

print(
    f"Indicizzazione completata in {index_s:.2f}s | "
    f"Collezione: {report.collection_name} | "
    f"doc_ids: {report.doc_ids} | "
    f"Rimossi vecchi punti: {report.deleted_existing_points} | "
    f"Nuovi punti indicizzati: {report.indexed_points} (dim={report.vector_dimension})"
)
print("-" * 95)

# Esegui una query di verifica
print(f"Query di prova: {TEST_QUERY!r}\n")
results = store.search(TEST_QUERY, embedder=embedder, top_k=3)
for rank, r in enumerate(results, start=1):
    preview = r.content.replace("\n", " ")[:140]
    print(
        f"  #{rank} score={r.score:.4f} | [{r.chunk_type.upper()}] {r.chunk_id} | "
        f"pag={r.page_range} | sez={r.section or '-'} | img={r.image_paths}"
    )
    print(f"     {preview}...\n")

store.close()

