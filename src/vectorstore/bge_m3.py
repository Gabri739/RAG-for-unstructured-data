"""Modulo di embedding basato su BAAI/bge-m3 tramite FlagEmbedding (`BGEM3FlagModel`).

Con una singola passata del modello estrae contemporaneamente:
  1. Il vettore DENSO (`dense_vecs`: 1024 float normalizzati L2) per la ricerca semantica classica.
  2. Il vettore SPARSO (`lexical_weights`: indici token del vocabolario + pesi appresi) salvato
     su Qdrant insieme al denso così da avere il database già pronto per la ricerca ibrida
     senza dover ricalcolare gli embedding in futuro.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Sequence

import torch

BGE_M3_MODEL_NAME: str = "BAAI/bge-m3"
BGE_M3_DIM: int = 1024
DEFAULT_MAX_LENGTH: int = 2048  # BGE-M3 supporta fino a 8192 token


@dataclass
class SparseEmbedding:
    """Rappresenta un vettore sparso (indici interi dei token e relativi pesi positivi)."""

    indices: list[int]
    values: list[float]


@dataclass
class BGEM3Output:
    """Contiene sia i vettori densi (1024-d) sia i vettori sparsi per un batch di testi."""

    dense_vecs: list[list[float]]
    sparse_vecs: list[SparseEmbedding]


def _lexical_dict_to_sparse(weights: dict[str | int, float]) -> SparseEmbedding:
    """Converte il dizionario `lexical_weights` di FlagEmbedding (`{str(token_id): weight}`)
    nelle due liste parallele `(indices, values)` richieste da `qdrant_client.models.SparseVector`.
    """
    indices: list[int] = []
    values: list[float] = []
    for token_id, weight in weights.items():
        w = float(weight)
        if w <= 0.0:
            continue
        indices.append(int(token_id))
        values.append(w)
    return SparseEmbedding(indices=indices, values=values)


def _resolve_model_path(model_name: str) -> str:
    """Scarica (o risolve dalla cache locale) solo i file PyTorch necessari di `BAAI/bge-m3`,
    ignorando il duplicato ONNX da 2.27 GB (`onnx/model.onnx_data`) presente nel repo HuggingFace.
    """
    if Path(model_name).exists():
        return model_name
    from huggingface_hub import snapshot_download

    try:
        return snapshot_download(
            repo_id=model_name,
            ignore_patterns=["onnx/*", "*.onnx*", "imgs/*", "*.DS_Store"],
            local_files_only=True,
        )
    except Exception:
        return snapshot_download(
            repo_id=model_name,
            ignore_patterns=["onnx/*", "*.onnx*", "imgs/*", "*.DS_Store"],
        )


class BGEM3Embedder:
    """Calcola gli embedding densi (1024-d) e sparsi (lexical weights) con `BAAI/bge-m3`."""

    def __init__(
        self,
        model_name: str = BGE_M3_MODEL_NAME,
        *,
        device: str | None = None,
        max_length: int = DEFAULT_MAX_LENGTH,
        batch_size: int = 8,
    ) -> None:
        from FlagEmbedding import BGEM3FlagModel

        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_length = max_length
        self.batch_size = batch_size
        self.dimension = BGE_M3_DIM

        resolved_path = _resolve_model_path(model_name)
        use_fp16 = self.device.startswith("cuda")
        self.model = BGEM3FlagModel(
            model_name_or_path=resolved_path,
            normalize_embeddings=True,
            use_fp16=use_fp16,
            devices=self.device,
            batch_size=batch_size,
            query_max_length=max_length,
            passage_max_length=max_length,
        )

    def embed_texts_both(
        self,
        texts: Sequence[str],
        *,
        batch_size: int | None = None,
    ) -> BGEM3Output:
        """Calcola in un solo passaggio sia i vettori densi (1024-d) sia i vettori sparsi."""
        if not texts:
            return BGEM3Output(dense_vecs=[], sparse_vecs=[])

        bs = batch_size or self.batch_size
        out = self.model.encode(
            list(texts),
            batch_size=bs,
            max_length=self.max_length,
            return_dense=True,
            return_sparse=True,
            return_colbert_vecs=False,
        )
        dense_list: list[list[float]] = out["dense_vecs"].tolist()
        sparse_list: list[SparseEmbedding] = [
            _lexical_dict_to_sparse(lw) for lw in out["lexical_weights"]
        ]
        return BGEM3Output(dense_vecs=dense_list, sparse_vecs=sparse_list)

    def embed_texts(
        self,
        texts: Sequence[str],
        *,
        batch_size: int | None = None,
    ) -> list[list[float]]:
        """Restituisce i soli vettori densi (1024 float) per una lista di testi."""
        if not texts:
            return []
        bs = batch_size or self.batch_size
        out = self.model.encode(
            list(texts),
            batch_size=bs,
            max_length=self.max_length,
            return_dense=True,
            return_sparse=False,
            return_colbert_vecs=False,
        )
        return out["dense_vecs"].tolist()

    def embed_query(self, query: str) -> list[float]:
        """Calcola il vettore denso (1024-d) per la ricerca classica di una domanda utente."""
        return self.embed_texts([query], batch_size=1)[0]


@lru_cache(maxsize=1)
def get_embedder(
    model_name: str = BGE_M3_MODEL_NAME,
    max_length: int = DEFAULT_MAX_LENGTH,
) -> BGEM3Embedder:
    """Carica e mette in cache una singola istanza di `BGEM3Embedder`."""
    return BGEM3Embedder(model_name=model_name, max_length=max_length)
