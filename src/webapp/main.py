"""Web app: carica PDF e fai domande sul loro contenuto.

Avvio (solo su questo computer):
    uv run uvicorn src.webapp.main:app --host 127.0.0.1 --port 8000

Non ha autenticazione: non esporla su internet così com'è.
"""

from __future__ import annotations

import logging
import shutil
import threading
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import quote, urlsplit

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from src.chunkers import REFERENCE_PATTERNS
from src.rag import OllamaError
from src.rag.ingest import EmptyDocumentError
from src.rag.progress import IngestProgress
from src.vectorstore import SearchResult

from .engine import EngineBusy, EngineNotReady, RagEngine
from .library import Library, make_doc_id

_log = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent.parent
STATIC_DIR = BASE_DIR / "static"
MAX_UPLOAD_BYTES = 100 * 1024 * 1024
SECURITY_HEADERS = {
    "X-Content-Type-Options": "nosniff",
    "Referrer-Policy": "no-referrer",
    "Content-Security-Policy": "default-src 'self'; img-src 'self' data:; object-src 'none'; "
    "base-uri 'none'; frame-ancestors 'none'",
}


class Question(BaseModel):
    doc_id: str = Field(min_length=1, max_length=200)
    question: str = Field(min_length=1, max_length=2000)


def _source(doc_id: str, n: int, hit: SearchResult, cited: list[int]) -> dict[str, Any]:
    return {
        "n": n,
        "pages": hit.pages,
        "page_range": hit.page_range or ", ".join(str(p) for p in hit.pages),
        "section": hit.section,
        "type": hit.chunk_type,
        "content": hit.content,
        "cited": bool(set(hit.pages) & set(cited)),
        "images": [
            f"/api/documents/{quote(doc_id)}/images/{quote(Path(p).name)}" for p in hit.image_paths
        ],
    }


def create_app(engine: RagEngine, library: Library, *, load_models: bool = True) -> FastAPI:
    # Avanzamento dei caricamenti di questa sessione, per documento
    jobs: dict[str, IngestProgress] = {}

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        library.mark_interrupted()
        if load_models:
            threading.Thread(target=engine.load, name="load-models", daemon=True).start()
        yield
        engine.close()

    app = FastAPI(title="RAG sui documenti", lifespan=lifespan)
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=["127.0.0.1", "localhost", "testserver"])

    @app.middleware("http")
    async def protect(request: Request, call_next):
        # Un altro sito aperto nel browser non deve poter inviare richieste a questa app
        origin = request.headers.get("origin")
        if request.method not in {"GET", "HEAD"} and origin:
            if urlsplit(origin).netloc != request.headers.get("host"):
                return JSONResponse({"detail": "Origine non consentita."}, status_code=403)
        response = await call_next(request)
        is_pdf = response.headers.get("content-type", "").startswith("application/pdf")
        for name, value in SECURITY_HEADERS.items():
            # La CSP può impedire al browser di mostrare il PDF
            if not (is_pdf and name == "Content-Security-Policy"):
                response.headers.setdefault(name, value)
        return response

    @app.exception_handler(EngineBusy)
    async def busy(_: Request, exc: EngineBusy):
        return JSONResponse({"detail": str(exc)}, status_code=409)

    @app.exception_handler(EngineNotReady)
    async def not_ready(_: Request, exc: EngineNotReady):
        return JSONResponse({"detail": str(exc)}, status_code=503)

    @app.exception_handler(OllamaError)
    async def ollama_error(_: Request, exc: OllamaError):
        return JSONResponse({"detail": str(exc)}, status_code=502)

    @app.get("/")
    def home():
        return FileResponse(STATIC_DIR / "index.html")

    @app.get("/rag.html")
    def rag_page():
        return FileResponse(STATIC_DIR / "rag.html")

    @app.get("/api/status")
    def status():
        return {
            "state": engine.state,
            "error": engine.error,
            "ingesting": engine.ingesting,
            "llm": engine.generator.status(),
            "model": engine.generator.model,
            "reference_codes": sorted(REFERENCE_PATTERNS),
        }

    @app.get("/api/documents")
    def documents():
        docs = library.list(engine.document_counts())
        for doc in docs:
            job = jobs.get(doc["id"])
            doc["progress"] = job.snapshot() if job is not None else None
        return docs

    @app.post("/api/documents", status_code=202)
    def upload(file: UploadFile = File(...), reference_codes: list[str] = Form(default=[])):
        filename = Path(file.filename or "").name
        if Path(filename).suffix.lower() != ".pdf":
            raise HTTPException(400, "Sono accettati solo file PDF.")
        unknown = set(reference_codes) - set(REFERENCE_PATTERNS)
        if unknown:
            raise HTTPException(400, f"Codici di riferimento sconosciuti: {', '.join(sorted(unknown))}")

        engine.reserve_ingest()
        doc_id = make_doc_id(filename)
        path = library.source_path(doc_id)
        try:
            path.parent.mkdir(parents=True)
            _save_pdf(file, path)
            library.write_meta(
                doc_id,
                title=Path(filename).stem,
                status="processing",
                message="",
                created=datetime.now().isoformat(timespec="seconds"),
            )
            jobs[doc_id] = IngestProgress()
            threading.Thread(
                target=_ingest, args=(doc_id, path, reference_codes), name="ingest", daemon=True
            ).start()
        except BaseException:
            engine.cancel_ingest()
            jobs.pop(doc_id, None)
            shutil.rmtree(path.parent, ignore_errors=True)
            shutil.rmtree(library.output_dir(doc_id), ignore_errors=True)
            raise
        return {"id": doc_id}

    def _ingest(doc_id: str, path: Path, reference_codes: list[str]) -> None:
        progress = jobs[doc_id]
        try:
            report = engine.ingest(
                path,
                library.output_dir(doc_id),
                doc_id=doc_id,
                reference_codes=reference_codes,
                progress=progress,
            )
        except EmptyDocumentError as exc:
            progress.fail(str(exc))
            library.write_meta(doc_id, status="error", message=str(exc))
        except Exception as exc:
            _log.exception("Elaborazione di %s fallita", doc_id)
            message = f"Elaborazione fallita ({type(exc).__name__})."
            progress.fail(message)
            library.write_meta(doc_id, status="error", message=message)
        else:
            progress.finish()
            note = f"OCR su {len(report.ocr_pages)} pagine" if report.ocr_pages else ""
            library.write_meta(doc_id, status="ready", message=note, pages=report.pages)

    @app.delete("/api/documents/{doc_id}", status_code=204)
    def delete_document(doc_id: str):
        """Elimina un documento dall'indice e, se caricato dall'app, anche i suoi file.

        Per i documenti creati dagli script i file restano su disco: l'app non li ha scritti.
        """
        entry = next((d for d in library.list(engine.document_counts()) if d["id"] == doc_id), None)
        if entry is None:
            raise HTTPException(404, "Documento non trovato.")
        if entry["status"] == "processing":
            raise HTTPException(409, "Il documento è ancora in elaborazione.")
        engine.delete(doc_id)
        jobs.pop(doc_id, None)
        if entry["managed"]:
            try:
                library.remove_files(doc_id)
            except OSError:
                _log.exception("File di %s non cancellati", doc_id)
                message = "Alcuni file sono in uso: chiudi il PDF e riprova a eliminare."
                library.write_meta(doc_id, status="error", message=message)
                raise HTTPException(409, message) from None

    @app.get("/api/documents/{doc_id}/pdf")
    def pdf(doc_id: str):
        path = library.find_pdf(doc_id)
        if path is None:
            raise HTTPException(404, "PDF non disponibile.")
        return FileResponse(path, media_type="application/pdf", content_disposition_type="inline")

    @app.get("/api/documents/{doc_id}/images/{name}")
    def image(doc_id: str, name: str):
        path = library.find_image(doc_id, name)
        if path is None:
            raise HTTPException(404, "Immagine non disponibile.")
        return FileResponse(path, media_type="image/png")

    @app.post("/api/chat")
    def chat(body: Question):
        question = body.question.strip()
        if not question:
            raise HTTPException(422, "Scrivi una domanda.")
        if engine.state == "ready" and body.doc_id not in engine.document_counts():
            raise HTTPException(404, "Documento non trovato.")
        reply = engine.ask(question, body.doc_id)
        answer = reply.answer
        return {
            "answer": answer.text,
            "refused": answer.refused,
            "truncated": answer.truncated,
            "cited_pages": answer.cited_pages,
            "sources": [
                _source(body.doc_id, n, hit, answer.cited_pages)
                for n, hit in enumerate(answer.context, start=1)
            ],
            "retrieval_s": reply.retrieval_s,
            "generation_s": reply.generation_s,
        }

    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
    return app


def _save_pdf(file: UploadFile, path: Path) -> None:
    """Scrive il PDF su disco controllando firma e dimensione."""
    size = 0
    with path.open("wb") as out:
        while block := file.file.read(1024 * 1024):
            if size == 0 and not block.startswith(b"%PDF-"):
                raise HTTPException(400, "Il file non è un PDF valido.")
            size += len(block)
            if size > MAX_UPLOAD_BYTES:
                raise HTTPException(413, f"Il PDF supera {MAX_UPLOAD_BYTES // 1024 // 1024} MB.")
            out.write(block)
    if size == 0:
        raise HTTPException(400, "Il file è vuoto.")


app = create_app(RagEngine(BASE_DIR / "data" / "qdrant"), Library(BASE_DIR / "data"))
