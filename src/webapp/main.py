"""RAG Chatbot WebApp - Backend FastAPI."""

import asyncio
import json
import os
import shutil
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from src.extractors.pdf_converter import convert_pdf, convert_image

app = FastAPI(title="RAG Chatbot")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Directories
BASE_DIR = Path(__file__).parent.parent.parent.resolve()
STATIC_DIR = BASE_DIR / "static"
DATA_DIR = BASE_DIR / "data" / "raw"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Serve static files
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
async def root():
    """Serve home page."""
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/rag.html")
async def rag_page():
    """Serve RAG chat page."""
    return FileResponse(STATIC_DIR / "rag.html")


@app.post("/api/upload")
async def upload_pdf(file: UploadFile = File(...), strategy: str = "vision"):
    """Upload PDF and process with OCR."""
    if not file.filename:
        raise HTTPException(400, "Missing filename")

    suffix = Path(file.filename).suffix.lower()
    allowed = {".pdf", ".png", ".jpg", ".jpeg", ".webp"}
    if suffix not in allowed:
        raise HTTPException(400, f"Unsupported: {suffix}")

    # Validate strategy
    if strategy not in {"vision", "docling"}:
        strategy = "vision"

    # Save file
    doc_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    doc_dir = DATA_DIR / doc_id
    doc_dir.mkdir()
    src_path = doc_dir / f"source{suffix}"

    with src_path.open("wb") as f:
        shutil.copyfileobj(file.file, f)

    # Process with selected strategy
    try:
        if suffix == ".pdf":
            result = await convert_pdf(src_path, doc_dir, strategy=strategy)
        else:
            result = await convert_image(src_path, doc_dir, strategy=strategy)

        return {
            "doc_id": doc_id,
            "pages": result.pages,
            "strategy": strategy,
            "markdown": result.markdown[:1000] + "..." if len(result.markdown) > 1000 else result.markdown
        }
    except Exception as e:
        shutil.rmtree(doc_dir, ignore_errors=True)
        raise HTTPException(500, str(e))


@app.get("/api/documents/{doc_id}")
async def get_document(doc_id: str):
    """Get processed document markdown."""
    doc_dir = DATA_DIR / doc_id
    md_path = doc_dir / "output.md"
    if not md_path.exists():
        raise HTTPException(404, "Document not found")
    return {"markdown": md_path.read_text(encoding="utf-8")}


@app.websocket("/ws/chat")
async def chat_websocket(websocket: WebSocket):
    """WebSocket chat endpoint."""
    await websocket.accept()

    try:
        while True:
            data = await websocket.receive_json()
            message = data.get("message", "")
            doc_id = data.get("doc_id")

            # Echo response for now (TODO: implement actual RAG)
            response = f"Received: {message}"
            if doc_id:
                response += f" (referencing doc: {doc_id})"

            await websocket.send_json({
                "type": "response",
                "content": response
            })

    except Exception:
        await websocket.close()


@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "rag-chatbot",
        "version": "0.1.0",
        "features": {
            "ocr": "available",
            "websocket": "available",
            "rag": "pending"
        }
    }
