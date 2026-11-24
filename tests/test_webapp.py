#!/usr/bin/env python3
"""Test della web app (API, libreria dei documenti, protezioni) senza modelli né LLM."""

import sys
import tempfile
import threading
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi.testclient import TestClient  # noqa: E402

from src.rag import Answer, OllamaError  # noqa: E402
from src.rag.ingest import EmptyDocumentError, IngestReport  # noqa: E402
from src.rag.progress import IngestProgress  # noqa: E402
from src.vectorstore import SearchResult  # noqa: E402
from src.webapp import main as webapp  # noqa: E402
from src.webapp.engine import EngineBusy, Reply  # noqa: E402
from src.webapp.library import Library, make_doc_id  # noqa: E402

PDF = b"%PDF-1.7\n%finto"


class FakeGenerator:
    model = "modello-test"

    def status(self):
        return "ready"


class FakeEngine:
    def __init__(self):
        self.state, self.error, self.ingesting = "ready", "", False
        self.generator = FakeGenerator()
        self.counts = {"doc1": 3}
        self.ingest_error = None
        self.ask_error = None
        self.release = threading.Event()
        self.release.set()

    def close(self):
        pass

    def document_counts(self):
        return dict(self.counts)

    def ask(self, question, doc_id):
        if self.ask_error:
            raise self.ask_error
        hit = SearchResult(
            1.0, "p", "c", doc_id, "table", "| A |", [], "Sezione", "43-44", [43, 44],
            ["table_images/t1_p43.png"], False, [],
        )
        return Reply(Answer("Sono 694 [pag. 43].", False, [43], [hit], "m"), 0.5, 1.0)

    def delete(self, doc_id):
        if self.ingesting and doc_id in self.counts:
            raise EngineBusy("occupato")
        self.counts.pop(doc_id, None)

    def reserve_ingest(self):
        if self.ingesting:
            raise EngineBusy("occupato")
        self.ingesting = True

    def cancel_ingest(self):
        self.ingesting = False

    def ingest(self, pdf_path, out_dir, *, doc_id, reference_codes, progress):
        try:
            progress.start("estrazione", "Leggo le pagine")
            progress.update(3, 7, "pagine")
            self.release.wait(5)
            if self.ingest_error:
                raise self.ingest_error
            self.codes = reference_codes
            self.counts[doc_id] = 2
            return IngestReport(doc_id, 7, 1, 2, [], 1.0)
        finally:
            self.ingesting = False


def _client(tmp: str):
    engine, library = FakeEngine(), Library(Path(tmp) / "data")
    return TestClient(webapp.create_app(engine, library, load_models=False)), engine, library


def _wait(client, doc_id, status):
    for _ in range(100):
        doc = next((d for d in client.get("/api/documents").json() if d["id"] == doc_id), None)
        if doc and doc["status"] == status:
            return doc
        time.sleep(0.02)
    raise AssertionError(f"{doc_id} non è diventato {status}: {doc}")


def test_progress():
    p = IngestProgress()
    assert [x["state"] for x in p.snapshot()["phases"]] == ["waiting"] * 5
    p.start("estrazione", "Leggo")
    assert p.snapshot()["phases"][0]["percent"] is None  # non ancora misurabile
    p.update(1, 3, "pagine")
    p.step_done("Una pagina")
    p.start("tabelle", "Unisco")
    s = p.snapshot()
    assert [(x["state"], x["percent"]) for x in s["phases"][:3]] == [("done", 100), ("active", None), ("waiting", 0)]
    assert s["action"] == "Unisco" and s["log"][0]["phase"] == "estrazione"
    p.update(5, 0, "cose")  # totale sconosciuto: nessuna percentuale inventata
    assert p.snapshot()["phases"][1]["percent"] is None
    p.fail("rotto")
    s = p.snapshot()
    assert [x["state"] for x in s["phases"][:3]] == ["done", "error", "waiting"] and s["action"] == "rotto"
    time.sleep(0.15)
    assert p.snapshot()["seconds"] == s["seconds"]  # il tempo si ferma
    ok = IngestProgress()
    for phase in ("estrazione", "tabelle", "chunking", "embeddings", "vectordb"):
        ok.start(phase, "x")
    ok.finish()
    assert all(x["state"] == "done" and x["percent"] == 100 for x in ok.snapshot()["phases"])
    print("[OK] test_progress")


def test_chat():
    with tempfile.TemporaryDirectory() as tmp:
        client, engine, _ = _client(tmp)
        with client:
            r = client.post("/api/chat", json={"doc_id": "doc1", "question": "Quanti?"}).json()
            assert r["answer"] == "Sono 694 [pag. 43]." and r["sources"][0]["cited"]
            assert r["sources"][0]["images"] == ["/api/documents/doc1/images/t1_p43.png"]
            assert client.post("/api/chat", json={"doc_id": "no", "question": "q"}).status_code == 404
            assert client.post("/api/chat", json={"doc_id": "doc1", "question": "  "}).status_code == 422
            long = {"doc_id": "doc1", "question": "x" * 2001}
            assert client.post("/api/chat", json=long).status_code == 422
            for error, code in ((EngineBusy("b"), 409), (OllamaError("o"), 502)):
                engine.ask_error = error
                r = client.post("/api/chat", json={"doc_id": "doc1", "question": "q"})
                assert r.status_code == code and r.json()["detail"] == str(error)
    print("[OK] test_chat")


def test_upload():
    with tempfile.TemporaryDirectory() as tmp:
        client, engine, library = _client(tmp)
        with client:
            files = {"file": ("Relazione 2024.pdf", PDF)}
            engine.release.clear()
            r = client.post("/api/documents", files=files, data={"reference_codes": ["gri"]})
            assert r.status_code == 202
            doc_id = r.json()["id"]
            assert doc_id.startswith("relazione-2024-")
            assert _wait(client, doc_id, "processing")["title"] == "Relazione 2024"
            job = _wait(client, doc_id, "processing")["progress"]
            for _ in range(100):  # il thread parte subito dopo la risposta
                if job["phases"][0]["state"] == "active" and job["phases"][0]["percent"]:
                    break
                time.sleep(0.02)
                job = _wait(client, doc_id, "processing")["progress"]
            assert job["action"] == "Leggo le pagine"
            assert (job["phases"][0]["percent"], job["phases"][0]["detail"]) == (42, "3 di 7 pagine")
            # Un solo documento alla volta, e il rifiuto non lascia file
            assert client.post("/api/documents", files=files).status_code == 409
            # Mentre si elabora non si elimina nulla
            assert client.delete(f"/api/documents/{doc_id}").status_code == 409
            assert client.delete("/api/documents/doc1").status_code == 409
            engine.release.set()
            doc = _wait(client, doc_id, "ready")
            assert (doc["pages"], doc["chunks"], doc["has_pdf"]) == (7, 2, True)
            assert doc["progress"]["phases"][0]["state"] == "done"
            assert next(d for d in client.get("/api/documents").json() if d["id"] == "doc1")["progress"] is None
            assert engine.codes == ["gri"]
            assert client.get(f"/api/documents/{doc_id}/pdf").content == PDF
            assert len(list(library.raw_dir.iterdir())) == 1

            for name, content, code in (
                ("a.txt", PDF, 400), ("a.pdf", b"non un pdf", 400), ("a.pdf", b"", 400)
            ):
                assert client.post("/api/documents", files={"file": (name, content)}).status_code == code
            bad = client.post("/api/documents", files=files, data={"reference_codes": ["boh"]})
            assert bad.status_code == 400
            assert not engine.ingesting and len(list(library.raw_dir.iterdir())) == 1

            for error, text in ((EmptyDocumentError("vuoto"), "vuoto"), (RuntimeError("x"), "RuntimeError")):
                engine.ingest_error = error
                failed = client.post("/api/documents", files=files).json()["id"]
                doc = _wait(client, failed, "error")
                assert text in doc["message"] and doc["progress"]["phases"][0]["state"] == "error"
                assert client.delete(f"/api/documents/{failed}").status_code == 204
                assert failed not in {d["id"] for d in client.get("/api/documents").json()}
    print("[OK] test_upload")


def test_delete():
    with tempfile.TemporaryDirectory() as tmp:
        client, engine, library = _client(tmp)
        # Documento creato dagli script: i suoi file non si toccano
        pdf = library.raw_dir / "originale" / "source.pdf"
        out = library.processed_dir / "docling" / "doc1"
        for folder in (pdf.parent, out):
            folder.mkdir(parents=True)
        pdf.write_bytes(PDF)
        (out / "timing.json").write_text('{"pdf": "data/raw/originale/source.pdf"}', encoding="utf-8")
        # Cartella con lo stesso nome del documento, ma non dell'app
        twin = library.raw_dir / "doc1" / "source.pdf"
        twin.parent.mkdir()
        twin.write_bytes(PDF)
        with client:
            doc_id = client.post("/api/documents", files={"file": ("a.pdf", PDF)}).json()["id"]
            assert _wait(client, doc_id, "ready")["managed"]
            assert not _wait(client, "doc1", "ready")["managed"]
            evil = client.delete(f"/api/documents/{doc_id}", headers={"origin": "http://sito.example"})
            assert evil.status_code == 403

            # Un file in uso: il documento resta nell'elenco e si può riprovare
            original = library.remove_files

            def locked(_):
                raise PermissionError("in uso")

            library.remove_files = locked
            r = client.delete(f"/api/documents/{doc_id}")
            assert r.status_code == 409 and "in uso" in r.json()["detail"]
            assert "in uso" in _wait(client, doc_id, "error")["message"]
            library.remove_files = original

            assert client.delete(f"/api/documents/{doc_id}").status_code == 204
            assert not library.source_path(doc_id).parent.exists()
            assert not library.output_dir(doc_id).exists()
            assert client.delete(f"/api/documents/{doc_id}").status_code == 404
            assert "doc1" in engine.counts

            assert client.delete("/api/documents/doc1").status_code == 204
            assert "doc1" not in engine.counts
            assert pdf.is_file() and twin.is_file() and (out / "timing.json").is_file()
            assert client.get("/api/documents").json() == []
            assert client.delete("/api/documents/..%2F..%2Fraw").status_code in (404, 405)
        try:
            library.remove_files("doc1")
        except ValueError:
            pass
        else:
            raise AssertionError("cancellati file non creati dall'app")
    print("[OK] test_delete")


def test_upload_size_limit():
    with tempfile.TemporaryDirectory() as tmp:
        client, engine, library = _client(tmp)
        original, webapp.MAX_UPLOAD_BYTES = webapp.MAX_UPLOAD_BYTES, 10
        try:
            with client:
                r = client.post("/api/documents", files={"file": ("a.pdf", PDF)})
            assert r.status_code == 413 and not engine.ingesting
            assert not list(library.raw_dir.iterdir())
        finally:
            webapp.MAX_UPLOAD_BYTES = original
    print("[OK] test_upload_size_limit")


def test_protections():
    with tempfile.TemporaryDirectory() as tmp:
        client, _, library = _client(tmp)
        secret = Path(tmp) / "segreto.pdf"
        secret.write_bytes(PDF)
        out = library.processed_dir / "docling" / "doc1"
        out.mkdir(parents=True)
        (out / "timing.json").write_text('{"pdf": "../segreto.pdf"}', encoding="utf-8")
        with client:
            body = {"doc_id": "doc1", "question": "q"}
            evil = client.post("/api/chat", json=body, headers={"origin": "http://sito.example"})
            assert evil.status_code == 403
            same = client.post("/api/chat", json=body, headers={"origin": "http://testserver"})
            assert same.status_code == 200
            assert client.get("/api/status", headers={"host": "sito.example"}).status_code == 400
            # Nessun file fuori dalla cartella dei dati
            for url in (
                "/api/documents/doc1/pdf",
                "/api/documents/..%2F..%2Fsegreto/pdf",
                "/api/documents/doc1/images/..%2F..%2Ftiming.json",
                "/api/documents/doc1/images/timing.json",
            ):
                assert client.get(url).status_code == 404, url
            assert client.get("/").headers["x-content-type-options"] == "nosniff"
    print("[OK] test_protections")


def test_library():
    with tempfile.TemporaryDirectory() as tmp:
        library = Library(Path(tmp) / "data")
        when = datetime(2026, 1, 2, 3, 4, 5)
        assert make_doc_id("Àè § !!.pdf", when).startswith("documento-20260102-030405")
        assert make_doc_id("../../x.pdf", when).startswith("x-")
        pdf = library.raw_dir / "x" / "source.pdf"
        pdf.parent.mkdir(parents=True)
        pdf.write_bytes(PDF)
        out = library.processed_dir / "docling" / "vecchio"
        out.mkdir(parents=True)
        # Percorso scritto da Windows negli script
        (out / "timing.json").write_text('{"pdf": "data\\\\raw\\\\x\\\\source.pdf", "pages": 9}', encoding="utf-8")
        doc = library.list({"vecchio": 5})[0]
        assert (doc["id"], doc["pages"], doc["chunks"], doc["has_pdf"]) == ("vecchio", 9, 5, True)
        library.write_meta("nuovo", title="N", status="processing")
        library.mark_interrupted()
        assert library.list({})[0]["status"] == "error"
        try:
            library.output_dir("../x")
        except ValueError:
            pass
        else:
            raise AssertionError("identificativo non valido accettato")
    print("[OK] test_library")


if __name__ == "__main__":
    test_progress()
    test_chat()
    test_upload()
    test_delete()
    test_upload_size_limit()
    test_protections()
    test_library()
    print("\nTUTTI I TEST WEB APP SUPERATI!")
