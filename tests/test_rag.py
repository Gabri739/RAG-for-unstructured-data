#!/usr/bin/env python3
"""Test della generazione (prompt, rifiuti, citazioni, errori di Ollama) senza un LLM reale."""

import json
import sys
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.rag import (  # noqa: E402
    NOT_FOUND_ANSWER,
    OllamaError,
    OllamaGenerator,
    build_user_prompt,
    is_refusal,
    parse_cited_pages,
    retrieve,
)
from src.vectorstore import SearchResult  # noqa: E402


def _hit(text: str, pages: list[int], section: str = "Fornitori") -> SearchResult:
    page_range = "-".join(str(p) for p in pages)
    return SearchResult(
        1.0, text, text, "doc", "text", text, [], section, page_range, pages, [], False, []
    )


def _generator(handler) -> OllamaGenerator:
    client = httpx.Client(base_url="http://test", transport=httpx.MockTransport(handler))
    return OllamaGenerator("modello-test", client=client)


def test_parse_cited_pages():
    assert parse_cited_pages("Sono 694 [pag. 43].") == [43]
    text = "A [pag. 43-44] e B [pagg. 50, 52 e 60] [pagina 7]"
    assert parse_cited_pages(text) == [7, 43, 44, 50, 52, 60]
    assert parse_cited_pages("Nel 2023 le imprese pagate erano 694") == []
    print("[OK] test_parse_cited_pages")


def test_is_refusal():
    assert is_refusal(NOT_FOUND_ANSWER)
    assert is_refusal("**Non presente nel documento**")
    assert not is_refusal("Sono 694 [pag. 43]. Il dato 2024 è non presente nel documento.")
    print("[OK] test_is_refusal")


def test_prompt_contains_pages_and_question_last():
    prompt = build_user_prompt("Quanti?", [_hit("uno", [43]), _hit("due", [44, 45], section="")])
    assert "[Estratto 1 | documento: doc | pag. 43 | sezione: Fornitori]\nuno" in prompt
    assert "[Estratto 2 | documento: doc | pag. 44-45]\ndue" in prompt
    assert prompt.endswith("Domanda: Quanti?")
    print("[OK] test_prompt_contains_pages_and_question_last")


def test_answer():
    seen = {}

    def ok(request: httpx.Request) -> httpx.Response:
        seen.update(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "message": {"content": " Sono 694 [pag. 43]. "},
                "prompt_eval_count": 100,
                "eval_count": 9,
                "done_reason": "stop",
            },
        )

    a = _generator(ok).answer("Quanti?", [_hit("694 fornitori", [43])])
    assert a.text == "Sono 694 [pag. 43]." and a.cited_pages == [43]
    assert not a.refused and not a.truncated
    assert seen["think"] is False and seen["options"]["num_ctx"] == 8192
    assert [m["role"] for m in seen["messages"]] == ["system", "user"]

    def cut(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"message": {"content": "x"}, "done_reason": "length"})

    assert _generator(cut).answer("q", [_hit("t", [1])]).truncated

    # Senza estratti non si interroga l'LLM
    def never(request: httpx.Request) -> httpx.Response:
        raise AssertionError("LLM interrogato senza contesto")

    empty = _generator(never).answer("q", [])
    assert empty.refused and empty.text == NOT_FOUND_ANSWER
    print("[OK] test_answer")


def test_status_and_unload():
    def tags(*names):
        return lambda r: httpx.Response(200, json={"models": [{"name": n} for n in names]})

    assert _generator(tags("modello-test")).status() == "ready"
    assert _generator(tags("altro")).status() == "missing_model"

    def unreachable(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("no", request=request)

    assert _generator(unreachable).status() == "unreachable"
    _generator(unreachable).unload()  # non deve sollevare errori
    print("[OK] test_status_and_unload")


def test_ollama_errors():
    def unreachable(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("no", request=request)

    cases = {
        "non è raggiungibile": unreachable,
        "ollama pull modello-test": lambda r: httpx.Response(404),
        "Errore di Ollama 500": lambda r: httpx.Response(500, text="boom"),
        "Risposta vuota": lambda r: httpx.Response(200, json={"message": {"content": "  "}}),
    }
    for expected, handler in cases.items():
        try:
            _generator(handler).answer("q", [_hit("t", [1])])
        except OllamaError as exc:
            assert expected in str(exc), exc
        else:
            raise AssertionError(f"errore atteso: {expected}")
    print("[OK] test_ollama_errors")


def test_retrieve_passes_options():
    class FakeStore:
        def hybrid_candidates(self, question, **kw):
            assert (question, kw["limit"], kw["doc_id"]) == ("q", 15, "d")
            return ["a", "b"]

    class FakeReranker:
        def rerank(self, question, pool, *, top_k):
            assert pool == ["a", "b"] and top_k == 6
            return ["b"]

    assert retrieve("q", FakeStore(), reranker=FakeReranker(), doc_id="d") == ["b"]
    print("[OK] test_retrieve_passes_options")


if __name__ == "__main__":
    test_parse_cited_pages()
    test_is_refusal()
    test_prompt_contains_pages_and_question_last()
    test_answer()
    test_status_and_unload()
    test_ollama_errors()
    test_retrieve_passes_options()
    print("\nTUTTI I TEST RAG SUPERATI!")
