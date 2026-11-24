"""Generazione della risposta con un LLM locale servito da Ollama.

L'LLM riceve i chunk del retrieval come estratti numerati, con pagina e sezione, e deve rispondere
solo con quelli. Il retrieval restituisce sempre dei chunk, anche per le domande fuori perimetro:
riconoscere che la risposta manca è compito di questo passaggio (`NOT_FOUND_ANSWER`).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Sequence

import httpx

from src.vectorstore import SearchResult

DEFAULT_OLLAMA_URL: str = "http://127.0.0.1:11434"
DEFAULT_LLM_MODEL: str = "qwen3.5:4b"
DEFAULT_NUM_CTX: int = 8192
DEFAULT_MAX_ANSWER_TOKENS: int = 512
NOT_FOUND_ANSWER: str = "Non presente nel documento."

SYSTEM_PROMPT: str = f"""Rispondi a domande su un documento usando SOLO gli estratti forniti.

Regole:
1. Rispondi in italiano, SOLO a ciò che è chiesto, in una o due frasi. Non aggiungere contesto, definizioni, cause, commenti o altri dati, anche se sono negli estratti.
2. Rispondi a tutte le parti della domanda: se chiede più anni, soggetti o voci, riporta il valore di ciascuno. Se chiede quali dati sono riportati, elenca i dati con i loro valori.
3. Termina ogni frase con la pagina da cui viene l'informazione, nel formato [pag. N], usando le pagine indicate negli estratti.
4. Copia numeri, unità di misura e nomi esattamente come sono scritti. Nelle tabelle leggi la riga e la colonna giuste: ogni riga è un dato distinto.
5. Non sommare, sottrarre o combinare valori. Se è chiesta una variazione, riporta i valori degli anni indicati.
6. Non usare conoscenze esterne e non fare supposizioni. Non attribuire a un soggetto o a una grandezza ciò che gli estratti dicono di un altro.
7. Verifica che anno, soggetto e grandezza richiesti coincidano con quelli degli estratti. Un dato simile ma riferito a un altro anno, a un altro soggetto o a un'altra grandezza non è una risposta. Lo stesso dato espresso con parole diverse invece lo è: in quel caso rispondi senza esprimere dubbi.
8. Se gli estratti non contengono la risposta, scrivi soltanto: {NOT_FOUND_ANSWER}"""

_CITATION_RE = re.compile(
    r"pagg?(?:\.|in[ae])?\s*(\d+(?:\s*(?:[-–,]|\be\b)\s*\d+)*)", re.IGNORECASE
)


class OllamaError(RuntimeError):
    """Ollama non raggiungibile, modello assente o risposta non valida."""


@dataclass
class Answer:
    """Risposta dell'LLM con gli estratti usati come contesto."""

    text: str
    refused: bool
    cited_pages: list[int]
    context: list[SearchResult]
    model: str
    prompt_tokens: int = 0
    answer_tokens: int = 0
    truncated: bool = False  # risposta tagliata, o prompt troppo lungo per `num_ctx`


def build_user_prompt(question: str, context: Sequence[SearchResult]) -> str:
    """Estratti numerati (documento, pagina, sezione, contenuto) seguiti dalla domanda."""
    blocks: list[str] = []
    for i, hit in enumerate(context, start=1):
        pages = hit.page_range or ", ".join(str(p) for p in hit.pages) or "?"
        head = f"[Estratto {i} | documento: {hit.doc_id} | pag. {pages}"
        head += f" | sezione: {hit.section}]" if hit.section else "]"
        blocks.append(f"{head}\n{hit.content}")
    return "\n\n".join(blocks) + f"\n\nDomanda: {question}"


def is_refusal(text: str) -> bool:
    """True se la risposta è il rifiuto concordato, e non una risposta che lo contiene in coda."""
    clean = re.sub(r"[*_`\"“”]", "", text).strip().lower()
    return clean.startswith(NOT_FOUND_ANSWER.rstrip(".").lower())


def parse_cited_pages(text: str) -> list[int]:
    """Pagine citate (`[pag. 43]`, `[pag. 43-44]`, `[pagg. 43, 45]`), senza doppioni e in ordine."""
    pages: set[int] = set()
    for m in _CITATION_RE.finditer(text):
        for start, end in re.findall(r"(\d+)(?:\s*[-–]\s*(\d+))?", m.group(1)):
            first, last = int(start), int(end or start)
            pages.update(range(first, last + 1) if first <= last <= first + 50 else (first, last))
    return sorted(pages)


class OllamaGenerator:
    """Genera risposte con un modello Ollama, a partire dai chunk del retrieval."""

    def __init__(
        self,
        model: str = DEFAULT_LLM_MODEL,
        *,
        base_url: str = DEFAULT_OLLAMA_URL,
        num_ctx: int = DEFAULT_NUM_CTX,
        max_answer_tokens: int = DEFAULT_MAX_ANSWER_TOKENS,
        timeout: float = 300.0,
        client: httpx.Client | None = None,
    ) -> None:
        self.model = model
        self.num_ctx = num_ctx
        self.max_answer_tokens = max_answer_tokens
        self.client = client or httpx.Client(base_url=base_url, timeout=timeout)

    def answer(self, question: str, context: Sequence[SearchResult]) -> Answer:
        """Risponde alla domanda usando solo `context`; senza estratti non interroga l'LLM."""
        hits = list(context)
        if not hits:
            return Answer(NOT_FOUND_ANSWER, True, [], [], self.model)

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_prompt(question, hits)},
        ]
        data = self._chat(messages)
        text = str(data.get("message", {}).get("content", "")).strip()
        if not text:
            raise OllamaError(f"Risposta vuota da {self.model}.")
        prompt_tokens = int(data.get("prompt_eval_count", 0))
        return Answer(
            text=text,
            refused=is_refusal(text),
            cited_pages=parse_cited_pages(text),
            context=hits,
            model=self.model,
            prompt_tokens=prompt_tokens,
            answer_tokens=int(data.get("eval_count", 0)),
            truncated=data.get("done_reason") == "length"
            or prompt_tokens + self.max_answer_tokens > self.num_ctx,
        )

    def _chat(self, messages: list[dict[str, str]]) -> dict[str, Any]:
        """Una chiamata a `/api/chat`."""
        payload = {
            "model": self.model,
            "stream": False,
            "think": False,  # questi modelli ragionano di default: molto più lento, senza vantaggi misurati
            "messages": messages,
            # Le penalità di ripetizione scoraggerebbero la copia esatta di numeri e nomi
            "options": {
                "temperature": 0,
                "seed": 1,
                "presence_penalty": 0.0,
                "frequency_penalty": 0.0,
                "repeat_penalty": 1.0,
                "num_ctx": self.num_ctx,
                "num_predict": self.max_answer_tokens,
            },
        }
        try:
            response = self.client.post("/api/chat", json=payload)
        except httpx.ConnectError as exc:
            raise OllamaError("Ollama non è raggiungibile: avvia l'app Ollama e riprova.") from exc
        except httpx.TimeoutException as exc:
            raise OllamaError(f"Ollama non ha risposto in tempo (modello {self.model}).") from exc
        if response.status_code == 404:
            raise OllamaError(f"Modello assente: esegui `ollama pull {self.model}`.")
        if response.is_error:
            raise OllamaError(f"Errore di Ollama {response.status_code}: {response.text[:300]}")
        return response.json()

    def unload(self) -> None:
        """Libera la memoria della GPU occupata dal modello (si ricarica alla domanda successiva)."""
        try:
            self.client.post("/api/generate", json={"model": self.model, "keep_alive": 0})
        except httpx.HTTPError:
            pass  # Ollama spento: non c'è nulla da liberare

    def status(self) -> str:
        """`"ready"`, `"missing_model"` oppure `"unreachable"`."""
        try:
            response = self.client.get("/api/tags", timeout=3.0)
            names = {m.get("name") for m in response.json().get("models", [])}
        except (httpx.HTTPError, ValueError):
            return "unreachable"
        return "ready" if self.model in names else "missing_model"
