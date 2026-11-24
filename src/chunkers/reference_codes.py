"""Codici di riferimento opzionali (es. GRI) che etichettano il contenuto che segue.

Alcuni documenti contengono marcatori come `[GRI 205-1]`: non sono titoli, ma etichette valide per
il contenuto successivo. Di default non viene riconosciuto nessun codice; chi ne ha bisogno passa
uno o più `ReferencePattern` al chunker (o `--reference-codes gri` agli script).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable, Iterable, Sequence


@dataclass(frozen=True)
class ReferencePattern:
    """Formato di un marcatore.

    - `marker_re`: individua il marcatore nel testo; il gruppo 1 contiene l'elenco dei codici.
    - `code_re`: riconosce un singolo codice dentro l'elenco.
    - `format_code`: forma normalizzata del codice.
    - `mention_re`: individua i codici citati in un testo libero, per esempio una domanda
      ("GRI 205-1 e 205-2"); il gruppo 1 contiene l'elenco dei codici.
    """

    name: str
    marker_re: re.Pattern[str]
    code_re: re.Pattern[str]
    format_code: Callable[[re.Match[str]], str]
    mention_re: re.Pattern[str] | None = None

    def parse(self, body: str) -> list[str]:
        """Codici normalizzati e senza doppioni contenuti nel corpo di un marcatore."""
        codes: list[str] = []
        for token in re.split(r"[,;\s]+", body):
            m = self.code_re.fullmatch(token.strip("[]()."))
            if m is not None and self.format_code(m) not in codes:
                codes.append(self.format_code(m))
        return codes


GRI_PATTERN = ReferencePattern(
    name="gri",
    marker_re=re.compile(r"\[\s*GRI\s+([^\]]+)\]", re.IGNORECASE),
    code_re=re.compile(r"(\d{1,3})[-.](\d{1,2})"),
    format_code=lambda m: f"GRI {m.group(1)}-{m.group(2)}",
    mention_re=re.compile(
        r"\bGRI\s*(\d{1,3}[-.]\d{1,2}(?!\d)"
        r"(?:\s*(?:[,;]|\be\b|\bed\b|\band\b)\s*(?:GRI\s*)?\d{1,3}[-.]\d{1,2}(?!\d))*)",
        re.IGNORECASE,
    ),
)

REFERENCE_PATTERNS: dict[str, ReferencePattern] = {p.name: p for p in (GRI_PATTERN,)}


def find_mentioned_codes(
    text: str, patterns: Iterable[ReferencePattern] = tuple(REFERENCE_PATTERNS.values())
) -> list[str]:
    """Codici citati in un testo libero (`"...al GRI 205-1?"` -> `["GRI 205-1"]`)."""
    codes: list[str] = []
    for pattern in patterns:
        if pattern.mention_re is None:
            continue
        for m in pattern.mention_re.finditer(text):
            codes += [c for c in pattern.parse(m.group(1)) if c not in codes]
    return codes


def format_marker(codes: Sequence[str]) -> str:
    """Riga di testo con i codici attivi (`["GRI 207-1"]` -> `"[GRI 207-1]"`)."""
    return f"[{', '.join(codes)}]" if codes else ""


def split_by_markers(
    text: str, patterns: Sequence[ReferencePattern]
) -> list[tuple[int, str, list[str]]]:
    """Divide `text` in segmenti `(offset, testo, codici)` in corrispondenza dei marcatori.

    I segmenti di testo hanno `codici == []`; i marcatori hanno `testo == ""`.
    """
    found: list[tuple[int, int, list[str]]] = []
    for pattern in patterns:
        for m in pattern.marker_re.finditer(text):
            codes = pattern.parse(m.group(1))
            if codes:
                found.append((m.start(), m.end(), codes))

    segments: list[tuple[int, str, list[str]]] = []
    last = 0
    for start, end, codes in sorted(found):
        if start < last:
            continue
        if text[last:start].strip():
            segments.append((last, text[last:start], []))
        segments.append((start, "", codes))
        last = end
    if text[last:].strip():
        segments.append((last, text[last:], []))
    return segments
