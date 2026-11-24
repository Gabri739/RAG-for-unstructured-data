"""Riconoscimento di testate, piè di pagina e numeri di pagina per ripetizione.

Docling marca quasi sempre questi elementi come `page_header` / `page_footer`, ma a volte li
classifica come testo o titolo. Qui vengono riconosciuti senza conoscere il documento: un testo
breve che compare alla stessa altezza, vicino al bordo, su più pagine vicine è una testata.
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from docling_core.types.doc import DocItemLabel
from docling_core.types.doc.common.content_layer import ContentLayer

if TYPE_CHECKING:
    from docling_core.types.doc import DoclingDocument

MARGIN_ZONE: float = 0.12  # frazione di pagina in alto e in basso
POSITION_TOLERANCE: float = 0.015  # differenza massima di altezza tra due testate uguali
MAX_RUNNING_CHARS: int = 120
MIN_PAGES: int = 3  # 2 nei documenti fino a 3 pagine
MAX_PAGE_GAP: int = 2  # testate alternate pari/dispari distano 2 pagine

_PAGE_NUMBER_RE = re.compile(r"^\W*\d{1,4}(?:\W{1,3}\d{1,4})?\W*$")
_FURNITURE_LABELS = {DocItemLabel.PAGE_HEADER, DocItemLabel.PAGE_FOOTER}


@dataclass
class _Occurrence:
    ref: str
    text: str
    page: int
    top: float
    label: Any
    is_furniture: bool


def page_box(doc: DoclingDocument, prov: Any) -> tuple[float, float, float, float] | None:
    """Riquadro `(sinistra, alto, destra, basso)` in frazioni di pagina (0-1, origine in alto)."""
    page = doc.pages.get(prov.page_no)
    if page is None or page.size is None or not page.size.width or not page.size.height:
        return None
    box = prov.bbox.to_top_left_origin(page_height=page.size.height)
    w, h = page.size.width, page.size.height
    return box.l / w, box.t / h, box.r / w, box.b / h


def is_page_number(text: str) -> bool:
    """True per un testo che è solo un numero di pagina (`12`, `- 12 -`, `12 / 80`)."""
    return bool(_PAGE_NUMBER_RE.match(text.strip()))


def _margin_occurrences(doc: DoclingDocument) -> list[_Occurrence]:
    """Testi brevi che stanno interamente nella fascia alta o bassa della pagina."""
    found: list[_Occurrence] = []
    for item in doc.texts:
        text = re.sub(r"\s+", " ", item.text or "").strip().lower()
        if not text or len(text) > MAX_RUNNING_CHARS or not item.prov:
            continue
        if item.label == DocItemLabel.CAPTION:
            continue
        box = page_box(doc, item.prov[0])
        if box is None or not (box[3] <= MARGIN_ZONE or box[1] >= 1 - MARGIN_ZONE):
            continue
        is_furniture = item.label in _FURNITURE_LABELS or item.content_layer == ContentLayer.FURNITURE
        found.append(
            _Occurrence(item.self_ref, text, item.prov[0].page_no, box[1], item.label, is_furniture)
        )
    return found


def _repeats_nearby(occ: _Occurrence, group: list[_Occurrence], min_pages: int) -> bool:
    """True se `group` ha, alla stessa altezza di `occ`, almeno `min_pages` pagine vicine tra loro.

    Un titolo che ricorre in capitoli lontani non è una testata.
    """
    pages = sorted({o.page for o in group if abs(o.top - occ.top) <= POSITION_TOLERANCE})
    near = [
        p
        for i, p in enumerate(pages)
        if (i > 0 and p - pages[i - 1] <= MAX_PAGE_GAP)
        or (i + 1 < len(pages) and pages[i + 1] - p <= MAX_PAGE_GAP)
    ]
    return occ.page in near and len(near) >= min_pages


def find_running_text_refs(doc: DoclingDocument) -> set[str]:
    """`self_ref` degli elementi del corpo che sono in realtà testate, piè di pagina o numeri di pagina."""
    occurrences = _margin_occurrences(doc)
    min_pages = MIN_PAGES if len(doc.pages) > MIN_PAGES else 2

    by_text: dict[str, list[_Occurrence]] = defaultdict(list)
    # Testate con numero di pagina ("Pag. 3", "Pag. 4"): stesso testo e numero che avanza con la pagina
    by_progress: dict[tuple[str, int], list[_Occurrence]] = defaultdict(list)
    for occ in occurrences:
        by_text[occ.text].append(occ)
        masked = re.sub(r"\d+", "#", occ.text)
        for number in {int(n) for n in re.findall(r"\d{1,4}", occ.text)}:
            by_progress[masked, number - occ.page].append(occ)

    refs: set[str] = set()
    for occ in occurrences:
        if occ.is_furniture:
            continue
        masked = re.sub(r"\d+", "#", occ.text)
        # "Capitolo 1", "Capitolo 2", ... su pagine consecutive sono titoli, non testate numerate
        is_heading = occ.label in {DocItemLabel.SECTION_HEADER, DocItemLabel.TITLE}
        numbers = set() if is_heading else {int(n) for n in re.findall(r"\d{1,4}", occ.text)}
        same_text = by_text[occ.text]
        if (
            is_page_number(occ.text)
            or any(o.is_furniture for o in same_text)
            or _repeats_nearby(occ, same_text, min_pages)
            or any(
                _repeats_nearby(occ, by_progress[masked, n - occ.page], MIN_PAGES) for n in numbers
            )
        ):
            refs.add(occ.ref)
    return refs
