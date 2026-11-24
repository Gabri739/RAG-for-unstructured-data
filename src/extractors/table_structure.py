"""Analisi della struttura di una tabella (header, righe-sezione, valori) senza parole chiave.

Le regole guardano solo la forma delle celle (numeri, testo, celle unite, vuote), quindi non
dipendono né dalla lingua né dal tipo di documento. Principio: nel dubbio si lascia la tabella
come l'ha estratta Docling, perché una correzione sbagliata è peggio di nessuna correzione.
"""

from __future__ import annotations

import re
from collections import Counter

MAX_HEADER_ROWS: int = 4

_YEAR = r"(?:19|20)\d{2}"
# Periodi usati come intestazione: 2023, 2023*, FY2023, Q1 2023, 2022/23, 2022-2023
_PERIOD = rf"(?:[A-Za-z]{{1,4}}\d?[\s.\-]?)?{_YEAR}(?:\s?[/–\-]\s?(?:{_YEAR}|\d{{2}}))?"
_PERIOD_RE = re.compile(rf"^{_PERIOD}\W{{0,2}}$")
_NUMBER = r"[\-–+(]?\s?(?:[€$£]\s?)?(?:\d{1,3}(?:\s\d{3})+(?:[.,]\d+)?|\d[\d.,'’]*)"
_NUMERIC_RE = re.compile(rf"^{_NUMBER}\s?(?:%|‰|[€$£]|[A-Za-zµ°/²³]{{1,4}})?\)?\*{{0,2}}$")
_PLAIN_INTEGER_RE = re.compile(r"^[\-–+]?\d+\*{0,2}$")
_PERIOD_AND_VALUE_RE = re.compile(rf"^({_PERIOD}\*{{0,2}})\s+({_NUMBER}\s?[%€$£]?\*{{0,2}})$")
_LABEL_AND_ROW_RE = re.compile(r"^(.+?[a-zà-ÿ)])\s+([A-ZÀ-Ý].+)$")
_BULLET_RE = re.compile(r"^[-–—•●○▪*]\s*\S")
_DASHES = {"-", "–", "—"}


def is_period(text: str) -> bool:
    """True per anni e periodi (es. `2023`, `FY2023`, `2022/23`): sono intestazioni, non valori."""
    return bool(_PERIOD_RE.match(text.strip()))


def is_numeric_value(text: str) -> bool:
    """True per numeri, percentuali, importi e misure con unità; i periodi non contano."""
    t = text.strip()
    return bool(_NUMERIC_RE.match(t)) and not is_period(t)


def is_unambiguous_value(text: str) -> bool:
    """True per valori che non possono essere un'intestazione (`1,10%`, `12.5`, `3 mm`, `€ 5`).

    Un intero semplice (`36`) è ambiguo: può essere un valore o il nome di una colonna.
    """
    t = text.strip()
    return is_numeric_value(t) and not _PLAIN_INTEGER_RE.match(t)


def _kind(text: str) -> str:
    t = text.strip()
    if not t:
        return "empty"
    if t in _DASHES or is_numeric_value(t):
        return "value"
    return "text"


def is_span_row(row: list[str]) -> bool:
    """True se ogni cella piena è ripetuta nella cella accanto (celle unite) e non ci sono valori."""
    if len(row) < 2 or not any(c.strip() for c in row):
        return False
    if any(_kind(c) == "value" for c in row):
        return False
    for j, c in enumerate(row):
        if not c:
            continue
        left = j > 0 and row[j - 1] == c
        right = j + 1 < len(row) and row[j + 1] == c
        if not (left or right):
            return False
    return True


def is_first_only_row(row: list[str]) -> bool:
    """True se solo la prima cella della riga contiene testo."""
    return len(row) > 1 and bool(row[0].strip()) and not any(c.strip() for c in row[1:])


def is_banner_row(row: list[str]) -> bool:
    """True per una riga che contiene un unico testo a tutta larghezza."""
    non_empty = [c for c in row if c]
    return is_first_only_row(row) or (len(non_empty) > 1 and len(set(non_empty)) == 1)


def section_style(text: str) -> str:
    """Stile tipografico di una riga-sezione: `enum` ("1) ..."), `caps` (maiuscola) o `plain`."""
    t = text.strip()
    if re.match(r"^(?:\d+|[A-Za-z]|[IVX]+)[).]\s", t):
        return "enum"
    return "caps" if t.isupper() else "plain"


def is_section_row(row: list[str]) -> bool:
    """True per righe-sezione: celle unite, oppure sola prima cella con titolo numerato/maiuscolo."""
    return is_span_row(row) or (
        is_first_only_row(row) and section_style(row[0]) in {"enum", "caps"}
    )


def blank_repeated_cells(row: list[str]) -> list[str]:
    """Svuota le ripetizioni adiacenti di una riga-sezione (`A | A | B | B` -> `A | | B | `)."""
    return [c if j == 0 or row[j - 1] != c else "" for j, c in enumerate(row)]


def _column_kinds(rows: list[list[str]], n_cols: int) -> list[str]:
    """Tipo prevalente (`value`, `text` o `empty`) di ogni colonna, ignorando le righe-sezione."""
    body = [r for r in rows if not is_section_row(r)]
    kinds: list[str] = []
    for j in range(n_cols):
        counts = Counter(_kind(r[j]) for r in body if j < len(r))
        counts.pop("empty", None)
        kinds.append(counts.most_common(1)[0][0] if counts else "empty")
    return kinds


def has_header_contrast(row: list[str], rows_below: list[list[str]]) -> bool:
    """True se la riga contiene solo testo e sta sopra almeno una colonna di valori.

    È il segnale strutturale più affidabile di un'intestazione, valido in ogni lingua.
    """
    if any(_kind(c) == "value" for c in row) or sum(1 for c in row if c.strip()) < 2:
        return False
    kinds = _column_kinds(rows_below, len(row))
    return any(c.strip() and kinds[j] == "value" for j, c in enumerate(row) if j > 0)


def looks_like_data_row(row: list[str], rows_below: list[list[str]]) -> bool:
    """True se la riga ha la stessa forma delle righe sottostanti (stessi tipi per colonna)."""
    if not row or not row[0].strip():
        return False
    kinds = _column_kinds(rows_below, len(row))
    compared = values = 0
    for j, c in enumerate(row):
        if j == 0 or _kind(c) == "empty" or kinds[j] == "empty":
            continue
        compared += 1
        if _kind(c) != kinds[j]:
            return False
        values += _kind(c) == "value"
    return compared > 0 and values > 0


def split_merged_header(
    rows: list[list[str]], flags: list[bool]
) -> tuple[list[list[str]], list[bool], list[int]]:
    """Separa header e prima riga dati finiti nelle stesse celle (es. `"2021 613"`, `"2022 624"`).

    Restituisce `(righe, flag_header, indice_di_riga_originale)`.
    """
    out_rows: list[list[str]] = []
    out_flags: list[bool] = []
    origin: list[int] = []
    for i, (row, flag) in enumerate(zip(rows, flags)):
        matches = [_PERIOD_AND_VALUE_RE.match(c) for c in row[1:]]
        if flag and len(row) >= 3 and all(matches):
            label = _LABEL_AND_ROW_RE.match(row[0])
            hdr_label, row_label = label.groups() if label else ("", row[0])
            out_rows.append([hdr_label] + [m.group(1) for m in matches if m])
            out_rows.append([row_label] + [m.group(2) for m in matches if m])
            out_flags += [True, False]
            origin += [i, i]
        else:
            out_rows.append(row)
            out_flags.append(flag)
            origin.append(i)
    return out_rows, out_flags, origin


def count_header_rows(rows: list[list[str]], flags: list[bool]) -> int:
    """Numero di righe iniziali che sono intestazione (compreso un eventuale titolo a tutta larghezza).

    Parte dai flag di Docling e li corregge solo quando la struttura è inequivocabile:
      - toglie una riga marcata come header se ha la stessa forma dei dati sotto e contiene valori
        che non possono essere nomi di colonna (mai la prima riga della tabella);
      - aggiunge una riga non marcata se è testo sopra colonne di valori e segue un titolo a tutta
        larghezza, oppure se completa un'intestazione su più livelli (celle unite o prima cella vuota).
    """
    n_rows = len(rows)
    n = 0
    while n < n_rows and flags[n] and n < MAX_HEADER_ROWS:
        if n > 0 and any(is_unambiguous_value(c) for c in rows[n][1:]):
            if looks_like_data_row(rows[n], rows[n + 1 :]):
                break
        n += 1

    while 0 < n < min(n_rows - 1, MAX_HEADER_ROWS):
        row = rows[n]
        below = rows[n + 1 :]
        after_banner = n == 1 and is_banner_row(rows[0])
        multi_level = not row[0].strip() or any(
            row[j] and row[j] == row[j - 1] for j in range(1, len(row))
        )
        if is_section_row(row) or not (after_banner or multi_level):
            break
        if not has_header_contrast(row, below):
            break
        n += 1

    if n == 0 and n_rows >= 3 and is_banner_row(rows[0]) and not is_section_row(rows[1]):
        if has_header_contrast(rows[1], rows[2:]):
            n = 2
    elif n == 0 and n_rows >= 2 and not is_banner_row(rows[0]):
        # La prima di più righe-sezione non è l'intestazione della tabella
        is_section = is_section_row(rows[0]) and any(is_section_row(r) for r in rows[1:])
        if has_header_contrast(rows[0], rows[1:]) and not is_section:
            n = 1
    return n


# Livelli gerarchici delle righe (più basso = più generale), usati per dividere le tabelle lunghe
LEVEL_MAIN_SECTION, LEVEL_SECTION, LEVEL_ROW, LEVEL_GROUP, LEVEL_DETAIL = range(5)


def _repeated_labels(rows: list[list[str]], section_flags: list[bool]) -> dict[str, int]:
    """Etichette della prima colonna che si ripetono in punti diversi della tabella.

    In una tabella a gruppi le righe di dettaglio si ripetono sotto ogni riga madre ("Di cui uomini",
    "of which men", ...), mentre le righe madri hanno etichette uniche. Il risultato è vuoto se lo
    schema non è chiaro (es. un elenco in cui un nome ricorre per caso).
    """
    labels = [
        None if sec or not r or not r[0].strip() else r[0].strip().lower()
        for r, sec in zip(rows, section_flags)
    ]
    runs: Counter[str] = Counter()
    for i, lab in enumerate(labels):
        if lab is not None and (i == 0 or labels[i - 1] != lab):
            runs[lab] += 1
    repeated = {lab: n for lab, n in runs.items() if n >= 2}
    if not repeated:
        return {}

    first = next((lab for lab in labels if lab is not None), None)
    if first in repeated:
        return {}
    groups: list[set[str]] = []
    current: set[str] | None = None
    for lab in labels:
        if lab is None:
            current = None
        elif lab not in repeated:
            current = set()
            groups.append(current)
        elif current is not None:
            current.add(lab)
    with_children = [g for g in groups if g]
    if len(with_children) < 2:
        return {}
    similar = sum(
        1
        for a, b in zip(with_children, with_children[1:])
        if len(a & b) >= 0.5 * min(len(a), len(b))
    )
    return repeated if similar >= 0.5 * (len(with_children) - 1) else {}


def row_levels(rows: list[list[str]], section_flags: list[bool]) -> list[int]:
    """Livello di ogni riga: sezione, riga normale/madre, sottogruppo, dettaglio."""
    repeated = _repeated_labels(rows, section_flags)
    max_freq = max(repeated.values(), default=0)
    levels: list[int] = []
    for row, sec in zip(rows, section_flags):
        label = row[0].strip() if row else ""
        if sec:
            levels.append(LEVEL_MAIN_SECTION if section_style(label) == "caps" else LEVEL_SECTION)
        elif _BULLET_RE.match(label):
            levels.append(LEVEL_DETAIL)
        elif label.lower() in repeated:
            # Le etichette più frequenti sono le più interne (es. genere dentro fascia d'età)
            freq = repeated[label.lower()]
            levels.append(LEVEL_DETAIL if freq == max_freq else LEVEL_GROUP)
        else:
            levels.append(LEVEL_ROW)
    return levels
