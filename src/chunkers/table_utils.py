"""Utilities for detecting and handling Markdown tables."""

import re
from dataclasses import dataclass
from typing import List, Optional, Tuple


@dataclass
class TableBlock:
    """Represents a detected table in Markdown."""
    headers: List[str]
    rows: List[List[str]]
    raw_lines: List[str]
    start_line: int
    end_line: int
    caption: Optional[str] = None  # Text immediately before table

    @property
    def is_valid(self) -> bool:
        """Check if table has valid structure."""
        return len(self.headers) > 0 and len(self.rows) > 0

    def to_markdown(self) -> str:
        """Convert back to Markdown format."""
        if not self.headers:
            return ""
        header_line = "| " + " | ".join(self.headers) + " |"
        separator = "|" + "|".join([" --- " for _ in self.headers]) + "|"
        rows = ["| " + " | ".join(row) + " |" for row in self.rows]
        return "\n".join([header_line, separator] + rows)


def parse_table(lines: List[str], start_idx: int) -> Optional[TableBlock]:
    """Parse a table starting at given line index.

    Returns TableBlock if valid table found, None otherwise.
    """
    table_lines = []
    end_idx = start_idx

    for i in range(start_idx, len(lines)):
        line = lines[i].strip()
        if line.startswith("|"):
            table_lines.append(line)
            end_idx = i
        elif line == "" and table_lines:
            # Empty line after table - end of table
            break
        elif table_lines:
            # Non-table line after table started - stop
            break

    if len(table_lines) < 2:
        return None

    # Parse header
    headers = [h.strip() for h in table_lines[0].split("|") if h.strip()]
    if not headers:
        return None

    # Skip separator line (line 1)
    # Parse data rows
    rows = []
    for line in table_lines[2:]:
        cells = [c.strip() for c in line.split("|") if c.strip() or c == ""]
        # Pad with empty strings if needed
        while len(cells) < len(headers):
            cells.append("")
        if cells:
            rows.append(cells[:len(headers)])

    return TableBlock(
        headers=headers,
        rows=rows,
        raw_lines=table_lines,
        start_line=start_idx,
        end_line=end_idx
    )


def extract_caption(lines: List[str], table_start: int) -> Optional[str]:
    """Extract caption text immediately before a table."""
    if table_start == 0:
        return None

    prev_line = lines[table_start - 1].strip()
    if prev_line and not prev_line.startswith("|") and not prev_line.startswith("#"):
        return prev_line
    return None


def find_all_tables(content: str) -> List[TableBlock]:
    """Find all tables in Markdown content."""
    lines = content.split("\n")
    tables = []
    i = 0

    while i < len(lines):
        line = lines[i].strip()
        if line.startswith("|"):
            table = parse_table(lines, i)
            if table:
                table.caption = extract_caption(lines, i)
                tables.append(table)
                i = table.end_line + 1
                continue
        i += 1

    return tables


def are_tables_continuation(table1: TableBlock, table2: TableBlock) -> bool:
    """Check if table2 is a continuation of table1.

    Heuristics:
    - Same number of columns
    - Headers match or table2 has generic headers (A, B, C...) or empty
    - Similar data patterns
    """
    # Must have same number of columns
    if len(table1.headers) != len(table2.headers):
        return False

    # Check if headers match exactly
    if table1.headers == table2.headers:
        return True

    # Check if table2 headers are generic/placeholder
    generic_patterns = [
        r"^[A-Z]$",  # Single letters
        r"^\d+$",    # Numbers only
        r"^[\-\s]+$" # Dashes/spaces
    ]
    is_generic = all(
        any(re.match(p, h) for p in generic_patterns)
        for h in table2.headers
    )

    if is_generic:
        return True

    # Check similarity: at least 50% headers match
    matching = sum(1 for h1, h2 in zip(table1.headers, table2.headers) if h1 == h2)
    return matching >= len(table1.headers) * 0.5


def merge_tables(table1: TableBlock, table2: TableBlock) -> TableBlock:
    """Merge two table blocks into one."""
    return TableBlock(
        headers=table1.headers,  # Keep original headers
        rows=table1.rows + table2.rows,
        raw_lines=table1.raw_lines + table2.raw_lines,
        start_line=table1.start_line,
        end_line=table2.end_line,
        caption=table1.caption or table2.caption
    )


def split_large_table(table: TableBlock, max_rows: int = 50) -> List[TableBlock]:
    """Split a large table into smaller chunks with repeated headers."""
    if len(table.rows) <= max_rows:
        return [table]

    chunks = []
    for i in range(0, len(table.rows), max_rows):
        chunk_rows = table.rows[i:i + max_rows]
        chunk = TableBlock(
            headers=table.headers,
            rows=chunk_rows,
            raw_lines=[],  # Will be regenerated
            start_line=table.start_line,
            end_line=table.end_line,
            caption=f"{table.caption} (Part {i // max_rows + 1})" if table.caption else f"Part {i // max_rows + 1}"
        )
        chunks.append(chunk)

    return chunks
