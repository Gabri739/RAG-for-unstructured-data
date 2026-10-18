"""Smart Markdown chunker using LangChain with table-aware logic."""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from langchain_text_splitters import MarkdownHeaderTextSplitter

from .table_utils import (
    find_all_tables,
    are_tables_continuation,
    merge_tables,
    split_large_table,
    TableBlock
)


@dataclass
class Chunk:
    """Represents a document chunk with metadata."""
    content: str
    headers: List[str] = field(default_factory=list)
    chunk_type: str = "text"  # "text" or "table"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.metadata:
            self.metadata = {}


class MarkdownChunker:
    """Chunk Markdown documents with smart table handling.

    Uses LangChain's MarkdownHeaderTextSplitter as base, with additional
    logic to handle multi-page tables and preserve context.
    """

    def __init__(
        self,
        headers_to_split_on: Optional[List[str]] = None,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        max_table_rows: int = 50,
        preserve_tables: bool = True
    ):
        """Initialize chunker.

        Args:
            headers_to_split_on: List of header markers to split on (e.g., ["#", "##"])
            chunk_size: Maximum chunk size in characters
            chunk_overlap: Overlap between chunks
            max_table_rows: Max rows before splitting a table
            preserve_tables: Whether to keep tables as atomic units
        """
        # LangChain expects list of tuples: [(marker, name), ...]
        raw_headers = headers_to_split_on or ["#", "##", "###"]
        self.headers_to_split_on = [(h, f"Header {i+1}") for i, h in enumerate(raw_headers)]
        self._raw_headers = raw_headers  # Keep original for regex matching
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.max_table_rows = max_table_rows
        self.preserve_tables = preserve_tables

        self._header_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=self.headers_to_split_on,
            strip_headers=False  # Keep headers in content
        )

    def _extract_header_stack(self, content: str) -> List[str]:
        """Extract current header hierarchy from content."""
        headers = []
        for line in content.split("\n"):
            for marker in self._raw_headers:
                if line.strip().startswith(marker + " "):
                    headers.append(line.strip())
                    break
        return headers

    def _merge_continuation_tables(self, tables: List[TableBlock]) -> List[TableBlock]:
        """Merge tables that are continuations of each other."""
        if not tables:
            return []

        merged = [tables[0]]
        for current in tables[1:]:
            last = merged[-1]
            if are_tables_continuation(last, current):
                merged[-1] = merge_tables(last, current)
            else:
                merged.append(current)
        return merged

    def _create_table_chunk(self, table: TableBlock, header_stack: List[str]) -> Chunk:
        """Create a chunk from a table block."""
        table_md = table.to_markdown()

        # Add caption as context if present
        if table.caption:
            content = f"{table.caption}\n\n{table_md}"
        else:
            content = table_md

        # Add header context as prefix
        if header_stack:
            header_context = "\n".join(header_stack)
            content = f"{header_context}\n\n{content}"

        return Chunk(
            content=content,
            headers=header_stack.copy(),
            chunk_type="table",
            metadata={
                "row_count": len(table.rows),
                "column_count": len(table.headers),
                "has_caption": table.caption is not None,
                "is_continuation": False  # Would need page info to determine
            }
        )

    def _process_tables_in_chunk(
        self,
        content: str,
        header_stack: List[str]
    ) -> List[Chunk]:
        """Process content containing tables, extracting them as separate chunks."""
        tables = find_all_tables(content)

        if not tables:
            # No tables, return original as text chunk
            return [Chunk(
                content=content,
                headers=header_stack.copy(),
                chunk_type="text"
            )]

        # Merge continuation tables
        merged_tables = self._merge_continuation_tables(tables)

        chunks = []
        last_end = 0
        lines = content.split("\n")

        for table in merged_tables:
            # Text before this table
            text_lines = lines[last_end:table.start_line]
            text_content = "\n".join(line for line in text_lines if line.strip())

            if text_content.strip():
                chunks.append(Chunk(
                    content=text_content,
                    headers=header_stack.copy(),
                    chunk_type="text"
                ))

            # Split large tables and create chunks
            table_chunks = split_large_table(table, self.max_table_rows)
            for tc in table_chunks:
                chunks.append(self._create_table_chunk(tc, header_stack))

            last_end = table.end_line + 1

        # Text after last table
        text_lines = lines[last_end:]
        text_content = "\n".join(line for line in text_lines if line.strip())
        if text_content.strip():
            chunks.append(Chunk(
                content=text_content,
                headers=header_stack.copy(),
                chunk_type="text"
            ))

        return chunks

    def chunk(self, markdown_content: str) -> List[Chunk]:
        """Process Markdown content and return chunks.

        Args:
            markdown_content: Raw Markdown string from OCR/Vision

        Returns:
            List of Chunk objects with metadata
        """
        # Step 1: Split by headers using LangChain
        lc_documents = self._header_splitter.split_text(markdown_content)

        all_chunks = []

        for doc in lc_documents:
            content = doc.page_content
            header_stack = self._extract_header_stack(content)

            if not self.preserve_tables:
                # Simple mode: just use header split
                all_chunks.append(Chunk(
                    content=content,
                    headers=header_stack,
                    chunk_type="text",
                    metadata={"source": "header_split"}
                ))
                continue

            # Step 2: Process tables in each header chunk
            chunks = self._process_tables_in_chunk(content, header_stack)

            # Step 3: Check size limits and split text chunks if needed
            for chunk in chunks:
                if chunk.chunk_type == "text" and len(chunk.content) > self.chunk_size:
                    # Split large text chunks recursively
                    sub_chunks = self._split_text_chunk(chunk)
                    all_chunks.extend(sub_chunks)
                else:
                    all_chunks.append(chunk)

        return all_chunks

    def _split_text_chunk(self, chunk: Chunk) -> List[Chunk]:
        """Split a large text chunk into smaller pieces with overlap."""
        words = chunk.content.split()
        chunks = []

        # Roughly estimate: 1 word ≈ 5 characters on average
        words_per_chunk = self.chunk_size // 5
        overlap_words = self.chunk_overlap // 5

        # Ensure we always make progress (step = chunk_size - overlap)
        step = max(1, words_per_chunk - overlap_words)

        start = 0
        part_num = 1

        while start < len(words):
            end = min(start + words_per_chunk, len(words))
            chunk_text = " ".join(words[start:end])

            # Add continuation markers for split chunks
            if part_num > 1:
                chunk_text = f"[...continued]\n{chunk_text}"
            if end < len(words):
                chunk_text = f"{chunk_text}\n[continues...]"

            chunks.append(Chunk(
                content=chunk_text,
                headers=chunk.headers.copy(),
                chunk_type="text",
                metadata={**chunk.metadata, "part": part_num, "total_parts": None}
            ))

            start += step
            part_num += 1

        # Update total_parts
        for c in chunks:
            c.metadata["total_parts"] = len(chunks)

        return chunks

    def chunk_with_context(
        self,
        markdown_content: str,
        add_contextual_headers: bool = True
    ) -> List[Chunk]:
        """Chunk with optional contextual header enhancement.

        Similar to chunk() but can add explanatory context to headers.
        """
        chunks = self.chunk(markdown_content)

        if not add_contextual_headers:
            return chunks

        # Optional: Add summary context to each chunk
        # This would require an LLM call - left as extension point
        return chunks
