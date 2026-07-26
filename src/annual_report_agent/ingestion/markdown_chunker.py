from __future__ import annotations

import re
from dataclasses import dataclass

from ..schemas import Chunk

PAGE_MARKER = re.compile(r"^\s*<!--\s*page\s*:\s*(\d+)\s*-->\s*$", re.IGNORECASE)
HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*$")


@dataclass(frozen=True, slots=True)
class MarkdownDocument:
    document_id: str
    company: str
    year: int
    markdown: str


def _split_long_text(text: str, max_chars: int, overlap_chars: int) -> list[str]:
    if len(text) <= max_chars:
        return [text]

    pieces: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + max_chars, len(text))
        if end < len(text):
            boundary = max(
                text.rfind("\n", start, end),
                text.rfind("。", start, end),
                text.rfind("；", start, end),
            )
            if boundary > start + max_chars // 2:
                end = boundary + 1
        piece = text[start:end].strip()
        if piece:
            pieces.append(piece)
        if end >= len(text):
            break
        start = max(end - overlap_chars, start + 1)
    return pieces


def chunk_markdown(
    document: MarkdownDocument,
    *,
    max_chars: int = 900,
    overlap_chars: int = 120,
    min_chars: int = 40,
) -> list[Chunk]:
    """Split Markdown while preserving section and page metadata.

    Page markers use the form ``<!-- page: 12 -->``. Tables and adjacent
    paragraphs remain together until the current section exceeds ``max_chars``.
    """

    if max_chars <= 0:
        raise ValueError("max_chars must be positive")
    if not 0 <= overlap_chars < max_chars:
        raise ValueError("overlap_chars must be between 0 and max_chars")

    current_page: int | None = None
    current_section = "文档开头"
    section_page: int | None = None
    buffer: list[str] = []
    segments: list[tuple[str, int | None, str]] = []

    def flush() -> None:
        nonlocal buffer
        text = "\n".join(buffer).strip()
        if text:
            for piece in _split_long_text(text, max_chars, overlap_chars):
                if len(piece) >= min_chars:
                    segments.append((current_section, section_page, piece))
        buffer = []

    for raw_line in document.markdown.splitlines():
        page_match = PAGE_MARKER.match(raw_line)
        if page_match:
            flush()
            current_page = int(page_match.group(1))
            section_page = current_page
            continue

        heading_match = HEADING.match(raw_line)
        if heading_match:
            flush()
            current_section = heading_match.group(2).strip()
            section_page = current_page
            continue

        if raw_line.strip() or buffer:
            buffer.append(raw_line.rstrip())
    flush()

    return [
        Chunk(
            chunk_id=f"{document.document_id}:{index:04d}",
            document_id=document.document_id,
            company=document.company,
            year=document.year,
            section=section,
            page=page,
            text=text,
        )
        for index, (section, page, text) in enumerate(segments)
    ]
