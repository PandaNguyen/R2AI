"""Chunking for long phapdien articles."""

from __future__ import annotations

from r2ai.data_ingest.phapdien.constants import (
    CHUNK_OVERLAP_CHARS,
    LONG_ARTICLE_CHARS,
    SECTION_BOUNDARY_PATTERN,
    SENTENCE_BOUNDARY_PATTERN,
    TARGET_CHUNK_CHARS,
)
from r2ai.data_ingest.phapdien.text import normalize_text


def split_long_text(text: str) -> list[str]:
    text = normalize_text(text)
    if len(text) <= LONG_ARTICLE_CHARS:
        return [text] if text else []

    parts = [part.strip() for part in SECTION_BOUNDARY_PATTERN.split(text) if part.strip()]
    if len(parts) <= 1:
        parts = [part.strip() for part in SENTENCE_BOUNDARY_PATTERN.split(text) if part.strip()]
    if len(parts) <= 1:
        return _hard_split(text)

    chunks: list[str] = []
    current = ""
    for part in parts:
        candidate = f"{current} {part}".strip() if current else part
        if len(candidate) <= TARGET_CHUNK_CHARS or not current:
            current = candidate
            continue
        chunks.append(current)
        overlap = current[-CHUNK_OVERLAP_CHARS:].strip()
        current = f"{overlap} {part}".strip() if overlap else part
    if current:
        chunks.append(current)
    return chunks


def _hard_split(text: str) -> list[str]:
    chunks = []
    step = TARGET_CHUNK_CHARS - CHUNK_OVERLAP_CHARS
    for start in range(0, len(text), step):
        chunk = text[start : start + TARGET_CHUNK_CHARS].strip()
        if chunk:
            chunks.append(chunk)
    return chunks
