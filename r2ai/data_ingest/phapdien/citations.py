"""Citation extraction from phapdien source notes."""

from __future__ import annotations

import json
from typing import Any, Iterable

from r2ai.data_ingest.phapdien.constants import ARTICLE_PATTERN, LAW_ID_PATTERN
from r2ai.data_ingest.phapdien.text import normalize_text


def extract_source_links(value: Any) -> list[dict[str, str]]:
    if not value:
        return []
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return []
    links: list[dict[str, str]] = []
    if isinstance(value, Iterable):
        for item in value:
            if isinstance(item, dict):
                text = normalize_text(item.get("text"))
                href = normalize_text(item.get("href"))
                if text or href:
                    links.append({"text": text, "href": href})
    return links


def extract_citation_candidates(
    source_note_text: str,
    source_links: Any,
) -> dict[str, Any]:
    instrument_text = normalize_text(source_note_text)
    law_ids = sorted(set(match.group(0).upper() for match in LAW_ID_PATTERN.finditer(instrument_text)))
    source_article_numbers = sorted(
        set(f"Điều {match.group(1)}" for match in ARTICLE_PATTERN.finditer(instrument_text))
    )
    link_titles = [link["text"] for link in extract_source_links(source_links) if link.get("text")]

    title_candidates: list[str] = []
    if link_titles:
        title_candidates.extend(link_titles)
    for law_id in law_ids:
        idx = instrument_text.upper().find(law_id)
        if idx >= 0:
            start = max(0, idx - 120)
            end = min(len(instrument_text), idx + len(law_id) + 220)
            title_candidates.append(instrument_text[start:end].strip(" .;,:-"))

    deduped_titles = []
    seen_titles = set()
    for title in title_candidates:
        normalized = normalize_text(title)
        key = normalized.lower()
        if normalized and key not in seen_titles:
            seen_titles.add(key)
            deduped_titles.append(normalized)

    if law_ids and deduped_titles:
        confidence = "high"
    elif instrument_text:
        confidence = "medium"
    else:
        confidence = "low"

    return {
        "source_instrument_text": instrument_text,
        "source_law_id_candidates": law_ids,
        "source_article_no_candidates": source_article_numbers,
        "source_doc_title_candidates": deduped_titles,
        "citation_confidence": confidence,
    }
