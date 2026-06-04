"""Canonical article and retrieval-unit records."""

from __future__ import annotations

from typing import Any

from r2ai.data_ingest.phapdien.chunking import split_structured_text
from r2ai.data_ingest.phapdien.citations import extract_citation_candidates
from r2ai.data_ingest.phapdien.constants import (
    CONTENT_PREVIEW_CHARS,
    DEFAULT_CHUNK_OVERLAP_TOKENS,
    DEFAULT_MAX_CHUNK_TOKENS,
)
from r2ai.data_ingest.phapdien.text import (
    extract_article_no,
    extract_source_article_title,
    extract_source_note_context,
    normalize_text,
    prefer_source_doc_title,
    source_title_has_name,
    stable_hash,
    stable_point_id,
    strip_related_content_note,
)


def build_source_title_map(articles: list[dict[str, Any]]) -> dict[str, str]:
    title_by_law_code: dict[str, str] = {}
    for article in articles:
        context = extract_source_note_context(article.get("source_note_text", ""))
        law_code = context["source_law_code"]
        source_title = context["source_doc_title"]
        if not law_code or not source_title:
            continue
        current_title = title_by_law_code.get(law_code, "")
        if source_title_has_name(source_title, law_code) and len(source_title) > len(current_title):
            title_by_law_code[law_code] = source_title
        elif not current_title:
            title_by_law_code[law_code] = source_title
    return title_by_law_code


def build_retrieval_text(
    article: dict[str, Any],
    source_title_by_law_code: dict[str, str] | None = None,
) -> str:
    source_context = extract_source_note_context(article.get("source_note_text", ""))
    law_code = source_context["source_law_code"]
    mapped_title = (source_title_by_law_code or {}).get(law_code, "")
    source_doc_title = prefer_source_doc_title(source_context["source_doc_title"], mapped_title, law_code)
    header = normalize_text(
        " ".join(
            part
            for part in [
                source_doc_title,
                source_context["source_article_no"],
                extract_source_article_title(article.get("article_title", "")),
            ]
            if part
        )
    )
    content = strip_related_content_note(article.get("content_text", ""))
    if header and content:
        return f"{header}:\n{content}"
    return header or content


def canonical_hash_basis(row: dict[str, Any]) -> str:
    article_anchor = normalize_text(row.get("article_anchor"))
    article_title = normalize_text(row.get("article_title"))
    return "|".join(
        [
            article_anchor,
            normalize_text(row.get("topic_id")),
            normalize_text(row.get("subject_id")),
            normalize_text(row.get("topic_number")),
            normalize_text(row.get("subject_number")),
            article_title,
        ]
    )


def canonicalize_article(row: dict[str, Any], disambiguator: int = 0) -> dict[str, Any]:
    article_anchor = normalize_text(row.get("article_anchor"))
    article_title = normalize_text(row.get("article_title"))
    article_no_raw, article_no_normalized = extract_article_no(article_title)
    hash_basis = canonical_hash_basis(row)
    if disambiguator:
        hash_basis = f"{hash_basis}|duplicate:{disambiguator}"

    article = {
        "canonical_article_id": stable_hash(hash_basis, "art"),
        "article_anchor": article_anchor,
        "article_title": article_title,
        "article_no_raw": article_no_raw,
        "article_no_normalized": article_no_normalized,
        "topic_id": normalize_text(row.get("topic_id")),
        "topic_number": row.get("topic_number"),
        "topic_title": normalize_text(row.get("topic_title")),
        "subject_id": normalize_text(row.get("subject_id")),
        "subject_number": row.get("subject_number"),
        "subject_title": normalize_text(row.get("subject_title")),
        "chapter_title": normalize_text(row.get("chapter_title")),
        "source_note_text": normalize_text(row.get("source_note_text")),
        "related_note_text": normalize_text(row.get("related_note_text")),
        "source_url": normalize_text(row.get("source_url")),
        "content_text": normalize_text(row.get("content_text")),
        "content_char_len": int(row.get("content_char_len") or len(normalize_text(row.get("content_text")))),
        "content_word_count": int(row.get("content_word_count") or len(normalize_text(row.get("content_text")).split())),
    }
    article.update(extract_citation_candidates(article["source_note_text"], row.get("source_links")))
    article["retrieval_text"] = build_retrieval_text(article)
    return article


def make_retrieval_units(
    article: dict[str, Any],
    max_chunk_tokens: int = DEFAULT_MAX_CHUNK_TOKENS,
    chunk_overlap_tokens: int = DEFAULT_CHUNK_OVERLAP_TOKENS,
    source_title_by_law_code: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    content = strip_related_content_note(article["content_text"])
    if not content and not (article["article_title"] or article["source_note_text"]):
        return []
    chunks = split_structured_text(
        content,
        max_tokens=max_chunk_tokens,
        overlap_tokens=chunk_overlap_tokens,
    )
    if not chunks:
        chunks = split_structured_text(
            article["retrieval_text"],
            max_tokens=max_chunk_tokens,
            overlap_tokens=chunk_overlap_tokens,
        )
    units = []
    for index, chunk in enumerate(chunks):
        chunk_id = f"{article['canonical_article_id']}:chunk:{index:03d}"
        unit_text = build_retrieval_text(
            {**article, "content_text": chunk.text},
            source_title_by_law_code=source_title_by_law_code,
        )
        units.append(
            {
                "point_id": stable_point_id(chunk_id),
                "chunk_id": chunk_id,
                "canonical_article_id": article["canonical_article_id"],
                "chunk_index": index,
                "chunk_count": len(chunks),
                "content_tree_path": chunk.path,
                "chunk_method": chunk.method,
                "retrieval_text": unit_text,
                "content_preview": chunk.text[:CONTENT_PREVIEW_CHARS],
                "article_title": article["article_title"],
                "article_no_normalized": article["article_no_normalized"],
                "chapter_title": article["chapter_title"],
                "topic_number": article["topic_number"],
                "topic_title": article["topic_title"],
                "subject_title": article["subject_title"],
                "source_note_text": article["source_note_text"],
                "source_url": article["source_url"],
                "related_note_text": article["related_note_text"],
                "source_law_id_candidates": article["source_law_id_candidates"],
                "source_article_no_candidates": article["source_article_no_candidates"],
                "source_doc_title_candidates": article["source_doc_title_candidates"],
                "citation_confidence": article["citation_confidence"],
            }
        )
    return units
