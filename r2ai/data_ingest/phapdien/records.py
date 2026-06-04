"""Canonical article and retrieval-unit records."""

from __future__ import annotations

from typing import Any

from r2ai.data_ingest.phapdien.chunking import split_long_text
from r2ai.data_ingest.phapdien.citations import extract_citation_candidates
from r2ai.data_ingest.phapdien.constants import CONTENT_PREVIEW_CHARS
from r2ai.data_ingest.phapdien.text import (
    extract_article_no,
    labeled_line,
    normalize_text,
    stable_hash,
    stable_point_id,
)


def build_retrieval_text(article: dict[str, Any]) -> str:
    fields = [
        labeled_line("Chủ đề", article.get("topic_title", "")),
        labeled_line("Đề mục", article.get("subject_title", "")),
        normalize_text(article.get("chapter_title", "")),
        labeled_line("Điều pháp điển", article.get("article_title", "")),
        labeled_line("Nguồn", article.get("source_note_text", "")),
        labeled_line("Nội dung", article.get("content_text", "")),
        labeled_line("Liên quan", article.get("related_note_text", "")),
    ]
    return "\n".join(part for part in fields if part)


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


def make_retrieval_units(article: dict[str, Any]) -> list[dict[str, Any]]:
    content = article["content_text"]
    if not content and not (article["article_title"] or article["source_note_text"]):
        return []
    chunks = split_long_text(content) or [article["retrieval_text"]]
    units = []
    for index, chunk_text in enumerate(chunks):
        chunk_id = f"{article['canonical_article_id']}:chunk:{index:03d}"
        unit_text = "\n".join(
            part
            for part in [
                labeled_line("Chủ đề", article["topic_title"]),
                labeled_line("Đề mục", article["subject_title"]),
                article["chapter_title"],
                labeled_line("Điều pháp điển", article["article_title"]),
                labeled_line("Nguồn", article["source_note_text"]),
                labeled_line("Nội dung", chunk_text),
                labeled_line("Liên quan", article["related_note_text"]),
            ]
            if part
        )
        units.append(
            {
                "point_id": stable_point_id(chunk_id),
                "chunk_id": chunk_id,
                "canonical_article_id": article["canonical_article_id"],
                "chunk_index": index,
                "chunk_count": len(chunks),
                "retrieval_text": unit_text,
                "content_preview": chunk_text[:CONTENT_PREVIEW_CHARS],
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
