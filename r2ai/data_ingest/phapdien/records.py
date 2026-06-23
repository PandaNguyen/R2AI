"""Canonical article and retrieval-unit records."""

from __future__ import annotations

import hashlib
from typing import Any

from r2ai.data_ingest.phapdien.chunking import count_tokens, split_structured_text
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
    normalize_competition_doc_title,
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

    retrieval_context = build_retrieval_context(article, source_title_by_law_code=source_title_by_law_code)
    content_budget = max(128, max_chunk_tokens - count_tokens(retrieval_context) - 8)
    chunks = split_structured_text(
        content,
        max_tokens=content_budget,
        overlap_tokens=chunk_overlap_tokens,
    )
    chunk_is_full_retrieval_text = False
    if not chunks:
        chunks = split_structured_text(
            retrieval_context,
            max_tokens=max_chunk_tokens,
            overlap_tokens=chunk_overlap_tokens,
        )
        chunk_is_full_retrieval_text = True

    competition_metadata = build_competition_metadata(
        article,
        source_title_by_law_code=source_title_by_law_code,
    )
    units = []
    for index, chunk in enumerate(chunks):
        chunk_id = f"{article['canonical_article_id']}:chunk:{index:03d}"
        unit_text = (
            chunk.text
            if chunk_is_full_retrieval_text
            else build_retrieval_text(
                {**article, "content_text": chunk.text},
                source_title_by_law_code=source_title_by_law_code,
            )
        )
        point_id = stable_point_id(chunk_id)
        legal_path = [
            value
            for value in [
                competition_metadata["source_document_title_type1"],
                competition_metadata["source_article_no"],
                article["topic_title"],
                article["subject_title"],
                article["chapter_title"],
                article["article_title"],
                *chunk.path,
            ]
            if value
        ]
        units.append(
            {
                "id": point_id,
                "point_id": point_id,
                "chunk_id": chunk_id,
                "dataset": "phapdien-moj-gov-vn",
                "document_id": competition_metadata["source_document_id"],
                "document_number": competition_metadata["source_document_number"],
                "document_title": competition_metadata["source_document_title_type1"],
                "legal_type": competition_metadata["source_document_type"],
                "legal_sectors": article["topic_title"],
                "canonical_article_id": article["canonical_article_id"],
                "chunk_index": index,
                "chunk_count": len(chunks),
                "content_tree_path": chunk.path,
                "chunk_method": chunk.method,
                "chunk_type": chunk.method,
                "node_id": article["article_anchor"],
                "node_type": "phapdien_article",
                "node_label": article["article_title"],
                "node_number": article["article_no_normalized"],
                "node_title": extract_source_article_title(article["article_title"]),
                "legal_path": legal_path,
                "legal_path_text": " > ".join(legal_path),
                "article_no": competition_metadata["source_article_no"],
                "article_title_source": competition_metadata["source_article_title"],
                "clause_no": "",
                "point_no": "",
                "appendix": "",
                "contains_table": False,
                "table_id": "",
                "table_caption": "",
                "table_headers": [],
                "table_row_start": None,
                "table_row_end": None,
                "table_total_rows": None,
                "line_start": None,
                "line_end": None,
                "retrieval_text": unit_text,
                "retrieval_text_sha1": hashlib.sha1(unit_text.encode("utf-8")).hexdigest(),
                "token_estimate": count_tokens(unit_text),
                "content_text": "" if chunk_is_full_retrieval_text else chunk.text,
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
                "phapdien_article_id": article["canonical_article_id"],
                "phapdien_article_title": article["article_title"],
                "phapdien_article_no": article["article_no_normalized"],
                "phapdien_topic_number": article["topic_number"],
                "phapdien_topic_title": article["topic_title"],
                "phapdien_subject_title": article["subject_title"],
                "phapdien_chapter_title": article["chapter_title"],
                **competition_metadata,
            }
        )
    return units


def build_retrieval_context(
    article: dict[str, Any],
    source_title_by_law_code: dict[str, str] | None = None,
) -> str:
    return build_retrieval_text(
        {**article, "content_text": ""},
        source_title_by_law_code=source_title_by_law_code,
    )


def build_competition_metadata(
    article: dict[str, Any],
    source_title_by_law_code: dict[str, str] | None = None,
) -> dict[str, Any]:
    source_context = extract_source_note_context(article.get("source_note_text", ""))
    law_id = source_context["source_law_code"]
    mapped_title = (source_title_by_law_code or {}).get(law_id, "")
    source_doc_title = prefer_source_doc_title(source_context["source_doc_title"], mapped_title, law_id)
    article_no = source_context["source_article_no"] or article.get("article_no_normalized", "")
    article_title = extract_source_article_title(article.get("article_title", ""))
    doc_title_type1 = normalize_competition_doc_title(source_doc_title, law_id, "type1")
    doc_title_type2 = normalize_competition_doc_title(source_doc_title, law_id, "type2")
    source_document_key = "|".join(part for part in [law_id, source_doc_title or doc_title_type1] if part)
    source_document_id = stable_hash(source_document_key, "src") if source_document_key else article["canonical_article_id"]
    return {
        "source_document_id": source_document_id,
        "source_document_number": law_id,
        "source_document_title": source_doc_title,
        "source_document_title_type1": doc_title_type1,
        "source_document_title_type2": doc_title_type2,
        "source_document_type": source_context["source_law_type"],
        "source_article_no": article_no,
        "source_article_title": article_title,
        "competition_law_id": law_id,
        "competition_doc_title_type1": doc_title_type1,
        "competition_doc_title_type2": doc_title_type2,
        "competition_article_no": article_no,
    }
