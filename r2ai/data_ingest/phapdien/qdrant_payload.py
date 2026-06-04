"""Qdrant payload preview records for Phase 2 indexing."""

from __future__ import annotations

from typing import Any

from r2ai.indexing.config import DEFAULT_DENSE_MODEL, DEFAULT_SPARSE_MODEL


def make_qdrant_preview(unit: dict[str, Any]) -> dict[str, Any]:
    payload = {
        key: unit[key]
        for key in [
            "canonical_article_id",
            "chunk_id",
            "chunk_index",
            "chunk_count",
            "content_tree_path",
            "chunk_method",
            "article_title",
            "article_no_normalized",
            "chapter_title",
            "topic_number",
            "topic_title",
            "subject_title",
            "source_note_text",
            "source_url",
            "related_note_text",
            "source_law_id_candidates",
            "source_article_no_candidates",
            "source_doc_title_candidates",
            "citation_confidence",
            "content_preview",
            "retrieval_text",
        ]
    }
    return {
        "id": unit["point_id"],
        "payload": payload,
        "vectors": {
            "dense": {"model": DEFAULT_DENSE_MODEL, "source": "local", "status": "not_embedded"},
            "sparse": {"model": DEFAULT_SPARSE_MODEL, "source": "qdrant", "field": "retrieval_text"},
        },
    }
