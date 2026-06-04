"""Qdrant payload preview records for Phase 2 indexing."""

from __future__ import annotations

from typing import Any


def make_qdrant_preview(unit: dict[str, Any]) -> dict[str, Any]:
    payload = {
        key: unit[key]
        for key in [
            "canonical_article_id",
            "chunk_id",
            "chunk_index",
            "chunk_count",
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
            "dense": {"model": "BAAI/bge-m3", "source": "local", "status": "not_embedded"},
            "sparse": {"model": "qdrant/bm25", "source": "qdrant", "field": "retrieval_text"},
        },
    }
