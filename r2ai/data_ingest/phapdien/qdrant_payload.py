"""Qdrant payload preview records for Phase 2 indexing."""

from __future__ import annotations

from typing import Any

from r2ai.indexing.config import DEFAULT_DENSE_MODEL, DEFAULT_SPARSE_MODEL


def make_qdrant_preview(unit: dict[str, Any]) -> dict[str, Any]:
    point_id = unit.get("id") or unit["point_id"]
    payload = {key: value for key, value in unit.items() if key != "id"}
    return {
        "id": point_id,
        "payload": payload,
        "vectors": {
            "dense": {"model": DEFAULT_DENSE_MODEL, "source": "local", "status": "not_embedded"},
            "sparse": {"model": DEFAULT_SPARSE_MODEL, "source": "qdrant", "field": "retrieval_text"},
        },
    }
