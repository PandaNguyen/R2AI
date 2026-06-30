"""Helpers for writing retrieval-only IR result rows before QA generation."""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from r2ai.search.contracts import SearchBackend, SearchHit, SearchQuery, SearchResponse


def search_many_ir(
    backend: SearchBackend,
    queries: Iterable[SearchQuery],
    *,
    progress_every: int = 0,
) -> list[dict[str, Any]]:
    query_list = list(queries)
    search_many = getattr(backend, "search_many", None)
    if callable(search_many):
        responses = search_many(query_list)
        return [response_to_ir_row(response) for response in responses]

    rows: list[dict[str, Any]] = []
    for index, query in enumerate(query_list, start=1):
        rows.append(response_to_ir_row(backend.search(query)))
        if progress_every > 0 and index % progress_every == 0:
            print(f"Searched {index} queries", flush=True)
    return rows


def response_to_ir_row(response: SearchResponse) -> dict[str, Any]:
    """Serialize a SearchResponse as a candidate-result row."""

    return {
        "id": response.query.id,
        "question": response.query.text,
        "result": {
            **response.metadata,
            "backend": response.backend,
            "results": [hit_to_result_item(hit) for hit in response.hits],
        },
    }


def hit_to_result_item(hit: SearchHit) -> dict[str, Any]:
    item = dict(hit.metadata)
    item.update(
        {
            "id": str(hit.id),
            "rank": hit.rank,
            "score": hit.score,
            "payload": hit.payload,
        }
    )
    return item


def write_ir_rows(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
            handle.write("\n")
