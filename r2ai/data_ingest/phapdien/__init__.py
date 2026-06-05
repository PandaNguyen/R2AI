"""Phapdien corpus ingestion for the baseline."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from r2ai.data_ingest.phapdien.builder import BuildPaths, build_phapdien_data
    from r2ai.data_ingest.phapdien.citations import extract_citation_candidates
    from r2ai.data_ingest.phapdien.chunking import split_long_text, split_structured_text
    from r2ai.data_ingest.phapdien.records import (
        build_retrieval_text,
        build_source_title_map,
        canonical_hash_basis,
        canonicalize_article,
        make_retrieval_units,
    )

__all__ = [
    "BuildPaths",
    "build_phapdien_data",
    "build_retrieval_text",
    "build_source_title_map",
    "canonical_hash_basis",
    "canonicalize_article",
    "extract_citation_candidates",
    "make_retrieval_units",
    "split_long_text",
    "split_structured_text",
]


def __getattr__(name: str) -> Any:
    if name in {"BuildPaths", "build_phapdien_data"}:
        from r2ai.data_ingest.phapdien.builder import BuildPaths, build_phapdien_data

        exports = {"BuildPaths": BuildPaths, "build_phapdien_data": build_phapdien_data}
        return exports[name]
    if name == "extract_citation_candidates":
        from r2ai.data_ingest.phapdien.citations import extract_citation_candidates

        return extract_citation_candidates
    if name in {"split_long_text", "split_structured_text"}:
        from r2ai.data_ingest.phapdien.chunking import split_long_text, split_structured_text

        exports = {"split_long_text": split_long_text, "split_structured_text": split_structured_text}
        return exports[name]
    if name in {
        "build_retrieval_text",
        "build_source_title_map",
        "canonical_hash_basis",
        "canonicalize_article",
        "make_retrieval_units",
    }:
        from r2ai.data_ingest.phapdien.records import (
            build_retrieval_text,
            build_source_title_map,
            canonical_hash_basis,
            canonicalize_article,
            make_retrieval_units,
        )

        exports = {
            "build_retrieval_text": build_retrieval_text,
            "build_source_title_map": build_source_title_map,
            "canonical_hash_basis": canonical_hash_basis,
            "canonicalize_article": canonicalize_article,
            "make_retrieval_units": make_retrieval_units,
        }
        return exports[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
