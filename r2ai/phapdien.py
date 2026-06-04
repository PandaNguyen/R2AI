"""Compatibility exports for phapdien ingestion.

New code should import from ``r2ai.data_ingest.phapdien``.
"""

from r2ai.data_ingest.phapdien import (
    BuildPaths,
    build_phapdien_data,
    build_retrieval_text,
    build_source_title_map,
    canonical_hash_basis,
    canonicalize_article,
    extract_citation_candidates,
    make_retrieval_units,
    split_long_text,
    split_structured_text,
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
