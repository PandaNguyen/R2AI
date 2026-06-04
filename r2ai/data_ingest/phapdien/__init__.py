"""Phapdien corpus ingestion for the baseline."""

from r2ai.data_ingest.phapdien.builder import BuildPaths, build_phapdien_data
from r2ai.data_ingest.phapdien.citations import extract_citation_candidates
from r2ai.data_ingest.phapdien.chunking import split_long_text
from r2ai.data_ingest.phapdien.records import (
    build_retrieval_text,
    canonical_hash_basis,
    canonicalize_article,
    make_retrieval_units,
)

__all__ = [
    "BuildPaths",
    "build_phapdien_data",
    "build_retrieval_text",
    "canonical_hash_basis",
    "canonicalize_article",
    "extract_citation_candidates",
    "make_retrieval_units",
    "split_long_text",
]
