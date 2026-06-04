"""Configuration for Qdrant ingestion."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from r2ai.data_ingest.phapdien.constants import DEFAULT_CHUNK_OVERLAP_TOKENS, DEFAULT_MAX_CHUNK_TOKENS


DEFAULT_COLLECTION = "r2ai_phapdien_baseline_v1"
DEFAULT_DENSE_MODEL = "AITeamVN/Vietnamese_Embedding_v2"
DEFAULT_SPARSE_MODEL = "Qdrant/bm25"
DEFAULT_DENSE_VECTOR_NAME = "dense"
DEFAULT_SPARSE_VECTOR_NAME = "bm25"
DEFAULT_QUERY_INSTRUCTION = (
    "Instruct: Given a Vietnamese legal question, retrieve relevant legal passages that answer the question\n"
    "Query: "
)


@dataclass(frozen=True)
class QdrantIngestConfig:
    source_dir: Path
    build_dir: Path
    collection_name: str
    qdrant_url: str
    qdrant_api_key: str
    dense_model_name: str = DEFAULT_DENSE_MODEL
    sparse_model_name: str = DEFAULT_SPARSE_MODEL
    dense_vector_name: str = DEFAULT_DENSE_VECTOR_NAME
    sparse_vector_name: str = DEFAULT_SPARSE_VECTOR_NAME
    batch_size: int = 16
    model_cache_dir: Path | None = None
    recreate_collection: bool = False
    skip_build: bool = False
    limit: int | None = None
    max_chunk_tokens: int = DEFAULT_MAX_CHUNK_TOKENS
    chunk_overlap_tokens: int = DEFAULT_CHUNK_OVERLAP_TOKENS

    @classmethod
    def from_env(
        cls,
        source_dir: Path,
        build_dir: Path,
        collection_name: str | None = None,
        dense_model_name: str = DEFAULT_DENSE_MODEL,
        sparse_model_name: str = DEFAULT_SPARSE_MODEL,
        batch_size: int = 16,
        model_cache_dir: Path | None = None,
        recreate_collection: bool = False,
        skip_build: bool = False,
        limit: int | None = None,
        max_chunk_tokens: int = DEFAULT_MAX_CHUNK_TOKENS,
        chunk_overlap_tokens: int = DEFAULT_CHUNK_OVERLAP_TOKENS,
    ) -> "QdrantIngestConfig":
        qdrant_url = os.getenv("QDRANT_URL", "").strip()
        qdrant_api_key = os.getenv("QDRANT_API_KEY", "").strip()
        if not qdrant_url:
            raise RuntimeError("Missing QDRANT_URL environment variable.")
        if not qdrant_api_key:
            raise RuntimeError("Missing QDRANT_API_KEY environment variable.")
        return cls(
            source_dir=source_dir,
            build_dir=build_dir,
            collection_name=collection_name or os.getenv("QDRANT_COLLECTION", DEFAULT_COLLECTION),
            qdrant_url=qdrant_url,
            qdrant_api_key=qdrant_api_key,
            dense_model_name=dense_model_name,
            sparse_model_name=sparse_model_name,
            batch_size=batch_size,
            model_cache_dir=model_cache_dir,
            recreate_collection=recreate_collection,
            skip_build=skip_build,
            limit=limit,
            max_chunk_tokens=max_chunk_tokens,
            chunk_overlap_tokens=chunk_overlap_tokens,
        )
