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
DEFAULT_QUERY_INSTRUCTION = ""
DEFAULT_PREFETCH_LIMIT = 20
DEFAULT_SEARCH_MODE = "hybrid"
DEFAULT_SEARCH_QDRANT_TIMEOUT = 30.0
DEFAULT_DOC_TITLE_FORMAT = "type1"
DEFAULT_ANSWER_ARTICLE_LIMIT: int | None = None
DEFAULT_RERANKER_MODEL = "AITeamVN/Vietnamese_Reranker"
DEFAULT_RERANKER_MAX_LENGTH = 2304
DEFAULT_RERANK_THRESHOLD: float | None = None
DEFAULT_EXCLUDE_LOCAL_DOCUMENTS = True
DEFAULT_REQUIRE_ARTICLE = True
DEFAULT_MODEL_CACHE_DIR = Path.cwd() / ".cache" 
DEFAULT_HNSW_M: int | None = None
DEFAULT_HNSW_EF_CONSTRUCT: int | None = None

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
    model_cache_dir: Path | None = DEFAULT_MODEL_CACHE_DIR
    recreate_collection: bool = False
    skip_build: bool = False
    limit: int | None = None
    max_chunk_tokens: int = DEFAULT_MAX_CHUNK_TOKENS
    chunk_overlap_tokens: int = DEFAULT_CHUNK_OVERLAP_TOKENS
    hnsw_m: int | None = DEFAULT_HNSW_M
    hnsw_ef_construct: int | None = DEFAULT_HNSW_EF_CONSTRUCT

    @classmethod
    def from_env(
        cls,
        source_dir: Path,
        build_dir: Path,
        collection_name: str | None = None,
        dense_model_name: str = DEFAULT_DENSE_MODEL,
        sparse_model_name: str = DEFAULT_SPARSE_MODEL,
        batch_size: int = 16,
        model_cache_dir: Path | None = DEFAULT_MODEL_CACHE_DIR,
        recreate_collection: bool = False,
        skip_build: bool = False,
        limit: int | None = None,
        max_chunk_tokens: int = DEFAULT_MAX_CHUNK_TOKENS,
        chunk_overlap_tokens: int = DEFAULT_CHUNK_OVERLAP_TOKENS,
        hnsw_m: int | None = DEFAULT_HNSW_M,
        hnsw_ef_construct: int | None = DEFAULT_HNSW_EF_CONSTRUCT,
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
            hnsw_m=hnsw_m,
            hnsw_ef_construct=hnsw_ef_construct,
        )


@dataclass(frozen=True)
class QdrantSearchConfig:
    collection_name: str
    qdrant_url: str
    qdrant_api_key: str
    query_text: str
    search_mode: str = DEFAULT_SEARCH_MODE
    dense_model_name: str = DEFAULT_DENSE_MODEL
    sparse_model_name: str = DEFAULT_SPARSE_MODEL
    dense_vector_name: str = DEFAULT_DENSE_VECTOR_NAME
    sparse_vector_name: str = DEFAULT_SPARSE_VECTOR_NAME
    top_k: int = 5
    prefetch_limit: int = DEFAULT_PREFETCH_LIMIT
    qdrant_timeout: float = DEFAULT_SEARCH_QDRANT_TIMEOUT
    doc_title_format: str = DEFAULT_DOC_TITLE_FORMAT
    answer_article_limit: int | None = DEFAULT_ANSWER_ARTICLE_LIMIT
    model_cache_dir: Path | None = DEFAULT_MODEL_CACHE_DIR
    query_vector: list[float] | None = None
    query_instruction: str = DEFAULT_QUERY_INSTRUCTION
    rerank: bool = False
    reranker_model_name: str = DEFAULT_RERANKER_MODEL
    reranker_max_length: int = DEFAULT_RERANKER_MAX_LENGTH
    rerank_threshold: float | None = DEFAULT_RERANK_THRESHOLD
    exclude_local_documents: bool = DEFAULT_EXCLUDE_LOCAL_DOCUMENTS
    require_article: bool = DEFAULT_REQUIRE_ARTICLE
    topic_title: str | None = None
    subject_title: str | None = None
    source_law_id: str | None = None
    source_article_no: str | None = None
    citation_confidence: str | None = None
    topic_number: int | None = None

    @classmethod
    def from_env(
        cls,
        query_text: str,
        collection_name: str | None = None,
        search_mode: str = DEFAULT_SEARCH_MODE,
        dense_model_name: str = DEFAULT_DENSE_MODEL,
        sparse_model_name: str = DEFAULT_SPARSE_MODEL,
        top_k: int = 5,
        prefetch_limit: int = DEFAULT_PREFETCH_LIMIT,
        qdrant_timeout: float = DEFAULT_SEARCH_QDRANT_TIMEOUT,
        doc_title_format: str = DEFAULT_DOC_TITLE_FORMAT,
        answer_article_limit: int | None = DEFAULT_ANSWER_ARTICLE_LIMIT,
        model_cache_dir: Path | None = DEFAULT_MODEL_CACHE_DIR,
        query_vector: list[float] | None = None,
        query_instruction: str = DEFAULT_QUERY_INSTRUCTION,
        rerank: bool = False,
        reranker_model_name: str = DEFAULT_RERANKER_MODEL,
        reranker_max_length: int = DEFAULT_RERANKER_MAX_LENGTH,
        rerank_threshold: float | None = DEFAULT_RERANK_THRESHOLD,
        exclude_local_documents: bool = DEFAULT_EXCLUDE_LOCAL_DOCUMENTS,
        require_article: bool = DEFAULT_REQUIRE_ARTICLE,
        topic_title: str | None = None,
        subject_title: str | None = None,
        source_law_id: str | None = None,
        source_article_no: str | None = None,
        citation_confidence: str | None = None,
        topic_number: int | None = None,
    ) -> "QdrantSearchConfig":
        qdrant_url = os.getenv("QDRANT_URL", "").strip()
        qdrant_api_key = os.getenv("QDRANT_API_KEY", "").strip()
        if not qdrant_url:
            raise RuntimeError("Missing QDRANT_URL environment variable.")
        if not qdrant_api_key:
            raise RuntimeError("Missing QDRANT_API_KEY environment variable.")
        return cls(
            collection_name=collection_name or os.getenv("QDRANT_COLLECTION", DEFAULT_COLLECTION),
            qdrant_url=qdrant_url,
            qdrant_api_key=qdrant_api_key,
            query_text=query_text,
            search_mode=search_mode,
            dense_model_name=dense_model_name,
            sparse_model_name=sparse_model_name,
            top_k=top_k,
            prefetch_limit=prefetch_limit,
            qdrant_timeout=qdrant_timeout,
            doc_title_format=doc_title_format,
            answer_article_limit=answer_article_limit,
            model_cache_dir=model_cache_dir,
            query_vector=query_vector,
            query_instruction=query_instruction,
            rerank=rerank,
            reranker_model_name=reranker_model_name,
            reranker_max_length=reranker_max_length,
            rerank_threshold=rerank_threshold,
            exclude_local_documents=exclude_local_documents,
            require_article=require_article,
            topic_title=topic_title,
            subject_title=subject_title,
            source_law_id=source_law_id,
            source_article_no=source_article_no,
            citation_confidence=citation_confidence,
            topic_number=topic_number,
        )
