"""Batch ingestion of phapdien retrieval units into Qdrant Cloud."""

from __future__ import annotations

import json
from itertools import islice
from pathlib import Path
from typing import Any, Iterable, Iterator

from r2ai.data_ingest.phapdien import BuildPaths, build_phapdien_data
from r2ai.indexing.config import QdrantIngestConfig


def ingest_phapdien_to_qdrant(config: QdrantIngestConfig) -> dict[str, Any]:
    """Build phapdien artifacts if needed, then upsert them to Qdrant."""
    if not config.skip_build or not _qdrant_preview_path(config.build_dir).exists():
        build_phapdien_data(
            BuildPaths(
                source_dir=config.source_dir,
                output_dir=config.build_dir,
                max_chunk_tokens=config.max_chunk_tokens,
                chunk_overlap_tokens=config.chunk_overlap_tokens,
            )
        )

    dense_model = _load_dense_model(config.dense_model_name, config.model_cache_dir)
    sparse_model = _load_sparse_model(config.sparse_model_name, config.model_cache_dir)
    dense_size = int(dense_model.get_sentence_embedding_dimension())

    client, models = _make_qdrant_client(config.qdrant_url, config.qdrant_api_key)
    _ensure_collection(
        client=client,
        models=models,
        collection_name=config.collection_name,
        dense_vector_name=config.dense_vector_name,
        sparse_vector_name=config.sparse_vector_name,
        dense_size=dense_size,
        recreate=config.recreate_collection,
    )
    _ensure_payload_indexes(client=client, models=models, collection_name=config.collection_name)

    total_points = 0
    preview_rows = read_qdrant_preview_rows(_qdrant_preview_path(config.build_dir), limit=config.limit)
    for batch_index, rows in enumerate(batched(preview_rows, config.batch_size), start=1):
        points = make_points(
            rows=rows,
            dense_model=dense_model,
            sparse_model=sparse_model,
            models=models,
            dense_vector_name=config.dense_vector_name,
            sparse_vector_name=config.sparse_vector_name,
        )
        client.upsert(collection_name=config.collection_name, points=points, wait=True)
        total_points += len(points)
        print(f"Upserted batch {batch_index}: {len(points)} points, total={total_points}")

    return {
        "collection_name": config.collection_name,
        "dense_model": config.dense_model_name,
        "sparse_model": config.sparse_model_name,
        "dense_size": dense_size,
        "points_upserted": total_points,
    }


def read_qdrant_preview_rows(path: Path, limit: int | None = None) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if limit is not None and index >= limit:
                return
            if line.strip():
                yield json.loads(line)


def batched(rows: Iterable[dict[str, Any]], batch_size: int) -> Iterator[list[dict[str, Any]]]:
    iterator = iter(rows)
    while True:
        batch = list(islice(iterator, batch_size))
        if not batch:
            return
        yield batch


def make_points(
    rows: list[dict[str, Any]],
    dense_model: Any,
    sparse_model: Any,
    models: Any,
    dense_vector_name: str,
    sparse_vector_name: str,
) -> list[Any]:
    texts = [row["payload"]["retrieval_text"] for row in rows]
    dense_vectors = dense_model.encode(
        texts,
        batch_size=len(texts),
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    sparse_vectors = list(sparse_model.embed(texts))

    points = []
    for row, dense_vector, sparse_vector in zip(rows, dense_vectors, sparse_vectors, strict=True):
        points.append(
            models.PointStruct(
                id=row["id"],
                payload=row["payload"],
                vector={
                    dense_vector_name: dense_vector.tolist(),
                    sparse_vector_name: models.SparseVector(
                        indices=[int(index) for index in sparse_vector.indices.tolist()],
                        values=[float(value) for value in sparse_vector.values.tolist()],
                    ),
                },
            )
        )
    return points


def _qdrant_preview_path(build_dir: Path) -> Path:
    return build_dir / "qdrant_payload_preview.jsonl"


def _preferred_torch_device() -> str:
    try:
        import torch
    except ImportError:
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


def _load_dense_model(model_name: str, cache_dir: Path | None) -> Any:
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError("sentence-transformers is required for dense embeddings. Install the ingest extra.") from exc

    kwargs: dict[str, Any] = {"device": _preferred_torch_device()}
    if cache_dir:
        kwargs["cache_folder"] = str(cache_dir)
    model = SentenceTransformer(model_name, **kwargs)
    model.max_seq_length = 2048
    return model


def _load_sparse_model(model_name: str, cache_dir: Path | None) -> Any:
    try:
        from fastembed import SparseTextEmbedding
    except ImportError as exc:
        raise RuntimeError("fastembed is required for BM25 sparse embeddings. Install the ingest extra.") from exc

    kwargs = {"cache_dir": str(cache_dir)} if cache_dir else {}
    return SparseTextEmbedding(model_name=model_name, **kwargs)


def _make_qdrant_client(qdrant_url: str, qdrant_api_key: str) -> tuple[Any, Any]:
    try:
        from qdrant_client import QdrantClient, models
    except ImportError as exc:
        raise RuntimeError("qdrant-client is required for Qdrant ingestion. Install the ingest extra.") from exc

    return QdrantClient(url=qdrant_url, api_key=qdrant_api_key, timeout=120), models


def _ensure_collection(
    client: Any,
    models: Any,
    collection_name: str,
    dense_vector_name: str,
    sparse_vector_name: str,
    dense_size: int,
    recreate: bool,
) -> None:
    if recreate and client.collection_exists(collection_name=collection_name):
        client.delete_collection(collection_name=collection_name)

    if client.collection_exists(collection_name=collection_name):
        return

    client.create_collection(
        collection_name=collection_name,
        vectors_config={
            dense_vector_name: models.VectorParams(size=dense_size, distance=models.Distance.COSINE),
        },
        sparse_vectors_config={
            sparse_vector_name: models.SparseVectorParams(modifier=models.Modifier.IDF),
        },
    )


def _ensure_payload_indexes(client: Any, models: Any, collection_name: str) -> None:
    field_schemas = {
        "topic_title": models.PayloadSchemaType.KEYWORD,
        "subject_title": models.PayloadSchemaType.KEYWORD,
        "citation_confidence": models.PayloadSchemaType.KEYWORD,
        "source_law_id_candidates": models.PayloadSchemaType.KEYWORD,
        "source_article_no_candidates": models.PayloadSchemaType.KEYWORD,
        "topic_number": models.PayloadSchemaType.INTEGER,
    }
    for field_name, field_schema in field_schemas.items():
        try:
            client.create_payload_index(
                collection_name=collection_name,
                field_name=field_name,
                field_schema=field_schema,
                wait=True,
            )
        except Exception as exc:  # Qdrant returns an error when an index already exists.
            print(f"Skipped payload index {field_name}: {exc}")
