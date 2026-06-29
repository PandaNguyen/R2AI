"""Batch ingestion of phapdien retrieval units into Qdrant Cloud."""

from __future__ import annotations

import gc
import json
import os
import re
from itertools import islice
from pathlib import Path
from typing import Any, Iterable, Iterator

from r2ai.data_ingest.phapdien import BuildPaths, build_phapdien_data
from r2ai.indexing.config import QdrantIngestConfig
from r2ai.indexing.retry import retry_request


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
    dense_max_seq_length = int(getattr(dense_model, "max_seq_length", 0) or 0)
    payload_metadata = embedding_payload_metadata(config, dense_size, dense_max_seq_length)

    client, models = _make_qdrant_client(config.qdrant_url, config.qdrant_api_key)
    _ensure_collection(
        client=client,
        models=models,
        collection_name=config.collection_name,
        dense_vector_name=config.dense_vector_name,
        sparse_vector_name=config.sparse_vector_name,
        dense_size=dense_size,
        hnsw_m=config.hnsw_m,
        hnsw_ef_construct=config.hnsw_ef_construct,
        recreate=config.recreate_collection,
    )
    _ensure_payload_indexes(client=client, models=models, collection_name=config.collection_name)

    checkpoint_path = _ingest_checkpoint_path(
        config.build_dir,
        config.collection_name,
        config.dense_model_name,
        config.sparse_model_name,
        dense_size,
        dense_max_seq_length,
    )
    if config.recreate_collection and checkpoint_path.exists():
        checkpoint_path.unlink()
    completed_ids = read_ingest_checkpoint(checkpoint_path)
    if not config.recreate_collection and should_rebuild_checkpoint_from_qdrant(checkpoint_path):
        rebuilt = rebuild_ingest_checkpoint_from_qdrant(
            client=client,
            collection_name=config.collection_name,
            preview_path=_qdrant_preview_path(config.build_dir),
            checkpoint_path=checkpoint_path,
            completed_ids=completed_ids,
            batch_size=max(config.batch_size, 256),
            limit=config.limit,
            payload_metadata=payload_metadata,
        )
        if rebuilt:
            print(f"Rebuilt Qdrant ingest checkpoint from remote collection: {rebuilt} existing point ids", flush=True)
    if completed_ids:
        print(f"Resuming Qdrant ingest: {len(completed_ids)} point ids already checkpointed", flush=True)

    total_points = 0
    skipped_points = count_checkpointed_preview_rows(_qdrant_preview_path(config.build_dir), completed_ids, config.limit)
    preview_rows = read_qdrant_preview_rows(_qdrant_preview_path(config.build_dir), limit=config.limit)
    pending_rows = (row for row in preview_rows if str(row["id"]) not in completed_ids)
    upsert_batch_size = max(1, config.upsert_batch_size)
    for embedding_batch_index, rows in enumerate(batched(pending_rows, config.batch_size), start=1):
        dense_vectors, sparse_vectors = embed_rows(
            rows=rows,
            dense_model=dense_model,
            sparse_model=sparse_model,
            dense_model_name=config.dense_model_name,
        )
        for upsert_batch_index, start in enumerate(range(0, len(rows), upsert_batch_size), start=1):
            end = min(start + upsert_batch_size, len(rows))
            point_rows = rows[start:end]
            points = make_points_from_vectors(
                rows=point_rows,
                dense_vectors=dense_vectors[start:end],
                sparse_vectors=sparse_vectors[start:end],
                models=models,
                dense_vector_name=config.dense_vector_name,
                sparse_vector_name=config.sparse_vector_name,
                payload_metadata=payload_metadata,
            )
            retry_request(
                lambda: client.upsert(collection_name=config.collection_name, points=points, wait=True),
                label=f"Qdrant upsert embedding batch {embedding_batch_index}.{upsert_batch_index}",
            )
            append_ingest_checkpoint(checkpoint_path, point_rows)
            completed_ids.update(str(row["id"]) for row in point_rows)
            total_points += len(points)
            print(
                f"Upserted embedding batch {embedding_batch_index}.{upsert_batch_index}: "
                f"{len(points)} points, total={total_points}",
                flush=True,
            )
            del points
            release_batch_memory()
        del rows, dense_vectors, sparse_vectors
        release_batch_memory(clear_cuda_cache=True)

    return {
        "collection_name": config.collection_name,
        "dense_model": config.dense_model_name,
        "sparse_model": config.sparse_model_name,
        "dense_size": dense_size,
        "hnsw_m": config.hnsw_m,
        "hnsw_ef_construct": config.hnsw_ef_construct,
        "embedding_batch_size": config.batch_size,
        "upsert_batch_size": config.upsert_batch_size,
        "points_upserted": total_points,
        "points_skipped_from_checkpoint": max(0, skipped_points),
        "checkpoint": str(checkpoint_path),
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


def release_batch_memory(*, clear_cuda_cache: bool = False) -> None:
    gc.collect()
    if not clear_cuda_cache:
        return
    try:
        import torch
    except ImportError:
        return
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def count_checkpointed_preview_rows(path: Path, completed_ids: set[str], limit: int | None = None) -> int:
    return sum(1 for row in read_qdrant_preview_rows(path, limit=limit) if str(row["id"]) in completed_ids)


def read_ingest_checkpoint(path: Path) -> set[str]:
    if not path.exists():
        return set()
    completed: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            point_id = row.get("id")
            if point_id is not None:
                completed.add(str(point_id))
    return completed


def append_ingest_checkpoint(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps({"id": row["id"]}, ensure_ascii=False) + "\n")
        handle.flush()


def should_rebuild_checkpoint_from_qdrant(checkpoint_path: Path) -> bool:
    value = os.getenv("R2AI_REBUILD_INGEST_CHECKPOINT_FROM_QDRANT")
    if value is None:
        return not checkpoint_path.exists()
    return value.strip().lower() not in {"0", "false", "no", "off", ""}


def rebuild_ingest_checkpoint_from_qdrant(
    *,
    client: Any,
    collection_name: str,
    preview_path: Path,
    checkpoint_path: Path,
    completed_ids: set[str],
    batch_size: int,
    limit: int | None,
    payload_metadata: dict[str, Any] | None = None,
) -> int:
    rebuilt = 0
    checked = 0
    print("Checking Qdrant for existing point ids to rebuild ingest checkpoint...", flush=True)
    for rows in batched(read_qdrant_preview_rows(preview_path, limit=limit), batch_size):
        candidate_ids = [row["id"] for row in rows if str(row["id"]) not in completed_ids]
        if not candidate_ids:
            continue
        points = retry_request(
            lambda: client.retrieve(
                collection_name=collection_name,
                ids=candidate_ids,
                with_payload=True,
                with_vectors=False,
            ),
            label=f"Qdrant retrieve existing ids {checked}-{checked + len(candidate_ids)}",
        )
        rows_by_id = {str(row["id"]): row for row in rows}
        existing_rows = []
        for point in points:
            point_id = _point_id(point)
            if (
                point_id is not None
                and point_id not in completed_ids
                and remote_point_matches_preview(
                    rows_by_id.get(point_id),
                    point,
                    payload_metadata=payload_metadata,
                )
            ):
                existing_rows.append({"id": point_id})
        if existing_rows:
            append_ingest_checkpoint(checkpoint_path, existing_rows)
            completed_ids.update(row["id"] for row in existing_rows)
            rebuilt += len(existing_rows)
        checked += len(candidate_ids)
        if checked % 10000 == 0:
            print(f"Checked {checked} preview point ids against Qdrant; found {rebuilt} existing", flush=True)
    return rebuilt


def _point_id(point: Any) -> str | None:
    if isinstance(point, dict):
        value = point.get("id")
    else:
        value = getattr(point, "id", None)
    return str(value) if value is not None else None


def remote_point_matches_preview(
    preview_row: dict[str, Any] | None,
    point: Any,
    payload_metadata: dict[str, Any] | None = None,
) -> bool:
    if preview_row is None:
        return False
    expected_payload = dict(preview_row.get("payload") or {})
    if payload_metadata:
        expected_payload.update(payload_metadata)
    remote_payload = _point_payload(point)
    if not expected_payload:
        return True
    fields = (
        "dataset",
        "document_id",
        "chunk_id",
        "chunk_index",
        "chunk_count",
        "canonical_article_id",
        "retrieval_text_sha1",
        "embedding_dense_model",
        "embedding_sparse_model",
        "embedding_dense_size",
        "embedding_dense_max_seq_length",
        "embedding_dense_vector_name",
        "embedding_sparse_vector_name",
    )
    checked = False
    for field in fields:
        if field not in expected_payload:
            continue
        checked = True
        if not metadata_values_match(expected_payload.get(field), remote_payload.get(field)):
            return False
    return checked


def _point_payload(point: Any) -> dict[str, Any]:
    if isinstance(point, dict):
        payload = point.get("payload")
    else:
        payload = getattr(point, "payload", None)
    return payload if isinstance(payload, dict) else {}


def metadata_values_match(expected: Any, actual: Any) -> bool:
    if expected is None:
        return actual is None
    if isinstance(expected, (list, dict)):
        return expected == actual
    return str(expected) == str(actual)


def make_points(
    rows: list[dict[str, Any]],
    dense_model: Any,
    sparse_model: Any,
    models: Any,
    dense_vector_name: str,
    sparse_vector_name: str,
    dense_model_name: str | None = None,
    payload_metadata: dict[str, Any] | None = None,
) -> list[Any]:
    dense_vectors, sparse_vectors = embed_rows(
        rows=rows,
        dense_model=dense_model,
        sparse_model=sparse_model,
        dense_model_name=dense_model_name,
    )
    return make_points_from_vectors(
        rows=rows,
        dense_vectors=dense_vectors,
        sparse_vectors=sparse_vectors,
        models=models,
        dense_vector_name=dense_vector_name,
        sparse_vector_name=sparse_vector_name,
        payload_metadata=payload_metadata,
    )


def embed_rows(
    *,
    rows: list[dict[str, Any]],
    dense_model: Any,
    sparse_model: Any,
    dense_model_name: str | None = None,
) -> tuple[Any, list[Any]]:
    texts = [row["payload"]["retrieval_text"] for row in rows]
    dense_vectors = dense_model.encode(
        texts,
        batch_size=len(texts),
        normalize_embeddings=True,
        show_progress_bar=False,
        **_dense_encode_kwargs(dense_model_name, prompt_name="document"),
    )
    sparse_vectors = list(sparse_model.embed(texts))
    return dense_vectors, sparse_vectors


def make_points_from_vectors(
    *,
    rows: list[dict[str, Any]],
    dense_vectors: Any,
    sparse_vectors: list[Any],
    models: Any,
    dense_vector_name: str,
    sparse_vector_name: str,
    payload_metadata: dict[str, Any] | None = None,
) -> list[Any]:
    points = []
    for row, dense_vector, sparse_vector in zip(rows, dense_vectors, sparse_vectors, strict=True):
        payload = dict(row["payload"])
        if payload_metadata:
            payload.update(payload_metadata)
        points.append(
            models.PointStruct(
                id=row["id"],
                payload=payload,
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


def _ingest_checkpoint_path(
    build_dir: Path,
    collection_name: str,
    dense_model_name: str = "",
    sparse_model_name: str = "",
    dense_size: int | None = None,
    dense_max_seq_length: int | None = None,
) -> Path:
    parts = [collection_name]
    if dense_model_name:
        parts.append(dense_model_name)
    if sparse_model_name:
        parts.append(sparse_model_name)
    if dense_size is not None:
        parts.append(str(dense_size))
    if dense_max_seq_length is not None:
        parts.append(f"seq{dense_max_seq_length}")
    checkpoint_key = "__".join(parts)
    safe_key = re.sub(r"[^A-Za-z0-9_.-]+", "_", checkpoint_key).strip("_") or "collection"
    return build_dir / f"qdrant_ingest_checkpoint_{safe_key}.jsonl"


def embedding_payload_metadata(
    config: QdrantIngestConfig,
    dense_size: int,
    dense_max_seq_length: int,
) -> dict[str, Any]:
    return {
        "embedding_dense_model": config.dense_model_name,
        "embedding_sparse_model": config.sparse_model_name,
        "embedding_dense_size": dense_size,
        "embedding_dense_max_seq_length": dense_max_seq_length,
        "embedding_dense_vector_name": config.dense_vector_name,
        "embedding_sparse_vector_name": config.sparse_vector_name,
    }


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
    if _dense_model_requires_remote_code(model_name):
        kwargs["trust_remote_code"] = True
    model = SentenceTransformer(model_name, **kwargs)
    model.max_seq_length = 32768 if _is_jina_v5_text_model(model_name) else 2048
    return model


def _dense_model_requires_remote_code(model_name: str) -> bool:
    return _is_jina_v5_text_model(model_name)


def _dense_encode_kwargs(model_name: str | None, prompt_name: str) -> dict[str, str]:
    if _is_jina_v5_text_model(model_name or ""):
        return {"task": "retrieval", "prompt_name": prompt_name}
    return {}


def _is_jina_v5_text_model(model_name: str) -> bool:
    return model_name.lower().startswith("jinaai/jina-embeddings-v5-text")


def _load_sparse_model(model_name: str, cache_dir: Path | None) -> Any:
    try:
        from fastembed import SparseTextEmbedding
    except ImportError as exc:
        raise RuntimeError("fastembed is required for BM25 sparse embeddings. Install the ingest extra.") from exc

    kwargs = {"cache_dir": str(cache_dir)} if cache_dir else {}
    return SparseTextEmbedding(model_name=model_name, **kwargs)


def _make_qdrant_client(qdrant_url: str, qdrant_api_key: str, timeout: float = 120) -> tuple[Any, Any]:
    try:
        from qdrant_client import QdrantClient, models
    except ImportError as exc:
        raise RuntimeError("qdrant-client is required for Qdrant ingestion. Install the ingest extra.") from exc

    return QdrantClient(url=qdrant_url, api_key=qdrant_api_key, timeout=timeout), models


def _ensure_collection(
    client: Any,
    models: Any,
    collection_name: str,
    dense_vector_name: str,
    sparse_vector_name: str,
    dense_size: int,
    hnsw_m: int | None,
    hnsw_ef_construct: int | None,
    recreate: bool,
) -> None:
    if recreate and retry_request(
        lambda: client.collection_exists(collection_name=collection_name),
        label=f"Qdrant collection_exists {collection_name}",
    ):
        retry_request(
            lambda: client.delete_collection(collection_name=collection_name),
            label=f"Qdrant delete_collection {collection_name}",
        )

    if retry_request(
        lambda: client.collection_exists(collection_name=collection_name),
        label=f"Qdrant collection_exists {collection_name}",
    ):
        _validate_existing_collection(
            client=client,
            collection_name=collection_name,
            dense_vector_name=dense_vector_name,
            sparse_vector_name=sparse_vector_name,
            dense_size=dense_size,
        )
        _update_hnsw_config(
            client=client,
            models=models,
            collection_name=collection_name,
            hnsw_m=hnsw_m,
            hnsw_ef_construct=hnsw_ef_construct,
        )
        return

    retry_request(
        lambda: client.create_collection(
            collection_name=collection_name,
            vectors_config={
                dense_vector_name: models.VectorParams(size=dense_size, distance=models.Distance.COSINE),
            },
            sparse_vectors_config={
                sparse_vector_name: models.SparseVectorParams(modifier=models.Modifier.IDF),
            },
            hnsw_config=_hnsw_config_diff(models, hnsw_m=hnsw_m, hnsw_ef_construct=hnsw_ef_construct),
        ),
        label=f"Qdrant create_collection {collection_name}",
    )


def _update_hnsw_config(
    *,
    client: Any,
    models: Any,
    collection_name: str,
    hnsw_m: int | None,
    hnsw_ef_construct: int | None,
) -> None:
    hnsw_config = _hnsw_config_diff(models, hnsw_m=hnsw_m, hnsw_ef_construct=hnsw_ef_construct)
    if hnsw_config is None:
        return
    retry_request(
        lambda: client.update_collection(
            collection_name=collection_name,
            hnsw_config=hnsw_config,
        ),
        label=f"Qdrant update_collection HNSW {collection_name}",
    )


def _hnsw_config_diff(models: Any, *, hnsw_m: int | None, hnsw_ef_construct: int | None) -> Any | None:
    kwargs = {}
    if hnsw_m is not None:
        kwargs["m"] = hnsw_m
    if hnsw_ef_construct is not None:
        kwargs["ef_construct"] = hnsw_ef_construct
    if not kwargs:
        return None
    return models.HnswConfigDiff(**kwargs)


def _validate_existing_collection(
    *,
    client: Any,
    collection_name: str,
    dense_vector_name: str,
    sparse_vector_name: str,
    dense_size: int,
) -> None:
    collection_info = retry_request(
        lambda: client.get_collection(collection_name=collection_name),
        label=f"Qdrant get_collection {collection_name}",
    )
    actual_dense_size = _collection_dense_vector_size(collection_info, dense_vector_name)
    if actual_dense_size is None:
        raise RuntimeError(
            f"Existing Qdrant collection {collection_name!r} has no dense vector named "
            f"{dense_vector_name!r}. Use --recreate-collection or a new --collection."
        )
    if actual_dense_size != dense_size:
        raise RuntimeError(
            f"Existing Qdrant collection {collection_name!r} has dense vector {dense_vector_name!r} "
            f"size {actual_dense_size}, but the selected dense model requires {dense_size}. "
            "Use --recreate-collection or a new --collection when changing embedding models."
        )
    if not _collection_has_sparse_vector(collection_info, sparse_vector_name):
        raise RuntimeError(
            f"Existing Qdrant collection {collection_name!r} has no sparse vector named "
            f"{sparse_vector_name!r}. Use --recreate-collection or a new --collection."
        )


def _collection_dense_vector_size(collection_info: Any, dense_vector_name: str) -> int | None:
    params = _collection_params(collection_info)
    vectors = _object_value(params, "vectors")
    vector_params = vectors.get(dense_vector_name) if isinstance(vectors, dict) else vectors
    size = _object_value(vector_params, "size")
    return int(size) if size is not None else None


def _collection_has_sparse_vector(collection_info: Any, sparse_vector_name: str) -> bool:
    params = _collection_params(collection_info)
    sparse_vectors = _object_value(params, "sparse_vectors")
    if isinstance(sparse_vectors, dict):
        return sparse_vector_name in sparse_vectors
    return sparse_vectors is not None


def _collection_params(collection_info: Any) -> Any:
    return _object_value(_object_value(collection_info, "config"), "params")


def _object_value(value: Any, key: str) -> Any:
    if isinstance(value, dict):
        return value.get(key)
    return getattr(value, key, None)


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
            retry_request(
                lambda: client.create_payload_index(
                    collection_name=collection_name,
                    field_name=field_name,
                    field_schema=field_schema,
                    wait=True,
                ),
                label=f"Qdrant create_payload_index {field_name}",
            )
        except Exception as exc:  # Qdrant returns an error when an index already exists.
            print(f"Skipped payload index {field_name}: {exc}")
