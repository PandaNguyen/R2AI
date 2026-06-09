"""Batch ingestion of phapdien retrieval units into Qdrant Cloud."""

from __future__ import annotations

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

    checkpoint_path = _ingest_checkpoint_path(config.build_dir, config.collection_name)
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
        )
        if rebuilt:
            print(f"Rebuilt Qdrant ingest checkpoint from remote collection: {rebuilt} existing point ids", flush=True)
    if completed_ids:
        print(f"Resuming Qdrant ingest: {len(completed_ids)} point ids already checkpointed", flush=True)

    total_points = 0
    skipped_points = count_checkpointed_preview_rows(_qdrant_preview_path(config.build_dir), completed_ids, config.limit)
    preview_rows = read_qdrant_preview_rows(_qdrant_preview_path(config.build_dir), limit=config.limit)
    pending_rows = (row for row in preview_rows if str(row["id"]) not in completed_ids)
    for batch_index, rows in enumerate(batched(pending_rows, config.batch_size), start=1):
        points = make_points(
            rows=rows,
            dense_model=dense_model,
            sparse_model=sparse_model,
            models=models,
            dense_vector_name=config.dense_vector_name,
            sparse_vector_name=config.sparse_vector_name,
        )
        retry_request(
            lambda: client.upsert(collection_name=config.collection_name, points=points, wait=True),
            label=f"Qdrant upsert batch {batch_index}",
        )
        append_ingest_checkpoint(checkpoint_path, rows)
        completed_ids.update(str(row["id"]) for row in rows)
        total_points += len(points)
        print(f"Upserted batch {batch_index}: {len(points)} points, total={total_points}")

    return {
        "collection_name": config.collection_name,
        "dense_model": config.dense_model_name,
        "sparse_model": config.sparse_model_name,
        "dense_size": dense_size,
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
                and remote_point_matches_preview(rows_by_id.get(point_id), point)
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


def remote_point_matches_preview(preview_row: dict[str, Any] | None, point: Any) -> bool:
    if preview_row is None:
        return False
    expected_payload = preview_row.get("payload") or {}
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


def _ingest_checkpoint_path(build_dir: Path, collection_name: str) -> Path:
    safe_collection = re.sub(r"[^A-Za-z0-9_.-]+", "_", collection_name).strip("_") or "collection"
    return build_dir / f"qdrant_ingest_checkpoint_{safe_collection}.jsonl"


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
        ),
        label=f"Qdrant create_collection {collection_name}",
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
