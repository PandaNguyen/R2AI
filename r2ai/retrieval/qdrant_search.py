"""Qdrant retrieval for Vietnamese legal retrieval units."""

from __future__ import annotations

import csv
import contextlib
import json
import re
import sys
import time
import zipfile
from dataclasses import replace
from pathlib import Path
from typing import Any, Iterable, Literal

from r2ai.indexing.config import (
    DEFAULT_JINA_RERANKER_MODEL,
    DEFAULT_QUERY_INSTRUCTION,
    DEFAULT_RERANKER_MODEL,
    QdrantSearchConfig,
)
from r2ai.indexing.qdrant_ingest import _dense_encode_kwargs, _load_dense_model, _load_sparse_model, _make_qdrant_client
from r2ai.indexing.retry import is_retryable_request_error, retry_request

RRF_K = 60
SEARCH_MODES = {"bm25", "dense", "hybrid"}
QUESTION_ID_KEYS = ("id", "question_id", "qid")
QUESTION_TEXT_KEYS = ("question", "query", "text")
ARTICLE_NO_PATTERN = re.compile(r"\bĐiều\s+\d+[A-Za-z]?", re.IGNORECASE)
HEADER_ARTICLE_SPLIT_PATTERN = re.compile(r"\s+Điều\s+\d+[A-Za-z]?\b", re.IGNORECASE)
DOC_TITLE_CODE_PATTERN = re.compile(
    r"^(?P<kind>.+?)\s+số\s+(?P<code>\S+)\s+(?P<title>.+)$",
    re.IGNORECASE,
)
DOC_TITLE_PATH_SEGMENT_PATTERN = re.compile(
    r"(?:^|[.;])\s*(?:(?:Phần|Chương|Mục|Tiểu mục)\s+[A-ZĐIVXLCDM\d]+(?:\.\d+)*|"
    r"Phụ lục|Danh mục|Mẫu số)\b",
    re.IGNORECASE,
)
DOC_TITLE_SOURCE_TYPE_PATTERN = re.compile(
    r"^(?:Bộ luật|Luật|Pháp lệnh|Nghị quyết|Nghị định|Thông tư liên tịch|Thông tư|Quyết định|"
    r"Văn bản hợp nhất)\b",
    re.IGNORECASE,
)
LOCAL_LAW_ID_PATTERN = re.compile(
    r"/(?:NQ-HĐND|QĐ-UBND|QĐ-HĐND|CT-UBND|QĐ-TTPVHCC|QĐ-S[A-ZĐ]+)\b",
    re.IGNORECASE,
)
CENTRAL_LAW_ID_PATTERN = re.compile(
    r"/(?:QH\d*|UBTVQH\d*|NĐ-CP|TT-[A-ZĐ]+|TTLT-[A-ZĐ]+|VBHN-[A-ZĐ]+|"
    r"QĐ-(?:TTG|BTC|BCT|BYT|BGDĐT|BLĐTBXH|BKHĐT|BNNPTNT|BTNMT|BTP|BXD|BCA|"
    r"BQP|BVHTTDL|BKHCN|NHNN|BHXH|VPCP))\b",
    re.IGNORECASE,
)
LOCAL_AUTHORITY_PATTERN = re.compile(
    r"\b(?:ủy ban nhân dân|ubnd|hội đồng nhân dân|hđnd|sở [a-zà-ỹ]|"
    r"trung tâm phục vụ hành chính công|cấp tỉnh|cấp huyện|cấp xã)\b",
    re.IGNORECASE,
)
LOCAL_TITLE_PATTERN = re.compile(
    r"(?:\b(?:tỉnh|thành phố|tp\.)\s+[A-ZÀ-ỸĐ]|\b(?:do|của)\s+(?:ủy ban nhân dân|ubnd|hội đồng nhân dân|hđnd|sở ))",
    re.IGNORECASE,
)
OutputFormat = Literal["json", "jsonl"]
TRACE_SAMPLE_LIMIT = 5


def trace_log_enabled(enabled: bool, event: str, **fields: Any) -> None:
    if not enabled:
        return
    record = {"event": event, **fields}
    print(json.dumps(record, ensure_ascii=False, sort_keys=True), file=sys.stderr, flush=True)


def trace_log(config: QdrantSearchConfig, event: str, **fields: Any) -> None:
    trace_log_enabled(config.trace_search, event, **fields)


def trace_search_results(
    config: QdrantSearchConfig,
    event: str,
    results: list[dict[str, Any]],
    *,
    query_id: Any | None = None,
    limit: int = TRACE_SAMPLE_LIMIT,
    **fields: Any,
) -> None:
    if not config.trace_search:
        return
    trace_log(
        config,
        event,
        query_id=query_id,
        count=len(results),
        samples=[trace_result_summary(item, config.doc_title_format) for item in results[:limit]],
        **fields,
    )


def trace_result_summary(item: dict[str, Any], doc_title_format: str) -> dict[str, Any]:
    payload = item.get("payload") or {}
    docs = payload_to_competition_docs(payload, doc_title_format=doc_title_format)
    articles = payload_to_competition_articles(payload, doc_title_format=doc_title_format)
    return {
        "id": item.get("id"),
        "rank": item.get("rank"),
        "score": item.get("score"),
        "rerank_score": item.get("rerank_score"),
        "retrieval_rank": item.get("retrieval_rank"),
        "doc": docs[0] if docs else "",
        "article": articles[0] if articles else "",
        "is_local": is_local_document_payload(payload),
        "has_article": bool(articles),
        "payload_keys": sorted(payload)[:12],
    }


def trace_submission_row(result: dict[str, Any], row: dict[str, Any]) -> None:
    trace_log_enabled(
        bool(result.get("trace_search")),
        "submission_row",
        query_id=row.get("id"),
        relevant_doc_count=len(row.get("relevant_docs") or []),
        relevant_article_count=len(row.get("relevant_articles") or []),
        answer=row.get("answer"),
        relevant_docs=row.get("relevant_docs") or [],
        relevant_articles=row.get("relevant_articles") or [],
    )


def search_qdrant(
    config: QdrantSearchConfig,
    *,
    dense_model: Any | None = None,
    sparse_model: Any | None = None,
    reranker: Any | None = None,
    client: Any | None = None,
    models: Any | None = None,
) -> dict[str, Any]:
    """Run bm25, dense, or hybrid retrieval against Qdrant."""
    mode = normalize_search_mode(config.search_mode)
    if client is None or models is None:
        client, models = _make_qdrant_client(
            config.qdrant_url,
            config.qdrant_api_key,
            timeout=config.qdrant_timeout,
        )

    query_text = config.query_text.strip()
    if not query_text:
        raise ValueError("query_text must not be empty.")

    query_filter = build_qdrant_filter(config, models)
    search_limit = search_limit_for_config(config)
    retrieval_limit = retrieval_limit_for_config(config, search_limit)
    trace_log(
        config,
        "search_start",
        query_text=query_text,
        collection=config.collection_name,
        mode=mode,
        top_k=config.top_k,
        prefetch_limit=search_limit,
        retrieval_limit=retrieval_limit,
        rerank=config.rerank,
        reranker_model=config.reranker_model_name if config.rerank else None,
        reranker_backend="jina" if config.rerank and config.use_jina_reranker else ("cross_encoder" if config.rerank else None),
        filters=summarize_filters(config),
    )
    dense_vector = None
    sparse_vector = None

    if mode in {"dense", "hybrid"}:
        dense_vector = dense_query_vector(config, dense_model=dense_model)
    if mode in {"bm25", "hybrid"}:
        if sparse_model is None:
            sparse_model = _load_sparse_model(config.sparse_model_name, config.model_cache_dir)
        sparse_vector = sparse_query_vector(query_text, sparse_model=sparse_model, models=models)

    if mode == "dense":
        hits = _response_points(
            _query_points(
                client,
                label="Qdrant dense query",
                collection_name=config.collection_name,
                query=dense_vector,
                using=config.dense_vector_name,
                query_filter=query_filter,
                limit=retrieval_limit,
                with_payload=True,
                with_vectors=False,
            )
        )
        results = ranked_points_to_results(hits, source="dense")
    elif mode == "bm25":
        hits = _response_points(
            _query_points(
                client,
                label="Qdrant BM25 query",
                collection_name=config.collection_name,
                query=sparse_vector,
                using=config.sparse_vector_name,
                query_filter=query_filter,
                limit=retrieval_limit,
                with_payload=True,
                with_vectors=False,
            )
        )
        results = ranked_points_to_results(hits, source="sparse")
    elif has_server_side_rrf(models):
        hits = _response_points(
            _query_points(
                client,
                label="Qdrant hybrid query",
                collection_name=config.collection_name,
                prefetch=[
                    models.Prefetch(query=sparse_vector, using=config.sparse_vector_name, limit=search_limit),
                    models.Prefetch(query=dense_vector, using=config.dense_vector_name, limit=search_limit),
                ],
                query=rrf_query(models),
                query_filter=query_filter,
                limit=retrieval_limit,
                with_payload=True,
                with_vectors=False,
            )
        )
        results = ranked_points_to_results(hits, source="hybrid")
    else:
        dense_hits = _response_points(
            _query_points(
                client,
                label="Qdrant client-side hybrid dense query",
                collection_name=config.collection_name,
                query=dense_vector,
                using=config.dense_vector_name,
                query_filter=query_filter,
                limit=search_limit,
                with_payload=True,
                with_vectors=False,
            )
        )
        sparse_hits = _response_points(
            _query_points(
                client,
                label="Qdrant client-side hybrid sparse query",
                collection_name=config.collection_name,
                query=sparse_vector,
                using=config.sparse_vector_name,
                query_filter=query_filter,
                limit=search_limit,
                with_payload=True,
                with_vectors=False,
            )
        )
        results = fuse_ranked_points(dense_hits=dense_hits, sparse_hits=sparse_hits, limit=retrieval_limit)

    trace_search_results(config, "search_raw_results", results, retrieval_limit=retrieval_limit)
    results = maybe_rerank_results(config, query_text, results, reranker=reranker)
    trace_search_results(config, "search_final_results", results, top_k=config.top_k)

    return {
        "collection_name": config.collection_name,
        "query_text": query_text,
        "search_mode": mode,
        "top_k": config.top_k,
        "prefetch_limit": search_limit,
        "doc_title_format": config.doc_title_format,
        "answer_article_limit": config.answer_article_limit,
        "exclude_local_documents": config.exclude_local_documents,
        "require_article": config.require_article,
        "trace_search": config.trace_search,
        "used_precomputed_dense_vector": config.query_vector is not None,
        "rerank": config.rerank,
        "reranker_model": config.reranker_model_name if config.rerank else None,
        "reranker_backend": "jina" if config.rerank and config.use_jina_reranker else ("cross_encoder" if config.rerank else None),
        "reranker_max_length": None if config.rerank and config.use_jina_reranker else (config.reranker_max_length if config.rerank else None),
        "rerank_threshold": config.rerank_threshold if config.rerank else None,
        "filters": summarize_filters(config),
        "results": results,
    }


def search_qdrant_batch(
    config: QdrantSearchConfig,
    *,
    questions: list[dict[str, Any]],
    query_vectors: list[list[float]] | None = None,
    dense_model: Any | None = None,
    sparse_model: Any | None = None,
    reranker: Any | None = None,
    client: Any | None = None,
    models: Any | None = None,
    progress_every: int = 0,
    query_batch_size: int = 64,
    checkpoint_path: Path | None = None,
) -> list[dict[str, Any]]:
    """Run retrieval for many questions, optionally with precomputed dense vectors."""
    if query_vectors is not None and len(query_vectors) != len(questions):
        raise ValueError(
            f"query vector count ({len(query_vectors)}) must match question count ({len(questions)})."
        )
    completed_rows = load_submission_checkpoint(checkpoint_path) if checkpoint_path is not None else {}
    if completed_rows:
        print(f"Resuming submission search: {len(completed_rows)} question ids already checkpointed", flush=True)
    pending_questions = []
    pending_vectors = [] if query_vectors is not None else None
    for index, question in enumerate(questions):
        if str(question["id"]) in completed_rows:
            continue
        pending_questions.append(question)
        if pending_vectors is not None:
            pending_vectors.append(query_vectors[index])
    if not pending_questions:
        return ordered_checkpoint_rows(questions, completed_rows)

    if client is None or models is None:
        client, models = _make_qdrant_client(
            config.qdrant_url,
            config.qdrant_api_key,
            timeout=config.qdrant_timeout,
        )
    mode = normalize_search_mode(config.search_mode)
    if mode in {"dense", "hybrid"} and query_vectors is None and dense_model is None:
        dense_model = _load_dense_model(config.dense_model_name, config.model_cache_dir)
    if mode in {"bm25", "hybrid"} and sparse_model is None:
        sparse_model = _load_sparse_model(config.sparse_model_name, config.model_cache_dir)
    if config.rerank and reranker is None:
        reranker = load_reranker(config.reranker_model_name, config.model_cache_dir, use_jina_reranker=config.use_jina_reranker)
    if (
        mode == "dense"
        and pending_vectors is not None
        and query_batch_size > 1
        and hasattr(client, "query_batch_points")
        and hasattr(models, "QueryRequest")
    ):
        new_rows = search_qdrant_dense_batch(
            config,
            questions=pending_questions,
            query_vectors=pending_vectors,
            client=client,
            models=models,
            reranker=reranker,
            progress_every=progress_every,
            query_batch_size=query_batch_size,
            checkpoint_path=checkpoint_path,
        )
        completed_rows.update({str(row["id"]): row for row in new_rows})
        return ordered_checkpoint_rows(questions, completed_rows)
    if (
        mode in {"bm25", "hybrid"}
        and query_batch_size > 1
        and hasattr(client, "query_batch_points")
        and hasattr(models, "QueryRequest")
        and (mode == "bm25" or (pending_vectors is not None and has_server_side_rrf(models)))
    ):
        if progress_every > 0:
            print(f"Building sparse query vectors for {len(pending_questions)} pending questions...", flush=True)
        sparse_vectors = sparse_query_vectors([question["question"] for question in pending_questions], sparse_model, models)
        new_rows = search_qdrant_sparse_or_hybrid_batch(
            config,
            questions=pending_questions,
            query_vectors=pending_vectors,
            sparse_vectors=sparse_vectors,
            client=client,
            models=models,
            reranker=reranker,
            progress_every=progress_every,
            query_batch_size=query_batch_size,
            checkpoint_path=checkpoint_path,
        )
        completed_rows.update({str(row["id"]): row for row in new_rows})
        return ordered_checkpoint_rows(questions, completed_rows)

    rows = []
    started_at = time.monotonic()
    total = len(pending_questions)
    for index, question in enumerate(pending_questions):
        query_vector = pending_vectors[index] if pending_vectors is not None else None
        try:
            result = search_qdrant(
                replace(
                    config,
                    query_text=question["question"],
                    query_vector=query_vector,
                ),
                dense_model=dense_model,
                sparse_model=sparse_model,
                reranker=reranker,
                client=client,
                models=models,
            )
        except Exception as exc:
            if not is_retryable_request_error(exc):
                raise
            print(f"Skipping question {question['id']} after exhausted request retries: {exc}", flush=True)
            result = empty_search_result(config, question["question"], mode)
        row = format_competition_row(question, result)
        append_submission_checkpoint(checkpoint_path, row)
        rows.append(row)
        done = index + 1
        if progress_every > 0 and (done == 1 or done == total or done % progress_every == 0):
            elapsed = time.monotonic() - started_at
            print(f"Searched {done}/{total} questions in {elapsed:.1f}s", flush=True)
    completed_rows.update({str(row["id"]): row for row in rows})
    return ordered_checkpoint_rows(questions, completed_rows)


def search_qdrant_dense_batch(
    config: QdrantSearchConfig,
    *,
    questions: list[dict[str, Any]],
    query_vectors: list[list[float]],
    client: Any,
    models: Any,
    reranker: Any | None = None,
    progress_every: int = 0,
    query_batch_size: int = 64,
    checkpoint_path: Path | None = None,
) -> list[dict[str, Any]]:
    query_filter = build_qdrant_filter(config, models)
    search_limit = search_limit_for_config(config)
    retrieval_limit = retrieval_limit_for_config(config, search_limit)
    rows = []
    started_at = time.monotonic()
    total = len(questions)
    for start in range(0, total, query_batch_size):
        end = min(start + query_batch_size, total)
        requests = [
            models.QueryRequest(
                query=[float(value) for value in vector],
                using=config.dense_vector_name,
                filter=query_filter,
                limit=retrieval_limit,
                with_payload=True,
                with_vector=False,
            )
            for vector in query_vectors[start:end]
        ]
        responses = _query_batch_points(
            client,
            collection_name=config.collection_name,
            requests=requests,
            label=f"Qdrant dense batch {start}-{end}",
            allow_empty_on_failure=True,
        )
        for question, response in zip(questions[start:end], responses, strict=True):
            result = {
                "collection_name": config.collection_name,
                "query_text": question["question"],
                "search_mode": "dense",
                "top_k": config.top_k,
                "prefetch_limit": search_limit_for_config(config),
                "doc_title_format": config.doc_title_format,
                "answer_article_limit": config.answer_article_limit,
                "exclude_local_documents": config.exclude_local_documents,
                "require_article": config.require_article,
                "trace_search": config.trace_search,
                "used_precomputed_dense_vector": True,
                "rerank": config.rerank,
                "reranker_model": config.reranker_model_name if config.rerank else None,
                "reranker_backend": "jina" if config.rerank and config.use_jina_reranker else ("cross_encoder" if config.rerank else None),
                "reranker_max_length": None if config.rerank and config.use_jina_reranker else (config.reranker_max_length if config.rerank else None),
                "rerank_threshold": config.rerank_threshold if config.rerank else None,
                "filters": summarize_filters(config),
                "results": maybe_rerank_results(
                    config,
                    question["question"],
                    ranked_points_to_results(_response_points(response), source="dense"),
                    reranker=reranker,
                    query_id=question["id"],
                ),
            }
            row = format_competition_row(question, result)
            append_submission_checkpoint(checkpoint_path, row)
            rows.append(row)
        done = end
        if progress_every > 0 and (done == total or done % progress_every == 0 or start == 0):
            elapsed = time.monotonic() - started_at
            print(f"Searched {done}/{total} questions in {elapsed:.1f}s", flush=True)
    return rows


def search_qdrant_sparse_or_hybrid_batch(
    config: QdrantSearchConfig,
    *,
    questions: list[dict[str, Any]],
    query_vectors: list[list[float]] | None,
    sparse_vectors: list[Any],
    client: Any,
    models: Any,
    reranker: Any | None = None,
    progress_every: int = 0,
    query_batch_size: int = 64,
    checkpoint_path: Path | None = None,
) -> list[dict[str, Any]]:
    mode = normalize_search_mode(config.search_mode)
    query_filter = build_qdrant_filter(config, models)
    search_limit = search_limit_for_config(config)
    retrieval_limit = retrieval_limit_for_config(config, search_limit)
    rows = []
    started_at = time.monotonic()
    total = len(questions)
    for start in range(0, total, query_batch_size):
        end = min(start + query_batch_size, total)
        requests = []
        for index in range(start, end):
            if mode == "bm25":
                requests.append(
                    models.QueryRequest(
                        query=sparse_vectors[index],
                        using=config.sparse_vector_name,
                        filter=query_filter,
                        limit=retrieval_limit,
                        with_payload=True,
                        with_vector=False,
                    )
                )
            else:
                requests.append(
                    models.QueryRequest(
                        prefetch=[
                            models.Prefetch(query=sparse_vectors[index], using=config.sparse_vector_name, limit=search_limit),
                            models.Prefetch(
                                query=[float(value) for value in query_vectors[index]],
                                using=config.dense_vector_name,
                                limit=search_limit,
                            ),
                        ],
                        query=rrf_query(models),
                        filter=query_filter,
                        limit=retrieval_limit,
                        with_payload=True,
                        with_vector=False,
                    )
                )
        responses = _query_batch_points(
            client,
            collection_name=config.collection_name,
            requests=requests,
            label=f"Qdrant {mode} batch {start}-{end}",
            allow_empty_on_failure=True,
        )
        for question, response in zip(questions[start:end], responses, strict=True):
            result = {
                "collection_name": config.collection_name,
                "query_text": question["question"],
                "search_mode": mode,
                "top_k": config.top_k,
                "prefetch_limit": search_limit,
                "doc_title_format": config.doc_title_format,
                "answer_article_limit": config.answer_article_limit,
                "exclude_local_documents": config.exclude_local_documents,
                "require_article": config.require_article,
                "trace_search": config.trace_search,
                "used_precomputed_dense_vector": query_vectors is not None,
                "rerank": config.rerank,
                "reranker_model": config.reranker_model_name if config.rerank else None,
                "reranker_backend": "jina" if config.rerank and config.use_jina_reranker else ("cross_encoder" if config.rerank else None),
                "reranker_max_length": None if config.rerank and config.use_jina_reranker else (config.reranker_max_length if config.rerank else None),
                "rerank_threshold": config.rerank_threshold if config.rerank else None,
                "filters": summarize_filters(config),
                "results": maybe_rerank_results(
                    config,
                    question["question"],
                    ranked_points_to_results(
                        _response_points(response),
                        source="sparse" if mode == "bm25" else "hybrid",
                    ),
                    reranker=reranker,
                    query_id=question["id"],
                ),
            }
            row = format_competition_row(question, result)
            append_submission_checkpoint(checkpoint_path, row)
            rows.append(row)
        done = end
        if progress_every > 0 and (done == total or done % progress_every == 0 or start == 0):
            elapsed = time.monotonic() - started_at
            print(f"Searched {done}/{total} questions in {elapsed:.1f}s", flush=True)
    return rows


def load_questions(path: Path) -> list[dict[str, str]]:
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        with path.open("r", encoding="utf-8") as handle:
            raw_rows = [json.loads(line) for line in handle if line.strip()]
    elif suffix == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        raw_rows = data.get("questions", data) if isinstance(data, dict) else data
    elif suffix == ".csv":
        with path.open("r", encoding="utf-8", newline="") as handle:
            raw_rows = list(csv.DictReader(handle))
    else:
        raise ValueError(f"Unsupported question file type: {path.suffix}")

    questions = []
    for index, row in enumerate(raw_rows):
        question_id = first_present(row, QUESTION_ID_KEYS, default=str(index))
        question_text = first_present(row, QUESTION_TEXT_KEYS)
        if not question_text:
            raise ValueError(f"Question row {index} is missing one of these fields: {QUESTION_TEXT_KEYS}")
        questions.append({"id": question_id, "question": str(question_text)})
    return questions


def load_query_vectors(path: Path) -> list[list[float]]:
    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("numpy is required to load precomputed .npy query embeddings.") from exc

    vectors = np.load(path)
    if vectors.ndim == 1:
        vectors = vectors.reshape(1, -1)
    if vectors.ndim != 2:
        raise ValueError(f"Expected a 1D or 2D .npy array, got shape {vectors.shape}.")
    return [[float(value) for value in row] for row in vectors.tolist()]


def write_submission(path: Path, rows: Iterable[dict[str, Any]], output_format: OutputFormat = "json") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    if output_format == "jsonl":
        with path.open("w", encoding="utf-8", newline="\n") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
                handle.write("\n")
        return
    path.write_text(json.dumps(rows, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def load_submission_checkpoint(path: Path | None) -> dict[str, dict[str, Any]]:
    if path is None or not path.exists():
        return {}
    rows: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            question_id = row.get("id")
            if question_id is not None:
                rows[str(question_id)] = row
    return rows


def append_submission_checkpoint(path: Path | None, row: dict[str, Any]) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()


def ordered_checkpoint_rows(
    questions: list[dict[str, Any]],
    rows_by_id: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    return [rows_by_id[str(question["id"])] for question in questions if str(question["id"]) in rows_by_id]


def write_submission_zip(zip_path: Path, results_path: Path) -> None:
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.write(results_path, arcname="results.json")


def empty_search_result(config: QdrantSearchConfig, query_text: str, mode: str) -> dict[str, Any]:
    return {
        "collection_name": config.collection_name,
        "query_text": query_text,
        "search_mode": mode,
        "top_k": config.top_k,
        "prefetch_limit": search_limit_for_config(config),
        "doc_title_format": config.doc_title_format,
        "answer_article_limit": config.answer_article_limit,
        "exclude_local_documents": config.exclude_local_documents,
        "require_article": config.require_article,
        "trace_search": config.trace_search,
        "used_precomputed_dense_vector": config.query_vector is not None,
        "rerank": config.rerank,
        "reranker_model": config.reranker_model_name if config.rerank else None,
        "reranker_backend": "jina" if config.rerank and config.use_jina_reranker else ("cross_encoder" if config.rerank else None),
        "reranker_max_length": None if config.rerank and config.use_jina_reranker else (config.reranker_max_length if config.rerank else None),
        "rerank_threshold": config.rerank_threshold if config.rerank else None,
        "filters": summarize_filters(config),
        "results": [],
    }


def dense_query_vector(config: QdrantSearchConfig, *, dense_model: Any | None) -> list[float]:
    if config.query_vector is not None:
        return [float(value) for value in config.query_vector]
    if dense_model is None:
        dense_model = _load_dense_model(config.dense_model_name, config.model_cache_dir)
    encode_kwargs = _dense_encode_kwargs(config.dense_model_name, prompt_name="query")
    dense_query_text = config.query_text.strip()
    if not encode_kwargs:
        query_instruction = config.query_instruction if config.query_instruction is not None else DEFAULT_QUERY_INSTRUCTION
        dense_query_text = f"{query_instruction}{dense_query_text}"
    dense_vector = dense_model.encode(
        [dense_query_text],
        batch_size=1,
        normalize_embeddings=True,
        show_progress_bar=False,
        **encode_kwargs,
    )[0]
    return [float(value) for value in dense_vector.tolist()]


def sparse_query_vector(query_text: str, *, sparse_model: Any, models: Any) -> Any:
    sparse_embedding = next(iter(sparse_model.embed([query_text])))
    return models.SparseVector(
        indices=[int(index) for index in sparse_embedding.indices.tolist()],
        values=[float(value) for value in sparse_embedding.values.tolist()],
    )


def sparse_query_vectors(query_texts: list[str], sparse_model: Any, models: Any) -> list[Any]:
    vectors = []
    for sparse_embedding in sparse_model.embed(query_texts):
        vectors.append(
            models.SparseVector(
                indices=[int(index) for index in sparse_embedding.indices.tolist()],
                values=[float(value) for value in sparse_embedding.values.tolist()],
            )
        )
    return vectors


def search_limit_for_config(config: QdrantSearchConfig) -> int:
    return max(config.top_k, config.prefetch_limit)


def retrieval_limit_for_config(config: QdrantSearchConfig, search_limit: int | None = None) -> int:
    search_limit = search_limit if search_limit is not None else search_limit_for_config(config)
    if config.rerank or config.exclude_local_documents or config.require_article:
        return search_limit
    return config.top_k


def build_qdrant_filter(config: QdrantSearchConfig, models: Any) -> Any | None:
    conditions = []
    if config.topic_title:
        conditions.append(
            models.FieldCondition(
                key="topic_title",
                match=models.MatchValue(value=config.topic_title.strip()),
            )
        )
    if config.subject_title:
        conditions.append(
            models.FieldCondition(
                key="subject_title",
                match=models.MatchValue(value=config.subject_title.strip()),
            )
        )
    if config.source_law_id:
        conditions.append(
            models.FieldCondition(
                key="source_law_id_candidates",
                match=models.MatchValue(value=config.source_law_id.strip().upper()),
            )
        )
    if config.source_article_no:
        conditions.append(
            models.FieldCondition(
                key="source_article_no_candidates",
                match=models.MatchValue(value=config.source_article_no.strip()),
            )
        )
    if config.citation_confidence:
        conditions.append(
            models.FieldCondition(
                key="citation_confidence",
                match=models.MatchValue(value=config.citation_confidence.strip()),
            )
        )
    if config.topic_number is not None:
        conditions.append(
            models.FieldCondition(
                key="topic_number",
                range=models.Range(gte=config.topic_number, lte=config.topic_number),
            )
        )
    if not conditions:
        return None
    return models.Filter(must=conditions)


def fuse_ranked_points(dense_hits: list[Any], sparse_hits: list[Any], limit: int) -> list[dict[str, Any]]:
    combined: dict[str, dict[str, Any]] = {}
    _merge_hits(combined, dense_hits, source="dense")
    _merge_hits(combined, sparse_hits, source="sparse")
    ranked = sorted(
        combined.values(),
        key=lambda item: (
            -item["score"],
            item["dense_rank"] if item["dense_rank"] is not None else 10**9,
            item["sparse_rank"] if item["sparse_rank"] is not None else 10**9,
            str(item["id"]),
        ),
    )
    return ranked[:limit]


def filter_eligible_results(config: QdrantSearchConfig, results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        item
        for item in results
        if payload_is_submission_eligible(
            item.get("payload") or {},
            doc_title_format=config.doc_title_format,
            exclude_local_documents=config.exclude_local_documents,
            require_article=config.require_article,
        )
    ]


def rank_output_results(results: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    output = []
    for rank, item in enumerate(results[:limit], start=1):
        ranked_item = dict(item)
        ranked_item["rank"] = rank
        output.append(ranked_item)
    return output


def maybe_rerank_results(
    config: QdrantSearchConfig,
    query_text: str,
    results: list[dict[str, Any]],
    *,
    reranker: Any | None = None,
    query_id: Any | None = None,
) -> list[dict[str, Any]]:
    raw_count = len(results)
    trace_search_results(config, "rerank_input_raw", results, query_id=query_id, query_text=query_text)
    results = filter_eligible_results(config, results)
    trace_search_results(
        config,
        "rerank_input_eligible",
        results,
        query_id=query_id,
        query_text=query_text,
        raw_count=raw_count,
        filtered_out=raw_count - len(results),
    )
    if not config.rerank:
        output = rank_output_results(results, config.top_k)
        trace_search_results(config, "rerank_skipped_output", output, query_id=query_id, top_k=config.top_k)
        return output
    if not results:
        trace_log(
            config,
            "rerank_empty",
            query_id=query_id,
            query_text=query_text,
            raw_count=raw_count,
            filtered_out=raw_count,
        )
        return []
    if reranker is None:
        reranker = load_reranker(
            config.reranker_model_name,
            config.model_cache_dir,
            use_jina_reranker=config.use_jina_reranker,
        )
    trace_log(
        config,
        "rerank_start",
        query_id=query_id,
        query_text=query_text,
        candidate_count=len(results),
        backend=reranker.get("backend") if isinstance(reranker, dict) else None,
        model=config.reranker_model_name,
        max_length=None if config.use_jina_reranker else config.reranker_max_length,
    )
    documents = [result_text_for_reranking(item) for item in results]
    scores = score_rerank_documents(
        query_text=query_text,
        documents=documents,
        reranker=reranker,
        max_length=config.reranker_max_length,
    )
    reranked = []
    for item, score in zip(results, scores, strict=True):
        reranked_item = dict(item)
        reranked_item["retrieval_rank"] = item.get("rank")
        reranked_item["retrieval_score"] = item.get("score")
        reranked_item["rerank_score"] = float(score)
        reranked_item["score"] = float(score)
        reranked.append(reranked_item)
    reranked.sort(
        key=lambda item: (
            -item["rerank_score"],
            item["retrieval_rank"] if item["retrieval_rank"] is not None else 10**9,
            str(item["id"]),
        )
    )
    if config.rerank_threshold is not None:
        reranked = [
            item for item in reranked
            if item["rerank_score"] >= config.rerank_threshold
        ]
    for rank, item in enumerate(reranked, start=1):
        item["rank"] = rank
        item["rerank_rank"] = rank
    output = reranked[: config.top_k]
    trace_search_results(
        config,
        "rerank_output",
        output,
        query_id=query_id,
        query_text=query_text,
        scored_count=len(reranked),
        threshold=config.rerank_threshold,
    )
    return output


def load_reranker(
    model_name: str,
    cache_dir: Path | None,
    *,
    use_jina_reranker: bool = False,
) -> dict[str, Any]:
    try:
        from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("transformers is required for reranking. Install the search extra.") from exc

    kwargs: dict[str, Any] = {"cache_dir": str(cache_dir)} if cache_dir else {}
    if use_jina_reranker and model_name == DEFAULT_RERANKER_MODEL:
        model_name = DEFAULT_JINA_RERANKER_MODEL
    if use_jina_reranker:
        model = AutoModel.from_pretrained(
            model_name,
            dtype="auto",
            trust_remote_code=True,
            **kwargs,
        )
        if hasattr(model, "eval"):
            model.eval()
        device = preferred_torch_device()
        if device and hasattr(model, "to"):
            model.to(device)
        return {"backend": "jina", "model": model}

    try:
        import sentencepiece  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "sentencepiece is required for the Vietnamese reranker tokenizer. "
            "Run `uv sync --extra search` or install `sentencepiece` in the current environment."
        ) from exc

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False, **kwargs)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, **kwargs)
    if hasattr(model, "eval"):
        model.eval()
    device = preferred_torch_device()
    if device and hasattr(model, "to"):
        model.to(device)
    return {"backend": "cross_encoder", "tokenizer": tokenizer, "model": model}


def score_rerank_documents(
    *,
    query_text: str,
    documents: list[str],
    reranker: dict[str, Any],
    max_length: int,
) -> list[float]:
    if reranker.get("backend") == "jina":
        return score_jina_rerank_documents(reranker["model"], query_text, documents)
    pairs = [[query_text, document] for document in documents]
    return score_rerank_pairs(
        tokenizer=reranker["tokenizer"],
        model=reranker["model"],
        pairs=pairs,
        max_length=max_length,
    )


def score_jina_rerank_documents(model: Any, query_text: str, documents: list[str]) -> list[float]:
    if not documents:
        return []
    results = model.rerank(query_text, documents)
    scores = [0.0] * len(documents)
    used: set[int] = set()
    for fallback_index, result in enumerate(results):
        if not isinstance(result, dict):
            continue
        raw_score = result.get("relevance_score", result.get("score", 0.0))
        index = first_jina_result_index(result)
        if index is None:
            index = first_unused_document_index(documents, result.get("document"), used)
        if index is None and fallback_index < len(documents) and fallback_index not in used:
            index = fallback_index
        if index is None or index in used or index >= len(documents):
            continue
        scores[index] = float(raw_score)
        used.add(index)
    return scores


def first_jina_result_index(result: dict[str, Any]) -> int | None:
    for key in ("index", "corpus_id", "doc_id", "document_index"):
        value = result.get(key)
        if isinstance(value, int):
            return value
        if isinstance(value, str) and value.isdigit():
            return int(value)
    return None


def first_unused_document_index(documents: list[str], document: Any, used: set[int]) -> int | None:
    if document is None:
        return None
    text = str(document)
    for index, candidate in enumerate(documents):
        if index not in used and candidate == text:
            return index
    return None


def score_rerank_pairs(tokenizer: Any, model: Any, pairs: list[list[str]], max_length: int) -> list[float]:
    try:
        import torch
    except ImportError:
        torch = None
    no_grad = torch.no_grad() if torch is not None else contextlib.nullcontext()
    with no_grad:
        inputs = tokenizer(
            pairs,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=max_length,
        )
        device = model_device(model)
        if device is not None:
            inputs = {
                key: value.to(device) if hasattr(value, "to") else value
                for key, value in inputs.items()
            }
        outputs = model(**inputs, return_dict=True)
    logits = outputs.get("logits") if isinstance(outputs, dict) else getattr(outputs, "logits")
    try:
        values = logits.view(-1).float().detach().cpu().tolist()
    except AttributeError:
        values = list(logits)
    return [float(value) for value in values]


def result_text_for_reranking(item: dict[str, Any]) -> str:
    payload = item.get("payload") or {}
    retrieval_text = str(payload.get("retrieval_text") or "").strip()
    if retrieval_text:
        return retrieval_text
    fallback_parts = []
    for key in ("article_title", "source_doc_title_candidates", "source_article_no_candidates"):
        fallback_parts.extend(as_text_list(payload.get(key)))
    return "\n".join(fallback_parts)


def preferred_torch_device() -> str | None:
    try:
        import torch
    except ImportError:
        return None
    return "cuda" if torch.cuda.is_available() else "cpu"


def model_device(model: Any) -> Any | None:
    try:
        return next(model.parameters()).device
    except (AttributeError, StopIteration, TypeError):
        return None


def ranked_points_to_results(hits: list[Any], source: str) -> list[dict[str, Any]]:
    results = []
    for rank, point in enumerate(hits, start=1):
        payload = _point_attr(point, "payload", {}) or {}
        item = {
            "id": str(_point_attr(point, "id")),
            "rank": rank,
            "score": _point_attr(point, "score"),
            "payload": payload,
        }
        if source == "dense":
            item.update({"dense_rank": rank, "dense_score": item["score"], "sparse_rank": None, "sparse_score": None})
        elif source == "sparse":
            item.update({"dense_rank": None, "dense_score": None, "sparse_rank": rank, "sparse_score": item["score"]})
        else:
            item.update({"hybrid_rank": rank, "hybrid_score": item["score"]})
        results.append(item)
    return results


def payload_is_submission_eligible(
    payload: dict[str, Any],
    *,
    doc_title_format: str = "type1",
    exclude_local_documents: bool = True,
    require_article: bool = True,
) -> bool:
    if exclude_local_documents and is_local_document_payload(payload):
        return False
    if require_article and not payload_to_competition_articles(payload, doc_title_format=doc_title_format):
        return False
    return True


def is_local_document_payload(payload: dict[str, Any]) -> bool:
    law_ids = payload_law_ids(payload)
    if any(LOCAL_LAW_ID_PATTERN.search(law_id) for law_id in law_ids):
        return True
    if any(CENTRAL_LAW_ID_PATTERN.search(law_id) for law_id in law_ids):
        return False

    authority_text = normalize_spaces(
        " ".join(
            part
            for key in ("issuing_authority", "source_authority", "authority")
            for part in as_text_list(payload.get(key))
        )
    )
    if authority_text and LOCAL_AUTHORITY_PATTERN.search(authority_text):
        return True

    title_text = normalize_spaces(
        " ".join(
            part
            for key in (
                "title",
                "document_title",
                "source_document_title",
                "source_note_text",
                "source_doc_title_candidates",
            )
            for part in as_text_list(payload.get(key))
        )
    )
    return bool(title_text and (LOCAL_AUTHORITY_PATTERN.search(title_text) or LOCAL_TITLE_PATTERN.search(title_text)))


def payload_output_law_ids(payload: dict[str, Any]) -> list[str]:
    law_ids: list[str] = []
    for key in (
        "competition_law_id",
        "source_document_number",
        "document_number",
        "law_id",
        "source_law_id_candidates",
    ):
        law_ids.extend(as_text_list(payload.get(key)))
    output = []
    seen = set()
    for law_id in law_ids:
        normalized = normalize_law_id_for_output(law_id)
        lookup = normalized.upper()
        if normalized and lookup not in seen:
            seen.add(lookup)
            output.append(normalized)
    return output


def normalize_law_id_for_output(law_id: str) -> str:
    normalized = normalize_spaces(law_id)
    normalized = re.sub(r"ND-CP\b", "NĐ-CP", normalized, flags=re.IGNORECASE)
    normalized = re.sub(r"QD-", "QĐ-", normalized, flags=re.IGNORECASE)
    return normalized


def payload_law_ids(payload: dict[str, Any]) -> list[str]:
    law_ids: list[str] = []
    for key in (
        "competition_law_id",
        "source_document_number",
        "document_number",
        "law_id",
        "source_law_id_candidates",
    ):
        law_ids.extend(as_text_list(payload.get(key)))
    output = []
    seen = set()
    for law_id in law_ids:
        normalized = law_id.strip().upper().replace("ND-CP", "NĐ-CP").replace("QD-", "QĐ-")
        if normalized and normalized not in seen:
            seen.add(normalized)
            output.append(normalized)
    return output


def format_competition_row(question: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
    relevant_docs: list[str] = []
    relevant_articles: list[str] = []
    answer_articles: list[str] = []
    seen_docs: set[str] = set()
    seen_articles: set[str] = set()
    doc_title_format = result.get("doc_title_format", "type1")
    answer_article_limit = result.get("answer_article_limit")
    exclude_local_documents = result.get("exclude_local_documents", True)
    require_article = result.get("require_article", True)

    for item in result["results"]:
        payload = item.get("payload") or {}
        if not payload_is_submission_eligible(
            payload,
            doc_title_format=doc_title_format,
            exclude_local_documents=exclude_local_documents,
            require_article=require_article,
        ):
            continue
        for doc in payload_to_competition_docs(payload, doc_title_format=doc_title_format):
            if doc not in seen_docs:
                seen_docs.add(doc)
                relevant_docs.append(doc)
        for article in payload_to_competition_articles(payload, doc_title_format=doc_title_format):
            if article not in seen_articles:
                seen_articles.add(article)
                relevant_articles.append(article)
                answer_article = article.rsplit("|", maxsplit=1)[-1]
                if answer_article not in answer_articles and (
                    answer_article_limit is None or len(answer_articles) < answer_article_limit
                ):
                    answer_articles.append(answer_article)

    row = {
        "id": question["id"],
        "question": question["question"],
        "answer": build_retrieval_only_answer(answer_articles),
        "relevant_docs": relevant_docs,
        "relevant_articles": relevant_articles,
    }
    trace_submission_row(result, row)
    return row


def payload_to_competition_docs(payload: dict[str, Any], doc_title_format: str = "type1") -> list[str]:
    law_id, doc_title, _ = payload_to_primary_competition_ref(payload, doc_title_format=doc_title_format)
    if not law_id or not doc_title:
        return []
    return [f"{law_id}|{doc_title}"]


def payload_to_competition_articles(payload: dict[str, Any], doc_title_format: str = "type1") -> list[str]:
    law_id, doc_title, article_no = payload_to_primary_competition_ref(payload, doc_title_format=doc_title_format)
    if not law_id or not doc_title or not article_no:
        return []
    return [f"{law_id}|{doc_title}|{article_no}"]


def payload_to_primary_competition_ref(
    payload: dict[str, Any],
    doc_title_format: str = "type1",
) -> tuple[str, str, str]:
    explicit_law_id = str(payload.get("competition_law_id") or "").strip()
    explicit_doc_title = str(payload.get(f"competition_doc_title_{doc_title_format}") or "").strip()
    explicit_article_no = str(payload.get("competition_article_no") or "").strip()
    if explicit_law_id and explicit_doc_title:
        doc_title = normalize_competition_doc_title(explicit_doc_title, explicit_law_id, doc_title_format)
        return explicit_law_id, doc_title, explicit_article_no

    header = retrieval_header(payload)
    law_ids = payload_output_law_ids(payload)
    law_id = explicit_law_id or first_payload_law_id(payload) or first_header_law_id(header, law_ids)
    article_no = explicit_article_no or first_payload_article_no(payload) or first_article_no(header)
    doc_title = explicit_doc_title or first_reasonable_doc_title(payload) or doc_title_from_header(header, article_no)
    doc_title = normalize_competition_doc_title(doc_title, law_id, doc_title_format)
    return law_id, doc_title, article_no


def first_payload_law_id(payload: dict[str, Any]) -> str:
    law_ids = payload_output_law_ids(payload)
    return law_ids[0] if law_ids else ""


def first_payload_article_no(payload: dict[str, Any]) -> str:
    for key in (
        "competition_article_no",
        "article_no_normalized",
        "article_no",
        "source_article_no",
        "source_article_no_candidates",
        "content_tree_path",
        "article_title",
    ):
        article_no = first_article_no(" ".join(as_text_list(payload.get(key))))
        if article_no:
            return article_no
    return ""


def retrieval_header(payload: dict[str, Any]) -> str:
    retrieval_text = str(payload.get("retrieval_text") or "")
    return retrieval_text.split(":\n", maxsplit=1)[0].strip()


def first_header_law_id(header: str, law_ids: list[str]) -> str:
    for law_id in law_ids:
        if law_id and law_id in header:
            return law_id
    return law_ids[0] if law_ids else ""


def first_article_no(text: str) -> str:
    match = ARTICLE_NO_PATTERN.search(text)
    return match.group(0).replace("điều", "Điều") if match else ""


def doc_title_from_header(header: str, article_no: str) -> str:
    if not header:
        return ""
    if article_no:
        split_match = HEADER_ARTICLE_SPLIT_PATTERN.search(header)
        if split_match:
            return header[: split_match.start()].strip(" -:;,.")
    return header.strip(" -:;,.")


def first_reasonable_doc_title(payload: dict[str, Any]) -> str:
    for key in (
        "source_doc_title_candidates",
        "source_document_title",
        "document_title",
        "title",
        "source_note_text",
    ):
        for title in as_text_list(payload.get(key)):
            title = title.strip("() ")
            if not ARTICLE_NO_PATTERN.fullmatch(title) and len(title) > 12:
                return title
    return ""


def normalize_competition_doc_title(doc_title: str, law_id: str, doc_title_format: str) -> str:
    doc_title = strip_doc_title_path(normalize_spaces(doc_title))
    match = DOC_TITLE_CODE_PATTERN.match(doc_title)
    if match:
        kind = normalize_spaces(match.group("kind"))
        code = normalize_spaces(match.group("code"))
        title = strip_repeated_source_type(
            uppercase_first(strip_doc_title_path(normalize_spaces(match.group("title")))), kind
        )
        if doc_title_format == "type2":
            return normalize_spaces(f"{kind} {code} {title}")
        return normalize_spaces(f"{kind} {title}")
    doc_title = re.sub(r"\bsố\s+", "", doc_title, flags=re.IGNORECASE)
    doc_title = normalize_spaces(doc_title)
    if law_id and law_id in doc_title:
        prefix, title = doc_title.split(law_id, maxsplit=1)
        kind = normalize_spaces(prefix.strip(" ,.;:-"))
        title = strip_repeated_source_type(
            uppercase_first(strip_doc_title_path(normalize_spaces(title.strip(" ,.;:-")))), kind
        )
        if kind and title:
            if doc_title_format == "type2":
                return normalize_spaces(f"{kind} {law_id} {title}")
            return normalize_spaces(f"{kind} {title}")
        if title:
            source_type, source_title = split_source_type(title)
            if source_type and source_title:
                if doc_title_format == "type2":
                    return normalize_spaces(f"{source_type} {law_id} {uppercase_first(source_title)}")
                return normalize_spaces(f"{source_type} {uppercase_first(source_title)}")
            if doc_title_format == "type1":
                return title
    if doc_title_format == "type2" and law_id and law_id not in doc_title:
        source_type, source_title = split_source_type(doc_title)
        if source_type and source_title:
            return normalize_spaces(f"{source_type} {law_id} {uppercase_first(source_title)}")
        first_word, rest = split_first_word(doc_title)
        if rest:
            return normalize_spaces(f"{first_word} {law_id} {uppercase_first(rest)}")
    return doc_title


def strip_doc_title_path(doc_title: str) -> str:
    match = DOC_TITLE_PATH_SEGMENT_PATTERN.search(doc_title)
    cut_at = match.start() if match else -1
    pipe_at = doc_title.find(" | ")
    if pipe_at >= 0 and (cut_at < 0 or pipe_at < cut_at):
        cut_at = pipe_at
    if cut_at < 0:
        return doc_title
    return normalize_spaces(doc_title[:cut_at].strip(" .;:-|"))


def split_source_type(doc_title: str) -> tuple[str, str]:
    match = DOC_TITLE_SOURCE_TYPE_PATTERN.match(doc_title)
    if not match:
        return "", doc_title
    source_type = normalize_spaces(match.group(0))
    source_title = normalize_spaces(doc_title[match.end() :].strip(" ,.;:-"))
    return source_type, source_title


def normalize_spaces(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def strip_repeated_source_type(title: str, source_type: str) -> str:
    title = normalize_spaces(title)
    source_type = normalize_spaces(source_type)
    if source_type and title.lower().startswith(source_type.lower() + " "):
        return normalize_spaces(title[len(source_type) :].strip(" ,.;:-"))
    return title


def uppercase_first(text: str) -> str:
    if not text:
        return text
    return text[0].upper() + text[1:]


def split_first_word(text: str) -> tuple[str, str]:
    parts = text.split(" ", maxsplit=1)
    if len(parts) == 1:
        return parts[0], ""
    return parts[0], parts[1]


def build_retrieval_only_answer(article_numbers: list[str]) -> str:
    if not article_numbers:
        return "Chưa xác định được điều luật liên quan từ kết quả truy hồi."
    joined = "; ".join(article_numbers)
    return f"Các căn cứ pháp luật liên quan được hệ thống truy hồi gồm: {joined}."


def as_text_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    text = str(value).strip()
    return [text] if text else []


def summarize_filters(config: QdrantSearchConfig) -> dict[str, Any]:
    return {
        "topic_title": config.topic_title,
        "subject_title": config.subject_title,
        "source_law_id": config.source_law_id,
        "source_article_no": config.source_article_no,
        "citation_confidence": config.citation_confidence,
        "topic_number": config.topic_number,
        "exclude_local_documents": config.exclude_local_documents,
        "require_article": config.require_article,
    }


def normalize_search_mode(search_mode: str) -> str:
    mode = search_mode.strip().lower()
    if mode not in SEARCH_MODES:
        raise ValueError(f"search_mode must be one of {sorted(SEARCH_MODES)}, got {search_mode!r}.")
    return mode


def has_server_side_rrf(models: Any) -> bool:
    return hasattr(models, "Prefetch") and (
        (hasattr(models, "RrfQuery") and hasattr(models, "Rrf"))
        or (hasattr(models, "FusionQuery") and hasattr(models, "Fusion"))
    )


def rrf_query(models: Any) -> Any:
    if hasattr(models, "RrfQuery") and hasattr(models, "Rrf"):
        return models.RrfQuery(rrf=models.Rrf())
    return models.FusionQuery(fusion=models.Fusion.RRF)


def first_present(row: dict[str, Any], keys: tuple[str, ...], default: Any = "") -> Any:
    for key in keys:
        value = row.get(key)
        if value is not None and value != "":
            return value
    return default


def _query_points(client: Any, *, label: str, **kwargs: Any) -> Any:
    return retry_request(lambda: client.query_points(**kwargs), label=label)


def _query_batch_points(
    client: Any,
    *,
    collection_name: str,
    requests: list[Any],
    label: str,
    allow_empty_on_failure: bool = False,
) -> list[Any | None]:
    try:
        return list(
            retry_request(
                lambda: client.query_batch_points(collection_name=collection_name, requests=requests),
                label=label,
            )
        )
    except Exception as exc:
        if len(requests) > 1 and is_retryable_request_error(exc):
            midpoint = len(requests) // 2
            print(f"{label} still failed after retries; splitting into smaller batches", flush=True)
            return _query_batch_points(
                client,
                collection_name=collection_name,
                requests=requests[:midpoint],
                label=f"{label} left",
                allow_empty_on_failure=allow_empty_on_failure,
            ) + _query_batch_points(
                client,
                collection_name=collection_name,
                requests=requests[midpoint:],
                label=f"{label} right",
                allow_empty_on_failure=allow_empty_on_failure,
            )
        if allow_empty_on_failure and is_retryable_request_error(exc):
            print(f"{label} failed after exhausted request retries; continuing with empty result: {exc}", flush=True)
            return [None for _ in requests]
        raise


def _merge_hits(combined: dict[str, dict[str, Any]], hits: list[Any], source: str) -> None:
    for rank, point in enumerate(hits, start=1):
        point_id = str(_point_attr(point, "id"))
        item = combined.setdefault(
            point_id,
            {
                "id": point_id,
                "rank": rank,
                "score": 0.0,
                "dense_rank": None,
                "dense_score": None,
                "sparse_rank": None,
                "sparse_score": None,
                "payload": _point_attr(point, "payload", {}),
            },
        )
        item["score"] += 1.0 / (RRF_K + rank)
        item["rank"] = min(item["rank"], rank)
        item[f"{source}_rank"] = rank
        item[f"{source}_score"] = _point_attr(point, "score")
        if not item["payload"]:
            item["payload"] = _point_attr(point, "payload", {})


def _response_points(response: Any) -> list[Any]:
    if response is None:
        return []
    if isinstance(response, list):
        return response
    if isinstance(response, dict):
        points = response.get("points")
        if points is None:
            result = response.get("result")
            points = result.get("points") if isinstance(result, dict) else None
        if points is None:
            return []
        return list(points)
    points = getattr(response, "points", None)
    if points is None:
        return []
    return list(points)


def _point_attr(point: Any, name: str, default: Any = None) -> Any:
    if isinstance(point, dict):
        return point.get(name, default)
    return getattr(point, name, default)
