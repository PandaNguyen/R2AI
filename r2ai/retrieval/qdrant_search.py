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
    DEFAULT_RRF_WEIGHTS,
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
VBHN_ID_PATTERN = re.compile(r"\b\d+\s*/\s*VBHN-[A-ZĐ]+\b|\bVBHN-[A-ZĐ]+\b", re.IGNORECASE)
VBHN_PREFIX_PATTERN = re.compile(r"^văn bản hợp nhất\b", re.IGNORECASE)
DOC_YEAR_PATTERN = re.compile(r"\bnăm\s+(?:19|20)\d{2}\b", re.IGNORECASE)
DOC_DATE_PATTERN = re.compile(
    r"\b(?:ngày\s+)?\d{1,2}[/-]\d{1,2}[/-](?:19|20)\d{2}\b|"
    r"\bngày\s+\d{1,2}\s+tháng\s+\d{1,2}\s+năm\s+(?:19|20)\d{2}\b",
    re.IGNORECASE,
)
DOC_CODE_PATTERN = re.compile(r"\b\d+\s*/\s*[A-ZĐ][A-ZĐ0-9/-]*\b", re.IGNORECASE)
OutputFormat = Literal["json", "jsonl"]
TRACE_SAMPLE_LIMIT = 5
LLM_SELECTOR_SYSTEM_PROMPT = "Bạn là trợ lý chọn candidate RAG tiếng Việt. Luôn trả lời bằng JSON hợp lệ."


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
    llm_selector: Any | None = None,
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
        rrf_weights=list(config.rrf_weights) if mode == "hybrid" else None,
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
    elif has_server_side_rrf(models, config.rrf_weights):
        hits = _response_points(
            _query_points(
                client,
                label="Qdrant hybrid query",
                collection_name=config.collection_name,
                prefetch=[
                    models.Prefetch(query=sparse_vector, using=config.sparse_vector_name, limit=search_limit),
                    models.Prefetch(query=dense_vector, using=config.dense_vector_name, limit=search_limit),
                ],
                query=rrf_query(models, config.rrf_weights),
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
        results = fuse_ranked_points(
            dense_hits=dense_hits,
            sparse_hits=sparse_hits,
            limit=retrieval_limit,
            rrf_weights=config.rrf_weights,
        )

    trace_search_results(config, "search_raw_results", results, retrieval_limit=retrieval_limit)
    results = maybe_rerank_results(config, query_text, results, reranker=reranker)
    results = maybe_select_llm_submission_candidates(config, query_text, results, llm_selector=llm_selector)
    trace_search_results(config, "search_final_results", results, top_k=config.top_k)

    return {
        "collection_name": config.collection_name,
        "query_text": query_text,
        "search_mode": mode,
        "top_k": config.top_k,
        "prefetch_limit": search_limit,
        "rrf_weights": list(config.rrf_weights) if mode == "hybrid" else None,
        "doc_title_format": config.doc_title_format,
        "answer_article_limit": config.answer_article_limit,
        "exclude_local_documents": config.exclude_local_documents,
        "require_article": config.require_article,
        "trace_search": config.trace_search,
        "keep_candidate_pool": config.keep_candidate_pool,
        "llm_select_candidates": config.llm_select_candidates,
        "llm_selector_model": config.llm_selector_model_name if config.llm_select_candidates else None,
        "llm_selector_max_candidates": config.llm_selector_max_candidates if config.llm_select_candidates else None,
        "llm_selector_context_window": config.llm_selector_context_window if config.llm_select_candidates else None,
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
    llm_selector: Any | None = None,
    client: Any | None = None,
    models: Any | None = None,
    progress_every: int = 0,
    query_batch_size: int = 64,
    checkpoint_path: Path | None = None,
    output_search_results: bool = False,
) -> list[dict[str, Any]]:
    """Run retrieval for many questions, optionally with precomputed dense vectors."""
    if output_search_results and not config.keep_candidate_pool:
        config = replace(config, keep_candidate_pool=True, llm_select_candidates=False)
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
    if config.llm_select_candidates and not output_search_results and llm_selector is None:
        llm_selector = load_llm_candidate_selector(config.llm_selector_model_name, config.model_cache_dir)
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
            llm_selector=llm_selector,
            progress_every=progress_every,
            query_batch_size=query_batch_size,
            checkpoint_path=checkpoint_path,
            output_search_results=output_search_results,
        )
        completed_rows.update({str(row["id"]): row for row in new_rows})
        return ordered_checkpoint_rows(questions, completed_rows)
    if (
        mode in {"bm25", "hybrid"}
        and query_batch_size > 1
        and hasattr(client, "query_batch_points")
        and hasattr(models, "QueryRequest")
        and (mode == "bm25" or (pending_vectors is not None and has_server_side_rrf(models, config.rrf_weights)))
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
            llm_selector=llm_selector,
            progress_every=progress_every,
            query_batch_size=query_batch_size,
            checkpoint_path=checkpoint_path,
            output_search_results=output_search_results,
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
                llm_selector=llm_selector,
                client=client,
                models=models,
            )
        except Exception as exc:
            if not is_retryable_request_error(exc):
                raise
            print(f"Skipping question {question['id']} after exhausted request retries: {exc}", flush=True)
            result = empty_search_result(config, question["question"], mode)
        row = format_search_result_row(question, result) if output_search_results else format_competition_row(question, result)
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
    llm_selector: Any | None = None,
    progress_every: int = 0,
    query_batch_size: int = 64,
    checkpoint_path: Path | None = None,
    output_search_results: bool = False,
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
            results = maybe_rerank_results(
                config,
                question["question"],
                ranked_points_to_results(_response_points(response), source="dense"),
                reranker=reranker,
                query_id=question["id"],
            )
            if not output_search_results:
                results = maybe_select_llm_submission_candidates(
                    config,
                    question["question"],
                    results,
                    llm_selector=llm_selector,
                    query_id=question["id"],
                )
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
                "keep_candidate_pool": config.keep_candidate_pool,
                "llm_select_candidates": config.llm_select_candidates,
                "llm_selector_model": config.llm_selector_model_name if config.llm_select_candidates else None,
                "llm_selector_max_candidates": config.llm_selector_max_candidates if config.llm_select_candidates else None,
                "llm_selector_context_window": config.llm_selector_context_window if config.llm_select_candidates else None,
                "used_precomputed_dense_vector": True,
                "rerank": config.rerank,
                "reranker_model": config.reranker_model_name if config.rerank else None,
                "reranker_backend": "jina" if config.rerank and config.use_jina_reranker else ("cross_encoder" if config.rerank else None),
                "reranker_max_length": None if config.rerank and config.use_jina_reranker else (config.reranker_max_length if config.rerank else None),
                "rerank_threshold": config.rerank_threshold if config.rerank else None,
                "filters": summarize_filters(config),
                "results": results,
            }
            row = format_search_result_row(question, result) if output_search_results else format_competition_row(question, result)
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
    llm_selector: Any | None = None,
    progress_every: int = 0,
    query_batch_size: int = 64,
    checkpoint_path: Path | None = None,
    output_search_results: bool = False,
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
                        query=rrf_query(models, config.rrf_weights),
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
            results = maybe_rerank_results(
                config,
                question["question"],
                ranked_points_to_results(
                    _response_points(response),
                    source="sparse" if mode == "bm25" else "hybrid",
                ),
                reranker=reranker,
                query_id=question["id"],
            )
            if not output_search_results:
                results = maybe_select_llm_submission_candidates(
                    config,
                    question["question"],
                    results,
                    llm_selector=llm_selector,
                    query_id=question["id"],
                )
            result = {
                "collection_name": config.collection_name,
                "query_text": question["question"],
                "search_mode": mode,
                "top_k": config.top_k,
                "prefetch_limit": search_limit,
                "rrf_weights": list(config.rrf_weights) if mode == "hybrid" else None,
                "doc_title_format": config.doc_title_format,
                "answer_article_limit": config.answer_article_limit,
                "exclude_local_documents": config.exclude_local_documents,
                "require_article": config.require_article,
                "trace_search": config.trace_search,
                "keep_candidate_pool": config.keep_candidate_pool,
                "llm_select_candidates": config.llm_select_candidates,
                "llm_selector_model": config.llm_selector_model_name if config.llm_select_candidates else None,
                "llm_selector_max_candidates": config.llm_selector_max_candidates if config.llm_select_candidates else None,
                "llm_selector_context_window": config.llm_selector_context_window if config.llm_select_candidates else None,
                "used_precomputed_dense_vector": query_vectors is not None,
                "rerank": config.rerank,
                "reranker_model": config.reranker_model_name if config.rerank else None,
                "reranker_backend": "jina" if config.rerank and config.use_jina_reranker else ("cross_encoder" if config.rerank else None),
                "reranker_max_length": None if config.rerank and config.use_jina_reranker else (config.reranker_max_length if config.rerank else None),
                "rerank_threshold": config.rerank_threshold if config.rerank else None,
                "filters": summarize_filters(config),
                "results": results,
            }
            row = format_search_result_row(question, result) if output_search_results else format_competition_row(question, result)
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


def format_search_result_row(question: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": question["id"],
        "question": question["question"],
        "result": result,
    }


def load_search_result_rows(path: Path) -> list[dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        with path.open("r", encoding="utf-8") as handle:
            rows = [json.loads(line) for line in handle if line.strip()]
    elif suffix == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        rows = data.get("results", data.get("rows", data)) if isinstance(data, dict) else data
    else:
        raise ValueError(f"Unsupported candidate result file type: {path.suffix}")
    if not isinstance(rows, list):
        raise ValueError("Candidate result file must contain a JSON list or JSONL rows.")
    return [normalize_search_result_row(row, index) for index, row in enumerate(rows)]


def normalize_search_result_row(row: dict[str, Any], index: int) -> dict[str, Any]:
    if not isinstance(row, dict):
        raise ValueError(f"Candidate result row {index} must be an object.")
    if "result" in row:
        result = row.get("result") or {}
        question_text = str(row.get("question") or result.get("query_text") or "")
        question_id = row.get("id", index)
        if not question_text:
            raise ValueError(f"Candidate result row {index} is missing question text.")
        return {"id": question_id, "question": question_text, "result": result}
    if "results" in row:
        question_text = str(row.get("question") or row.get("query_text") or "")
        question_id = row.get("id", index)
        if not question_text:
            raise ValueError(f"Candidate result row {index} is missing question text.")
        return {"id": question_id, "question": question_text, "result": row}
    raise ValueError(f"Candidate result row {index} is missing a result/results field.")


def select_submission_rows_from_search_results(
    search_result_rows: list[dict[str, Any]],
    config: QdrantSearchConfig,
    *,
    llm_selector: Any | None = None,
    progress_every: int = 0,
) -> list[dict[str, Any]]:
    if llm_selector is None:
        llm_selector = load_llm_candidate_selector(config.llm_selector_model_name, config.model_cache_dir)
    rows = []
    started_at = time.monotonic()
    total = len(search_result_rows)
    for index, search_row in enumerate(search_result_rows):
        result = dict(search_row["result"])
        question = {"id": search_row["id"], "question": search_row["question"]}
        selector_config = selector_config_for_search_result(config, question, result)
        selected_results = maybe_select_llm_submission_candidates(
            selector_config,
            question["question"],
            list(result.get("results") or []),
            llm_selector=llm_selector,
            query_id=question["id"],
        )
        result.update(
            {
                "query_text": question["question"],
                "results": selected_results,
                "llm_select_candidates": True,
                "llm_selector_model": selector_config.llm_selector_model_name,
                "llm_selector_max_candidates": selector_config.llm_selector_max_candidates,
                "llm_selector_context_window": selector_config.llm_selector_context_window,
            }
        )
        rows.append(format_competition_row(question, result))
        done = index + 1
        if progress_every > 0 and (done == 1 or done == total or done % progress_every == 0):
            elapsed = time.monotonic() - started_at
            print(f"Selected candidates for {done}/{total} questions in {elapsed:.1f}s", flush=True)
    return rows


def selector_config_for_search_result(
    config: QdrantSearchConfig,
    question: dict[str, Any],
    result: dict[str, Any],
) -> QdrantSearchConfig:
    return replace(
        config,
        query_text=str(question["question"]),
        top_k=int(result.get("top_k") or config.top_k),
        doc_title_format=str(result.get("doc_title_format") or config.doc_title_format),
        answer_article_limit=result.get("answer_article_limit", config.answer_article_limit),
        exclude_local_documents=bool(result.get("exclude_local_documents", config.exclude_local_documents)),
        require_article=bool(result.get("require_article", config.require_article)),
        trace_search=bool(result.get("trace_search", config.trace_search)),
        keep_candidate_pool=False,
        llm_select_candidates=True,
        llm_selector_context_window=int(result.get("llm_selector_context_window") or config.llm_selector_context_window),
    )


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
        "rrf_weights": list(config.rrf_weights) if mode == "hybrid" else None,
        "doc_title_format": config.doc_title_format,
        "answer_article_limit": config.answer_article_limit,
        "exclude_local_documents": config.exclude_local_documents,
        "require_article": config.require_article,
        "trace_search": config.trace_search,
        "keep_candidate_pool": config.keep_candidate_pool,
        "llm_select_candidates": config.llm_select_candidates,
        "llm_selector_model": config.llm_selector_model_name if config.llm_select_candidates else None,
        "llm_selector_max_candidates": config.llm_selector_max_candidates if config.llm_select_candidates else None,
        "llm_selector_context_window": config.llm_selector_context_window if config.llm_select_candidates else None,
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


def keep_candidate_pool_for_config(config: QdrantSearchConfig) -> bool:
    return config.keep_candidate_pool or config.llm_select_candidates


def retrieval_limit_for_config(config: QdrantSearchConfig, search_limit: int | None = None) -> int:
    search_limit = search_limit if search_limit is not None else search_limit_for_config(config)
    if keep_candidate_pool_for_config(config) or config.rerank or config.exclude_local_documents or config.require_article:
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


def fuse_ranked_points(
    dense_hits: list[Any],
    sparse_hits: list[Any],
    limit: int,
    rrf_weights: tuple[float, float] = DEFAULT_RRF_WEIGHTS,
) -> list[dict[str, Any]]:
    sparse_weight, dense_weight = rrf_weights
    combined: dict[str, dict[str, Any]] = {}
    _merge_hits(combined, dense_hits, source="dense", weight=dense_weight)
    _merge_hits(combined, sparse_hits, source="sparse", weight=sparse_weight)
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


def deduplicate_results_by_document(
    results: list[dict[str, Any]],
    *,
    doc_title_format: str = "type1",
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    index_by_key: dict[tuple[str, ...], int] = {}
    for item in results:
        key = result_document_article_dedup_key(item, doc_title_format=doc_title_format)
        if key not in index_by_key:
            index_by_key[key] = len(output)
            output.append(item)
            continue
        existing_index = index_by_key[key]
        if should_replace_duplicate_document(
            current=item,
            existing=output[existing_index],
            doc_title_format=doc_title_format,
        ):
            output[existing_index] = item
    return output


def result_document_article_dedup_key(
    item: dict[str, Any],
    *,
    doc_title_format: str = "type1",
) -> tuple[str, ...]:
    payload = item.get("payload") or {}
    law_id, doc_title, article_no = payload_to_primary_competition_ref(payload, doc_title_format=doc_title_format)
    doc_key = document_dedup_key(payload, law_id, doc_title)
    article_key = normalize_dedup_text(article_no)
    if doc_key and article_key:
        return ("article", *doc_key, article_key)
    if doc_key:
        return ("doc", *doc_key)
    return ("point", str(item.get("id", "")))


def should_replace_duplicate_document(
    *,
    current: dict[str, Any],
    existing: dict[str, Any],
    doc_title_format: str = "type1",
) -> bool:
    current_sort = document_recency_sort_key(current, doc_title_format=doc_title_format)
    existing_sort = document_recency_sort_key(existing, doc_title_format=doc_title_format)
    if current_sort != existing_sort:
        return current_sort > existing_sort
    return retrieval_preference_sort_key(current) > retrieval_preference_sort_key(existing)


def document_recency_sort_key(item: dict[str, Any], *, doc_title_format: str = "type1") -> tuple[int, int, int]:
    payload = item.get("payload") or {}
    law_id, doc_title, _ = payload_to_primary_competition_ref(payload, doc_title_format=doc_title_format)
    if not is_vbhn_document(law_id, doc_title, payload):
        return (0, 0, 0)
    return document_issue_date_sort_key(payload, doc_title)


def retrieval_preference_sort_key(item: dict[str, Any]) -> tuple[float, int]:
    score = item.get("rerank_score", item.get("score", 0.0))
    try:
        score_value = float(score)
    except (TypeError, ValueError):
        score_value = 0.0
    rank = item.get("rank")
    try:
        rank_value = int(rank)
    except (TypeError, ValueError):
        rank_value = 10**9
    return (score_value, -rank_value)


def document_issue_date_sort_key(payload: dict[str, Any], doc_title: str) -> tuple[int, int, int]:
    for key in (
        "issuance_date",
        "ngay_ban_hanh",
        "signing_date",
        "signed_date",
        "issue_date",
        "issued_date",
        "promulgation_date",
        "source_document_issuance_date",
    ):
        for value in as_text_list(payload.get(key)):
            parsed = parse_document_date(value)
            if parsed != (0, 0, 0):
                return parsed
    parsed = parse_document_date(doc_title)
    if parsed != (0, 0, 0):
        return parsed
    match = DOC_YEAR_PATTERN.search(doc_title)
    if match:
        return (int(match.group(0).split()[-1]), 0, 0)
    return (0, 0, 0)


def parse_document_date(text: str) -> tuple[int, int, int]:
    normalized = normalize_spaces(str(text or ""))
    match = re.search(r"\b(?:ngày\s+)?(\d{1,2})[/-](\d{1,2})[/-]((?:19|20)\d{2})\b", normalized, re.IGNORECASE)
    if match:
        day, month, year = (int(part) for part in match.groups())
        return year, month, day
    match = re.search(
        r"\bngày\s+(\d{1,2})\s+tháng\s+(\d{1,2})\s+năm\s+((?:19|20)\d{2})\b",
        normalized,
        re.IGNORECASE,
    )
    if match:
        day, month, year = (int(part) for part in match.groups())
        return year, month, day
    return (0, 0, 0)


def document_dedup_key(payload: dict[str, Any], law_id: str, doc_title: str) -> tuple[str, ...]:
    title = doc_title or first_reasonable_doc_title(payload)
    if is_vbhn_document(law_id, title, payload):
        subject_title = normalize_vbhn_subject_title(title)
        if subject_title:
            return ("vbhn", subject_title)
    date = document_issue_date(payload, title)
    normalized_title = normalize_dedup_text(title)
    normalized_law_id = normalize_dedup_text(law_id)
    if normalized_title and date:
        return ("title-date", normalized_title, date)
    if normalized_title and normalized_law_id:
        return ("title-id", normalized_title, normalized_law_id)
    if normalized_title:
        return ("title", normalized_title)
    return ()


def is_vbhn_document(law_id: str, doc_title: str, payload: dict[str, Any]) -> bool:
    values = [law_id, doc_title]
    for key in ("source_document_number", "document_number", "law_id", "source_document_title", "title"):
        values.extend(as_text_list(payload.get(key)))
    return any(VBHN_ID_PATTERN.search(value) or VBHN_PREFIX_PATTERN.search(value) for value in values if value)


def normalize_vbhn_subject_title(doc_title: str) -> str:
    title = normalize_spaces(doc_title)
    title = VBHN_PREFIX_PATTERN.sub("", title).strip(" ,.;:-")
    title = DOC_CODE_PATTERN.sub("", title)
    title = DOC_DATE_PATTERN.sub("", title)
    title = DOC_YEAR_PATTERN.sub("", title)
    title = re.sub(r"\bhợp nhất\b", "", title, flags=re.IGNORECASE, count=1)
    title = re.sub(r"\bdo\s+.+?\s+ban hành\b", "", title, flags=re.IGNORECASE)
    title = re.sub(r"\bcủa\s+(?:văn phòng quốc hội|quốc hội|chính phủ)\b", "", title, flags=re.IGNORECASE)
    return normalize_dedup_text(title)


def document_issue_date(payload: dict[str, Any], doc_title: str) -> str:
    for key in (
        "issuance_date",
        "ngay_ban_hanh",
        "signing_date",
        "signed_date",
        "issue_date",
        "issued_date",
        "promulgation_date",
        "source_document_issuance_date",
    ):
        for value in as_text_list(payload.get(key)):
            normalized = normalize_dedup_text(value)
            if normalized:
                return normalized
    match = DOC_DATE_PATTERN.search(doc_title)
    if match:
        return normalize_dedup_text(match.group(0))
    match = DOC_YEAR_PATTERN.search(doc_title)
    if match:
        return normalize_dedup_text(match.group(0))
    return ""


def normalize_dedup_text(text: str) -> str:
    return normalize_spaces(str(text or "")).casefold().strip(" ,.;:-|\"'")


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
    eligible_count = len(results)
    results = deduplicate_results_by_document(results, doc_title_format=config.doc_title_format)
    trace_search_results(
        config,
        "rerank_input_deduped",
        results,
        query_id=query_id,
        query_text=query_text,
        eligible_count=eligible_count,
        deduplicated_out=eligible_count - len(results),
    )
    if not config.rerank:
        limit = len(results) if keep_candidate_pool_for_config(config) else config.top_k
        output = rank_output_results(results, limit)
        trace_search_results(
            config,
            "rerank_skipped_output",
            output,
            query_id=query_id,
            top_k=config.top_k,
            candidate_pool=keep_candidate_pool_for_config(config),
        )
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
    output_limit = len(reranked) if keep_candidate_pool_for_config(config) else config.top_k
    output = reranked[:output_limit]
    trace_search_results(
        config,
        "rerank_output",
        output,
        query_id=query_id,
        query_text=query_text,
        scored_count=len(reranked),
        threshold=config.rerank_threshold,
        candidate_pool=keep_candidate_pool_for_config(config),
    )
    return output


def maybe_select_llm_submission_candidates(
    config: QdrantSearchConfig,
    query_text: str,
    results: list[dict[str, Any]],
    *,
    llm_selector: Any | None = None,
    query_id: Any | None = None,
) -> list[dict[str, Any]]:
    if not config.llm_select_candidates:
        return results
    if not results:
        trace_log(config, "llm_selector_empty", query_id=query_id, query_text=query_text)
        return []

    candidate_limit = config.llm_selector_max_candidates if config.llm_selector_max_candidates > 0 else len(results)
    candidates = results[:candidate_limit]
    if llm_selector is None:
        llm_selector = load_llm_candidate_selector(config.llm_selector_model_name, config.model_cache_dir)
    tokenizer = llm_selector.get("tokenizer") if isinstance(llm_selector, dict) else None
    candidate_token_budget = llm_selector_candidate_token_budget(config)

    prompt = build_llm_candidate_selector_prompt(
        query_text,
        candidates,
        doc_title_format=config.doc_title_format,
        snippet_chars=config.llm_selector_snippet_chars,
        tokenizer=tokenizer,
        candidate_token_budget=candidate_token_budget,
        system_prompt=LLM_SELECTOR_SYSTEM_PROMPT,
    )
    prompt_token_count = count_llm_selector_prompt_tokens(tokenizer, LLM_SELECTOR_SYSTEM_PROMPT, prompt)
    candidate_token_count = count_llm_selector_context_tokens(tokenizer, extract_llm_selector_context(prompt))
    raw_response = generate_llm_candidate_selector_response(
        llm_selector,
        prompt,
        max_new_tokens=config.llm_selector_max_new_tokens,
        temperature=config.llm_selector_temperature,
        system_prompt=LLM_SELECTOR_SYSTEM_PROMPT,
    )
    selected_indexes = parse_llm_candidate_selection(raw_response, len(candidates))
    if selected_indexes is None or not selected_indexes:
        fallback = rank_output_results(results, config.top_k)
        trace_search_results(
            config,
            "llm_selector_empty_or_parse_failed_fallback",
            fallback,
            query_id=query_id,
            query_text=query_text,
            raw_response=truncate_text(raw_response, 2000),
            selected_indexes=selected_indexes,
            fallback_top_k=config.top_k,
            prompt_tokens=prompt_token_count,
            candidate_tokens=candidate_token_count,
            candidate_token_budget=candidate_token_budget,
        )
        return fallback

    selected = []
    for selected_rank, candidate_index in enumerate(selected_indexes, start=1):
        item = dict(candidates[candidate_index - 1])
        item["pre_selector_rank"] = item.get("rank")
        item["llm_candidate_index"] = candidate_index
        item["llm_selected_rank"] = selected_rank
        selected.append(item)
    output = rank_output_results(selected, len(selected))
    trace_search_results(
        config,
        "llm_selector_output",
        output,
        query_id=query_id,
        query_text=query_text,
        candidate_count=len(candidates),
        selected_indexes=selected_indexes,
        raw_response=truncate_text(raw_response, 2000),
        prompt_tokens=prompt_token_count,
        candidate_tokens=candidate_token_count,
        candidate_token_budget=candidate_token_budget,
    )
    return output


def build_llm_candidate_selector_prompt(
    query_text: str,
    candidates: list[dict[str, Any]],
    *,
    doc_title_format: str = "type1",
    snippet_chars: int = 0,
    tokenizer: Any | None = None,
    candidate_token_budget: int | None = None,
    system_prompt: str = LLM_SELECTOR_SYSTEM_PROMPT,
) -> str:
    context = ""
    for index, item in enumerate(candidates, start=1):
        block = format_llm_candidate_block(
            index,
            item,
            doc_title_format=doc_title_format,
            snippet_chars=snippet_chars,
        )
        next_context = append_prompt_block(context, block)
        if context_fits_llm_candidate_budget(next_context, tokenizer, candidate_token_budget):
            context = next_context
            continue
        fitted_block = fit_llm_candidate_block(
            context,
            index,
            item,
            doc_title_format=doc_title_format,
            snippet_chars=snippet_chars,
            tokenizer=tokenizer,
            candidate_token_budget=candidate_token_budget,
        )
        if not fitted_block:
            break
        context = append_prompt_block(context, fitted_block)
    return "\n".join([*llm_selector_prompt_header(query_text), context.strip()]).strip()


def llm_selector_prompt_header(query_text: str) -> list[str]:
    return [
        "Hãy xác định các ngữ cảnh/candidate có chứa thông tin pháp luật cần thiết để trả lời câu hỏi.",
        "Yêu cầu:",
        "- Chỉ sử dụng thông tin trong candidate được cung cấp.",
        "- Chọn tất cả candidate positive, kể cả khi câu hỏi cần tích hợp nhiều điều/văn bản.",
        "- Bỏ candidate negative, nhiễu, hoặc chỉ giống từ khóa nhưng không giúp trả lời câu hỏi.",
        "- Trả về DUY NHẤT JSON hợp lệ: {\"selected\": [1, 3], \"answerable\": true}.",
        "- Nếu không candidate nào có thông tin cần thiết, trả về: {\"selected\": [], \"answerable\": false}.",
        "",
        "### Câu hỏi :",
        query_text,
        "",
        "### Ngữ cảnh :",
    ]


def format_llm_candidate_block(
    index: int,
    item: dict[str, Any],
    *,
    doc_title_format: str,
    snippet_chars: int,
) -> str:
    payload = item.get("payload") or {}
    docs = payload_to_competition_docs(payload, doc_title_format=doc_title_format)
    articles = payload_to_competition_articles(payload, doc_title_format=doc_title_format)
    retrieval_text = result_text_for_reranking(item)
    if snippet_chars > 0:
        retrieval_text = truncate_text(retrieval_text, snippet_chars)
    return llm_candidate_block_text(
        index,
        doc=docs[0] if docs else "",
        article=articles[0] if articles else "",
        score=item.get("score"),
        retrieval_text=retrieval_text,
    )


def llm_candidate_block_text(
    index: int,
    *,
    doc: str,
    article: str,
    score: Any,
    retrieval_text: str,
) -> str:
    return "\n".join(
        [
            f"- Candidate {index}:",
            f"  Văn bản: {doc}",
            f"  Điều: {article}",
            f"  Điểm rerank/truy hồi: {score}",
            "  Nội dung:",
            indent_candidate_text(retrieval_text),
        ]
    ).strip()


def fit_llm_candidate_block(
    current_prompt: str,
    index: int,
    item: dict[str, Any],
    *,
    doc_title_format: str,
    snippet_chars: int,
    tokenizer: Any | None,
    candidate_token_budget: int | None,
) -> str:
    payload = item.get("payload") or {}
    docs = payload_to_competition_docs(payload, doc_title_format=doc_title_format)
    articles = payload_to_competition_articles(payload, doc_title_format=doc_title_format)
    text = result_text_for_reranking(item)
    if snippet_chars > 0:
        text = truncate_text(text, snippet_chars)
    doc = docs[0] if docs else ""
    article = articles[0] if articles else ""
    score = item.get("score")
    low = 0
    high = len(text)
    best = ""
    while low <= high:
        midpoint = (low + high) // 2
        truncated = truncate_text(text, midpoint) if midpoint < len(text) else text
        block = llm_candidate_block_text(index, doc=doc, article=article, score=score, retrieval_text=truncated)
        if context_fits_llm_candidate_budget(append_prompt_block(current_prompt, block), tokenizer, candidate_token_budget):
            best = block
            low = midpoint + 1
        else:
            high = midpoint - 1
    return best


def append_prompt_block(prompt: str, block: str) -> str:
    return f"{prompt.rstrip()}\n\n{block.strip()}" if block.strip() else prompt


def context_fits_llm_candidate_budget(
    context: str,
    tokenizer: Any | None,
    candidate_token_budget: int | None,
) -> bool:
    if tokenizer is None or candidate_token_budget is None or candidate_token_budget <= 0:
        return True
    token_count = count_llm_selector_context_tokens(tokenizer, context)
    return token_count is not None and token_count <= candidate_token_budget


def count_llm_selector_prompt_tokens(tokenizer: Any | None, system_prompt: str, prompt: str) -> int | None:
    if tokenizer is None:
        return None
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt},
    ]
    if hasattr(tokenizer, "apply_chat_template"):
        try:
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except TypeError:
            text = "\n".join(message["content"] for message in messages)
    else:
        text = "\n".join(message["content"] for message in messages)
    try:
        encoded = tokenizer(text, add_special_tokens=False)
    except TypeError:
        encoded = tokenizer(text)
    input_ids = encoded.get("input_ids") if isinstance(encoded, dict) else getattr(encoded, "input_ids", encoded)
    return len(input_ids)


def count_llm_selector_context_tokens(tokenizer: Any | None, context: str) -> int | None:
    if tokenizer is None:
        return None
    try:
        encoded = tokenizer(context, add_special_tokens=False)
    except TypeError:
        encoded = tokenizer(context)
    input_ids = encoded.get("input_ids") if isinstance(encoded, dict) else getattr(encoded, "input_ids", encoded)
    return len(input_ids)


def extract_llm_selector_context(prompt: str) -> str:
    marker = "### Ngữ cảnh :"
    if marker not in prompt:
        return prompt
    return prompt.split(marker, maxsplit=1)[1].strip()


def llm_selector_candidate_token_budget(config: QdrantSearchConfig) -> int:
    candidate_window = max(0, int(config.llm_selector_context_window or 0))
    if candidate_window <= 0:
        return 0
    return candidate_window


def indent_candidate_text(text: str) -> str:
    normalized = str(text or "").strip()
    if not normalized:
        return "  "
    return "\n".join(f"  {line}" for line in normalized.splitlines())


def parse_llm_candidate_selection(response_text: str, candidate_count: int) -> list[int] | None:
    data = parse_first_json_value(response_text)
    if data is None:
        return None
    if isinstance(data, dict):
        values = first_present(data, ("selected", "indices", "indexes", "candidates", "relevant_candidates"))
        if values is None and data.get("answerable") is False:
            values = []
    elif isinstance(data, list):
        values = data
    else:
        return None
    if not isinstance(values, list):
        return None
    return normalize_selected_candidate_indexes(values, candidate_count)


def parse_first_json_value(text: str) -> Any | None:
    for candidate in json_candidate_strings(text):
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue
    return None


def json_candidate_strings(text: str) -> list[str]:
    candidates = [str(text or "").strip()]
    fenced = re.search(r"```(?:json)?\s*(.*?)```", candidates[0], flags=re.IGNORECASE | re.DOTALL)
    if fenced:
        candidates.append(fenced.group(1).strip())
    snippet = first_balanced_json_snippet(candidates[0])
    if snippet:
        candidates.append(snippet)
    return [candidate for candidate in candidates if candidate]


def first_balanced_json_snippet(text: str) -> str:
    start_positions = [position for position in (text.find("{"), text.find("[")) if position >= 0]
    if not start_positions:
        return ""
    start = min(start_positions)
    stack: list[str] = []
    in_string = False
    escaped = False
    pairs = {"{": "}", "[": "]"}
    for position in range(start, len(text)):
        char = text[position]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
            continue
        if char in pairs:
            stack.append(pairs[char])
            continue
        if stack and char == stack[-1]:
            stack.pop()
            if not stack:
                return text[start : position + 1]
    return ""


def normalize_selected_candidate_indexes(values: list[Any], candidate_count: int) -> list[int]:
    selected: list[int] = []
    seen: set[int] = set()
    for value in values:
        if isinstance(value, dict):
            value = first_present(value, ("index", "id", "candidate", "candidate_index"))
        try:
            index = int(str(value).strip())
        except (TypeError, ValueError):
            continue
        if 1 <= index <= candidate_count and index not in seen:
            seen.add(index)
            selected.append(index)
    return selected


def load_llm_candidate_selector(model_name: str, cache_dir: Path | None) -> dict[str, Any]:
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("transformers and torch are required for --llm-select-candidates.") from exc

    kwargs: dict[str, Any] = {"cache_dir": str(cache_dir)} if cache_dir else {}
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, **kwargs)
    model_kwargs: dict[str, Any] = {"trust_remote_code": True, "use_cache": True, **kwargs}
    if torch.cuda.is_available():
        model_kwargs.update({"torch_dtype": torch.bfloat16, "device_map": "auto"})
    try:
        model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    except (ImportError, ValueError):
        if "device_map" not in model_kwargs:
            raise
        model_kwargs.pop("device_map", None)
        model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
        device = preferred_torch_device()
        if device and hasattr(model, "to"):
            model.to(device)
    if hasattr(model, "eval"):
        model.eval()
    if getattr(tokenizer, "pad_token_id", None) is None and getattr(tokenizer, "eos_token_id", None) is not None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    return {"model": model, "tokenizer": tokenizer, "model_name": model_name}


def generate_llm_candidate_selector_response(
    llm_selector: Any,
    prompt: str,
    *,
    max_new_tokens: int,
    temperature: float,
    system_prompt: str = LLM_SELECTOR_SYSTEM_PROMPT,
) -> str:
    if callable(llm_selector):
        return str(llm_selector(prompt))
    tokenizer = llm_selector["tokenizer"]
    model = llm_selector["model"]
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt},
    ]
    if hasattr(tokenizer, "apply_chat_template"):
        model_inputs = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
    else:
        text = "\n".join(message["content"] for message in messages)
        model_inputs = tokenizer(text, return_tensors="pt")
    device = model_device(model) or getattr(model, "device", None)
    if device is not None and hasattr(model_inputs, "to"):
        model_inputs = model_inputs.to(device)
    generate_kwargs: dict[str, Any] = {
        "max_new_tokens": max_new_tokens,
        "do_sample": temperature > 0,
        "pad_token_id": getattr(tokenizer, "pad_token_id", None),
    }
    if temperature > 0:
        generate_kwargs["temperature"] = temperature
    try:
        import torch
    except ImportError:
        torch = None
    no_grad = torch.no_grad() if torch is not None else contextlib.nullcontext()
    with no_grad:
        output_ids = model.generate(**model_inputs, **generate_kwargs)
    input_ids = model_inputs["input_ids"] if isinstance(model_inputs, dict) else model_inputs.input_ids
    generated_ids = [
        output[len(input_id) :]
        for input_id, output in zip(input_ids, output_ids, strict=True)
    ]
    return tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()


def truncate_text(text: str, max_chars: int) -> str:
    text = str(text or "").strip()
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + "..."


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

    for item in deduplicate_results_by_document(result["results"], doc_title_format=doc_title_format):
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


def has_server_side_rrf(models: Any, rrf_weights: tuple[float, float] = DEFAULT_RRF_WEIGHTS) -> bool:
    if not hasattr(models, "Prefetch"):
        return False
    if hasattr(models, "RrfQuery") and hasattr(models, "Rrf"):
        return True
    return rrf_weights == DEFAULT_RRF_WEIGHTS and hasattr(models, "FusionQuery") and hasattr(models, "Fusion")


def rrf_query(models: Any, rrf_weights: tuple[float, float] = DEFAULT_RRF_WEIGHTS) -> Any:
    if hasattr(models, "RrfQuery") and hasattr(models, "Rrf"):
        try:
            return models.RrfQuery(rrf=models.Rrf(weights=list(rrf_weights)))
        except TypeError:
            if rrf_weights == DEFAULT_RRF_WEIGHTS:
                return models.RrfQuery(rrf=models.Rrf())
            raise ValueError("Custom rrf_weights require qdrant-client support for models.Rrf(weights=...).")
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


def _merge_hits(combined: dict[str, dict[str, Any]], hits: list[Any], source: str, weight: float = 1.0) -> None:
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
        item["score"] += weight / (RRF_K + rank)
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
