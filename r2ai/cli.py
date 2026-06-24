"""Command line interface for R2AI baseline utilities."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from r2ai.data_ingest.phapdien import BuildPaths, build_phapdien_data
from r2ai.data_ingest.phapdien.download import ensure_phapdien_data
from r2ai.indexing.config import (
    DEFAULT_COLLECTION,
    DEFAULT_ANSWER_ARTICLE_LIMIT,
    DEFAULT_DENSE_MODEL,
    DEFAULT_DOC_TITLE_FORMAT,
    DEFAULT_HNSW_EF_CONSTRUCT,
    DEFAULT_HNSW_M,
    DEFAULT_PREFETCH_LIMIT,
    DEFAULT_RERANKER_MAX_LENGTH,
    DEFAULT_RERANKER_MODEL,
    DEFAULT_SEARCH_QDRANT_TIMEOUT,
    DEFAULT_SEARCH_MODE,
    DEFAULT_SPARSE_MODEL,
    QdrantIngestConfig,
    QdrantSearchConfig,
)
from r2ai.indexing.qdrant_ingest import ingest_phapdien_to_qdrant
from r2ai.retrieval.qdrant_search import (
    load_query_vectors,
    load_questions,
    search_qdrant,
    search_qdrant_batch,
    write_submission,
    write_submission_zip,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="R2AI phapdien-only baseline tools")
    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser("build-phapdien-data", help="Build Phase 1 phapdien JSONL artifacts")
    build.add_argument(
        "--source-dir",
        type=Path,
        default=Path("data/phapdien-moj-gov-vn"),
        help="Directory containing phapdien parquet and ontology files",
    )
    build.add_argument(
        "--output-dir",
        type=Path,
        default=Path("build"),
        help="Directory to write Phase 1 artifacts",
    )
    build.add_argument("--max-chunk-tokens", type=int, default=2048)
    build.add_argument("--chunk-overlap-tokens", type=int, default=256)

    ensure_data = subparsers.add_parser("ensure-phapdien-data", help="Download phapdien files if missing")
    ensure_data.add_argument(
        "--source-dir",
        type=Path,
        default=Path("data/phapdien-moj-gov-vn"),
        help="Directory that should contain phapdien parquet and ontology files",
    )
    ensure_data.add_argument(
        "--repo-id",
        default="tmquan/phapdien-moj-gov-vn",
        help="Hugging Face dataset repo id used by the fallback downloader",
    )

    ingest = subparsers.add_parser("ingest-qdrant", help="Build phapdien data and ingest into Qdrant Cloud")
    ingest.add_argument("--source-dir", type=Path, default=Path("data/phapdien-moj-gov-vn"))
    ingest.add_argument("--build-dir", type=Path, default=Path("build"))
    ingest.add_argument(
        "--collection",
        default=None,
        help=f"Qdrant collection name; defaults to QDRANT_COLLECTION or {DEFAULT_COLLECTION}",
    )
    ingest.add_argument("--dense-model", default=DEFAULT_DENSE_MODEL)
    ingest.add_argument("--sparse-model", default=DEFAULT_SPARSE_MODEL)
    ingest.add_argument("--batch-size", type=int, default=16)
    ingest.add_argument("--model-cache-dir", type=Path, default=None)
    ingest.add_argument("--recreate-collection", action="store_true")
    ingest.add_argument("--skip-build", action="store_true")
    ingest.add_argument("--limit", type=int, default=None, help="Optional point limit for smoke tests")
    ingest.add_argument("--max-chunk-tokens", type=int, default=2048)
    ingest.add_argument("--chunk-overlap-tokens", type=int, default=256)
    ingest.add_argument("--hnsw-m", type=int, default=DEFAULT_HNSW_M, help="Optional Qdrant HNSW m value")
    ingest.add_argument(
        "--hnsw-ef-construct",
        "--hnsw-ef",
        dest="hnsw_ef_construct",
        type=int,
        default=DEFAULT_HNSW_EF_CONSTRUCT,
        help="Optional Qdrant HNSW ef_construct value; --hnsw-ef is a shorthand alias",
    )

    search = subparsers.add_parser("search-qdrant", help="Search the Qdrant Cloud collection with hybrid retrieval")
    search.add_argument("query", help="Vietnamese legal question or keyword query")
    search.add_argument(
        "--collection",
        default=None,
        help=f"Qdrant collection name; defaults to QDRANT_COLLECTION or {DEFAULT_COLLECTION}",
    )
    search.add_argument("--mode", choices=["bm25", "dense", "hybrid"], default=DEFAULT_SEARCH_MODE)
    search.add_argument("--dense-model", default=DEFAULT_DENSE_MODEL)
    search.add_argument("--sparse-model", default=DEFAULT_SPARSE_MODEL)
    search.add_argument("--top-k", type=int, default=5)
    search.add_argument("--prefetch-limit", type=int, default=DEFAULT_PREFETCH_LIMIT)
    search.add_argument("--qdrant-timeout", type=float, default=DEFAULT_SEARCH_QDRANT_TIMEOUT)
    search.add_argument("--doc-title-format", choices=["type1", "type2"], default=DEFAULT_DOC_TITLE_FORMAT)
    search.add_argument("--answer-article-limit", type=int, default=DEFAULT_ANSWER_ARTICLE_LIMIT)
    search.add_argument("--model-cache-dir", type=Path, default=None)
    search.add_argument("--query-embedding", type=Path, default=None, help="Optional .npy vector for this query")
    search.add_argument("--query-instruction", default="", help="Optional prefix for on-the-fly dense embedding")
    search.add_argument("--rerank", action="store_true", help="Rerank retrieved candidates with a cross-encoder")
    search.add_argument("--reranker-model", default=DEFAULT_RERANKER_MODEL)
    search.add_argument("--reranker-max-length", type=int, default=DEFAULT_RERANKER_MAX_LENGTH)
    search.add_argument(
        "--rerank-threshold",
        "--rerank-thresold",
        dest="rerank_threshold",
        type=float,
        default=None,
        help="Only keep reranked results with rerank_score >= this threshold",
    )
    search.add_argument("--topic-title", default=None)
    search.add_argument("--subject-title", default=None)
    search.add_argument("--source-law-id", default=None)
    search.add_argument("--source-article-no", default=None)
    search.add_argument("--citation-confidence", default=None)
    search.add_argument("--topic-number", type=int, default=None)

    submit = subparsers.add_parser("submit-qdrant", help="Create competition results.json from Qdrant retrieval")
    submit.add_argument("--questions", type=Path, required=True, help="Question file: .json, .jsonl, or .csv")
    submit.add_argument("--output", type=Path, default=Path("results.json"))
    submit.add_argument("--output-format", choices=["json", "jsonl"], default="json")
    submit.add_argument("--zip-output", type=Path, default=None, help="Optional flat zip containing results.json")
    submit.add_argument(
        "--checkpoint-output",
        type=Path,
        default=None,
        help="JSONL checkpoint for completed question predictions; defaults to <output>.checkpoint.jsonl",
    )
    submit.add_argument("--no-resume", action="store_true", help="Ignore any existing submission checkpoint")
    submit.add_argument(
        "--collection",
        default=None,
        help=f"Qdrant collection name; defaults to QDRANT_COLLECTION or {DEFAULT_COLLECTION}",
    )
    submit.add_argument("--mode", choices=["bm25", "dense", "hybrid"], default=DEFAULT_SEARCH_MODE)
    submit.add_argument("--dense-model", default=DEFAULT_DENSE_MODEL)
    submit.add_argument("--sparse-model", default=DEFAULT_SPARSE_MODEL)
    submit.add_argument("--top-k", type=int, default=5)
    submit.add_argument("--prefetch-limit", type=int, default=DEFAULT_PREFETCH_LIMIT)
    submit.add_argument("--qdrant-timeout", type=float, default=DEFAULT_SEARCH_QDRANT_TIMEOUT)
    submit.add_argument("--doc-title-format", choices=["type1", "type2"], default=DEFAULT_DOC_TITLE_FORMAT)
    submit.add_argument("--answer-article-limit", type=int, default=DEFAULT_ANSWER_ARTICLE_LIMIT)
    submit.add_argument("--model-cache-dir", type=Path, default=None)
    submit.add_argument("--query-embeddings", type=Path, default=None, help="Optional .npy matrix aligned to questions")
    submit.add_argument("--query-instruction", default="", help="Optional prefix for on-the-fly dense embedding")
    submit.add_argument("--rerank", action="store_true", help="Rerank retrieved candidates with a cross-encoder")
    submit.add_argument("--reranker-model", default=DEFAULT_RERANKER_MODEL)
    submit.add_argument("--reranker-max-length", type=int, default=DEFAULT_RERANKER_MAX_LENGTH)
    submit.add_argument(
        "--rerank-threshold",
        "--rerank-thresold",
        dest="rerank_threshold",
        type=float,
        default=None,
        help="Only keep reranked results with rerank_score >= this threshold",
    )
    submit.add_argument("--limit", type=int, default=None, help="Optional number of questions for smoke tests")
    submit.add_argument("--progress-every", type=int, default=25)
    submit.add_argument("--qdrant-batch-size", type=int, default=64, help="Batch size for dense .npy Qdrant queries")
    submit.add_argument("--topic-title", default=None)
    submit.add_argument("--subject-title", default=None)
    submit.add_argument("--source-law-id", default=None)
    submit.add_argument("--source-article-no", default=None)
    submit.add_argument("--citation-confidence", default=None)
    submit.add_argument("--topic-number", type=int, default=None)
    return parser


def load_env_file(path: Path = Path(".env")) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", maxsplit=1)
        key = key.strip()
        value = value.strip().strip("\"'")
        if key and key not in os.environ:
            os.environ[key] = value


def main() -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    load_env_file()
    parser = build_parser()
    args = parser.parse_args()
    if getattr(args, "rerank_threshold", None) is not None and not getattr(args, "rerank", False):
        parser.error("--rerank-threshold requires --rerank")

    if args.command == "build-phapdien-data":
        report = build_phapdien_data(
            BuildPaths(
                source_dir=args.source_dir,
                output_dir=args.output_dir,
                max_chunk_tokens=args.max_chunk_tokens,
                chunk_overlap_tokens=args.chunk_overlap_tokens,
            )
        )
        counts = report["counts"]
        print(
            "Built phapdien data: "
            f"{counts['articles']} articles, {counts['retrieval_units']} retrieval units "
            f"-> {args.output_dir}"
        )
        return

    if args.command == "ensure-phapdien-data":
        path = ensure_phapdien_data(source_dir=args.source_dir, repo_id=args.repo_id)
        print(f"Phapdien data ready at {path}")
        return

    if args.command == "ingest-qdrant":
        result = ingest_phapdien_to_qdrant(
            QdrantIngestConfig.from_env(
                source_dir=args.source_dir,
                build_dir=args.build_dir,
                collection_name=args.collection,
                dense_model_name=args.dense_model,
                sparse_model_name=args.sparse_model,
                batch_size=args.batch_size,
                model_cache_dir=args.model_cache_dir,
                recreate_collection=args.recreate_collection,
                skip_build=args.skip_build,
                limit=args.limit,
                max_chunk_tokens=args.max_chunk_tokens,
                chunk_overlap_tokens=args.chunk_overlap_tokens,
                hnsw_m=args.hnsw_m,
                hnsw_ef_construct=args.hnsw_ef_construct,
            )
        )
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
        return

    if args.command == "search-qdrant":
        query_vector = load_query_vectors(args.query_embedding)[0] if args.query_embedding else None
        result = search_qdrant(
            QdrantSearchConfig.from_env(
                query_text=args.query,
                collection_name=args.collection,
                search_mode=args.mode,
                dense_model_name=args.dense_model,
                sparse_model_name=args.sparse_model,
                top_k=args.top_k,
                prefetch_limit=args.prefetch_limit,
                qdrant_timeout=args.qdrant_timeout,
                doc_title_format=args.doc_title_format,
                answer_article_limit=args.answer_article_limit,
                model_cache_dir=args.model_cache_dir,
                query_vector=query_vector,
                query_instruction=args.query_instruction,
                rerank=args.rerank,
                reranker_model_name=args.reranker_model,
                reranker_max_length=args.reranker_max_length,
                rerank_threshold=args.rerank_threshold,
                topic_title=args.topic_title,
                subject_title=args.subject_title,
                source_law_id=args.source_law_id,
                source_article_no=args.source_article_no,
                citation_confidence=args.citation_confidence,
                topic_number=args.topic_number,
            )
        )
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
        return

    if args.command == "submit-qdrant":
        questions = load_questions(args.questions)
        query_vectors = load_query_vectors(args.query_embeddings) if args.query_embeddings else None
        if args.limit is not None:
            questions = questions[: args.limit]
            if query_vectors is not None:
                query_vectors = query_vectors[: args.limit]
        checkpoint_output = None
        if not args.no_resume:
            checkpoint_output = args.checkpoint_output or args.output.with_name(f"{args.output.name}.checkpoint.jsonl")
        rows = search_qdrant_batch(
            QdrantSearchConfig.from_env(
                query_text="placeholder",
                collection_name=args.collection,
                search_mode=args.mode,
                dense_model_name=args.dense_model,
                sparse_model_name=args.sparse_model,
                top_k=args.top_k,
                prefetch_limit=args.prefetch_limit,
                qdrant_timeout=args.qdrant_timeout,
                doc_title_format=args.doc_title_format,
                answer_article_limit=args.answer_article_limit,
                model_cache_dir=args.model_cache_dir,
                query_instruction=args.query_instruction,
                rerank=args.rerank,
                reranker_model_name=args.reranker_model,
                reranker_max_length=args.reranker_max_length,
                rerank_threshold=args.rerank_threshold,
                topic_title=args.topic_title,
                subject_title=args.subject_title,
                source_law_id=args.source_law_id,
                source_article_no=args.source_article_no,
                citation_confidence=args.citation_confidence,
                topic_number=args.topic_number,
            ),
            questions=questions,
            query_vectors=query_vectors,
            progress_every=args.progress_every,
            query_batch_size=args.qdrant_batch_size,
            checkpoint_path=checkpoint_output,
        )
        write_submission(args.output, rows, output_format=args.output_format)
        if args.zip_output:
            if args.output_format != "json":
                raise RuntimeError("--zip-output requires --output-format json for Challenge submission.")
            write_submission_zip(args.zip_output, args.output)
        print(f"Wrote {len(rows)} predictions to {args.output}")
        if args.zip_output:
            print(f"Wrote flat submission zip to {args.zip_output}")
        return

    parser.error(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
