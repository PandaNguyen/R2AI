"""Command line interface for R2AI baseline utilities."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from r2ai.data_ingest.phapdien import BuildPaths, build_phapdien_data
from r2ai.data_ingest.phapdien.download import ensure_phapdien_data
from r2ai.indexing.config import DEFAULT_COLLECTION, DEFAULT_DENSE_MODEL, DEFAULT_SPARSE_MODEL, QdrantIngestConfig
from r2ai.indexing.qdrant_ingest import ingest_phapdien_to_qdrant


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
    build.add_argument("--max-chunk-tokens", type=int, default=384)
    build.add_argument("--chunk-overlap-tokens", type=int, default=48)

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
    ingest.add_argument("--collection", default=DEFAULT_COLLECTION)
    ingest.add_argument("--dense-model", default=DEFAULT_DENSE_MODEL)
    ingest.add_argument("--sparse-model", default=DEFAULT_SPARSE_MODEL)
    ingest.add_argument("--batch-size", type=int, default=16)
    ingest.add_argument("--model-cache-dir", type=Path, default=None)
    ingest.add_argument("--recreate-collection", action="store_true")
    ingest.add_argument("--skip-build", action="store_true")
    ingest.add_argument("--limit", type=int, default=None, help="Optional point limit for smoke tests")
    ingest.add_argument("--max-chunk-tokens", type=int, default=2048)
    ingest.add_argument("--chunk-overlap-tokens", type=int, default=128)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

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
            )
        )
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
        return

    parser.error(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
