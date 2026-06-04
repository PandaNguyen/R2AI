"""Command line interface for R2AI baseline utilities."""

from __future__ import annotations

import argparse
from pathlib import Path

from r2ai.data_ingest.phapdien import BuildPaths, build_phapdien_data


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
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "build-phapdien-data":
        report = build_phapdien_data(BuildPaths(source_dir=args.source_dir, output_dir=args.output_dir))
        counts = report["counts"]
        print(
            "Built phapdien data: "
            f"{counts['articles']} articles, {counts['retrieval_units']} retrieval units "
            f"-> {args.output_dir}"
        )
        return

    parser.error(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
