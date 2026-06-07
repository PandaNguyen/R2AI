"""Ensure vietnamese-legal-documents parquet files exist locally."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


DEFAULT_REPO_ID = "vohuutridung/vietnamese-legal-documents"
DEFAULT_GIT_URL = f"https://huggingface.co/datasets/{DEFAULT_REPO_ID}"
MIN_PARQUET_BYTES = 10_000


def has_vld_data(source_dir: Path) -> bool:
    metadata = source_dir / "metadata" / "data-00000-of-00001.parquet"
    content_files = list((source_dir / "content").glob("*.parquet"))
    return (
        metadata.exists()
        and metadata.stat().st_size > MIN_PARQUET_BYTES
        and any(path.stat().st_size > MIN_PARQUET_BYTES for path in content_files)
    )


def clone_if_missing(source_dir: Path, git_url: str) -> None:
    if source_dir.exists():
        return
    source_dir.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "clone", git_url, str(source_dir)], check=False)


def snapshot_download(source_dir: Path, repo_id: str) -> None:
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RuntimeError("huggingface_hub is required. Run: uv sync --extra ingest") from exc

    source_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=source_dir,
        allow_patterns=[
            "README.md",
            ".gitattributes",
            "metadata/*.parquet",
            "content/*.parquet",
            "charts/*",
        ],
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=Path("data/vietnamese-legal-documents"))
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--git-url", default=DEFAULT_GIT_URL)
    parser.add_argument(
        "--download-method",
        choices=["snapshot", "git-first"],
        default="git-first",
        help="Use snapshot on Kaggle to avoid a large .git/LFS checkout.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.download_method == "git-first":
        clone_if_missing(args.source_dir, args.git_url)
    if not has_vld_data(args.source_dir):
        snapshot_download(args.source_dir, args.repo_id)
    if not has_vld_data(args.source_dir):
        raise RuntimeError(f"Could not find valid VLD parquet files under {args.source_dir}")
    print(f"VLD data ready at {args.source_dir}")


if __name__ == "__main__":
    main()
