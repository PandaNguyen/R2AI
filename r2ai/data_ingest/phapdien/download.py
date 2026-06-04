"""Download helpers for the phapdien dataset."""

from __future__ import annotations

from pathlib import Path


PHAPDIEN_DATASET_REPO_ID = "tmquan/phapdien-moj-gov-vn"
PHAPDIEN_GIT_URL = "https://huggingface.co/datasets/tmquan/phapdien-moj-gov-vn"
MIN_PARQUET_BYTES = 10_000


def has_phapdien_articles(source_dir: Path) -> bool:
    return source_dir.exists() and any(path.stat().st_size > MIN_PARQUET_BYTES for path in source_dir.glob("articles-*.parquet"))


def download_phapdien_snapshot(source_dir: Path, repo_id: str = PHAPDIEN_DATASET_REPO_ID) -> Path:
    """Download the phapdien dataset files needed by the local builder.

    This is the fallback path for Kaggle images where ``git clone`` works but
    Git LFS is unavailable, leaving parquet files as pointer stubs.
    """
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RuntimeError("huggingface_hub is required to download phapdien data. Install the ingest extra.") from exc

    source_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=source_dir,
        allow_patterns=[
            "articles-*.parquet",
            "ontology*.csv",
            "ontology*.parquet",
            "subjects.parquet",
            "tree_nodes.parquet",
            "analytics.json",
            "README.md",
        ],
    )
    return source_dir


def ensure_phapdien_data(source_dir: Path, repo_id: str = PHAPDIEN_DATASET_REPO_ID) -> Path:
    if has_phapdien_articles(source_dir):
        return source_dir
    return download_phapdien_snapshot(source_dir=source_dir, repo_id=repo_id)
