"""Build business-focused VLD Qdrant artifacts from parquet data.

This is the production-oriented companion to the preview scripts. It keeps the
tree/chunk semantics from ``build_vld_tree_preview.py`` and
``build_vld_chunk_preview.py`` while streaming content parquet files so the
full business subset can be built on Kaggle.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterator

try:
    import pyarrow.parquet as pq
except ImportError as exc:  # pragma: no cover - environment guard.
    raise RuntimeError("pyarrow is required. Run: uv sync --extra data") from exc

from build_vld_business_collection import classify_metadata
from build_vld_chunk_preview import build_chunks_for_tree, make_qdrant_preview
from build_vld_tree_preview import build_tree


def iter_parquet_rows(path: Path, columns: list[str] | None = None) -> Iterator[dict[str, Any]]:
    pf = pq.ParquetFile(path)
    for batch in pf.iter_batches(columns=columns):
        table = batch.to_pydict()
        keys = list(table)
        for values in zip(*(table[key] for key in keys), strict=True):
            yield dict(zip(keys, values, strict=True))


def load_business_metadata(
    vld_root: Path,
    *,
    min_year: int,
    ids_file: Path | None,
    limit: int,
) -> tuple[dict[int, dict[str, Any]], dict[str, Any]]:
    allowed_ids = load_ids(ids_file) if ids_file else None
    metadata_path = vld_root / "metadata" / "data-00000-of-00001.parquet"
    rows: dict[int, dict[str, Any]] = {}
    total = 0
    kept = 0
    tier_counts: Counter[str] = Counter()
    type_counts: Counter[str] = Counter()
    sector_counts: Counter[str] = Counter()

    for row in iter_parquet_rows(metadata_path):
        total += 1
        doc_id = int(row["id"])
        if allowed_ids is not None and doc_id not in allowed_ids:
            continue
        keep, tier = classify_metadata(row, min_year=min_year)
        if not keep:
            continue
        row["_business_scope_tier"] = tier
        rows[doc_id] = row
        kept += 1
        tier_counts[tier] += 1
        type_counts[str(row.get("legal_type") or "<missing>").strip()] += 1
        for sector in split_sectors(row.get("legal_sectors")):
            sector_counts[sector] += 1
        if limit > 0 and kept >= limit:
            break

    return rows, {
        "total_metadata_rows_scanned": total,
        "kept_documents": len(rows),
        "tier_counts": dict(tier_counts),
        "top_legal_types": dict(type_counts.most_common(25)),
        "top_sectors": dict(sector_counts.most_common(25)),
    }


def split_sectors(value: Any) -> list[str]:
    text = str(value or "").strip()
    if not text:
        return []
    return [part.strip() for part in text.replace("|", ",").split(",") if part.strip()]


def load_ids(path: Path) -> set[int]:
    ids: set[int] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        for part in line.replace(",", " ").split():
            ids.add(int(part))
    return ids


def build_artifacts(args: argparse.Namespace) -> dict[str, Any]:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metadata_by_id, metadata_report = load_business_metadata(
        args.vld_root,
        min_year=args.min_year,
        ids_file=args.ids_file,
        limit=args.limit,
    )
    wanted_ids = set(metadata_by_id)

    metadata_out_path = args.output_dir / "metadata.jsonl"
    ids_out_path = args.output_dir / "document_ids.txt"
    articles_path = args.output_dir / "articles.jsonl"
    chunks_path = args.output_dir / "retrieval_units.jsonl"
    preview_path = args.output_dir / "qdrant_payload_preview.jsonl"

    ids_out_path.write_text("\n".join(str(doc_id) for doc_id in sorted(wanted_ids)) + "\n", encoding="utf-8")
    with metadata_out_path.open("w", encoding="utf-8", newline="\n") as handle:
        for doc_id in sorted(wanted_ids):
            handle.write(json.dumps(clean_metadata(metadata_by_id[doc_id]), ensure_ascii=False) + "\n")

    documents_built = 0
    chunk_count = 0
    chunk_type_counts: Counter[str] = Counter()
    table_chunk_count = 0
    errors: list[dict[str, Any]] = []

    with (
        articles_path.open("w", encoding="utf-8", newline="\n") as articles_out,
        chunks_path.open("w", encoding="utf-8", newline="\n") as chunks_out,
        preview_path.open("w", encoding="utf-8", newline="\n") as preview_out,
    ):
        for parquet_path in sorted((args.vld_root / "content").glob("*.parquet")):
            for content_row in iter_parquet_rows(parquet_path, columns=["id", "content"]):
                doc_id = int(content_row["id"])
                metadata = metadata_by_id.get(doc_id)
                if metadata is None:
                    continue
                try:
                    document = build_tree(metadata, str(content_row.get("content") or ""))
                    chunks = build_chunks_for_tree(
                        document,
                        max_text_tokens=args.max_text_tokens,
                        table_rows_per_chunk=args.table_rows_per_chunk,
                    )
                except Exception as exc:  # Keep long Kaggle jobs moving.
                    errors.append({"document_id": doc_id, "error": str(exc)})
                    continue

                documents_built += 1
                articles_out.write(
                    json.dumps(
                        {
                            **document["metadata"],
                            "business_scope_tier": metadata.get("_business_scope_tier"),
                            "chunk_count": len(chunks),
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                for chunk in chunks:
                    chunk["business_scope_tier"] = metadata.get("_business_scope_tier")
                    chunk_count += 1
                    chunk_type_counts[chunk["chunk_type"]] += 1
                    table_chunk_count += int(bool(chunk.get("contains_table")))
                    chunks_out.write(json.dumps(chunk, ensure_ascii=False) + "\n")
                    preview_out.write(json.dumps(make_qdrant_preview(chunk), ensure_ascii=False) + "\n")

                if args.progress_every > 0 and documents_built % args.progress_every == 0:
                    print(f"Built {documents_built} documents, {chunk_count} chunks", flush=True)

    report = {
        "dataset": "vietnamese-legal-documents",
        "vld_root": str(args.vld_root),
        "output_dir": str(args.output_dir),
        "filter": {
            "min_year": args.min_year,
            "limit": args.limit,
            "ids_file": str(args.ids_file) if args.ids_file else "",
        },
        "chunking": {
            "tokenizer": "tiktoken/cl100k_base",
            "max_text_tokens": args.max_text_tokens,
            "table_rows_per_chunk": args.table_rows_per_chunk,
        },
        "metadata": metadata_report,
        "documents_built": documents_built,
        "chunk_count": chunk_count,
        "chunk_type_counts": dict(chunk_type_counts),
        "table_chunk_count": table_chunk_count,
        "error_count": len(errors),
        "errors_preview": errors[:20],
        "outputs": {
            "metadata": str(metadata_out_path),
            "document_ids": str(ids_out_path),
            "articles": str(articles_path),
            "retrieval_units": str(chunks_path),
            "qdrant_payload_preview": str(preview_path),
            "report": str(args.output_dir / "build_report.json"),
        },
    }
    (args.output_dir / "build_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return report


def clean_metadata(row: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in row.items() if not key.startswith("_")}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vld-root", type=Path, default=Path("data/vietnamese-legal-documents"))
    parser.add_argument("--output-dir", type=Path, default=Path("build/vld_business_scope"))
    parser.add_argument("--min-year", type=int, default=2010)
    parser.add_argument("--max-text-tokens", type=int, default=2048)
    parser.add_argument("--table-rows-per-chunk", type=int, default=8)
    parser.add_argument("--ids-file", type=Path)
    parser.add_argument("--limit", type=int, default=0, help="Optional document limit for smoke tests.")
    parser.add_argument("--progress-every", type=int, default=100)
    return parser


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except AttributeError:
        pass
    args = build_parser().parse_args()
    report = build_artifacts(args)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
