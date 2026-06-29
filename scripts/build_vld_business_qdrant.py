"""Build business-focused VLD Qdrant artifacts from parquet data.

This is the production-oriented companion to the preview scripts. It keeps the
tree/chunk semantics from ``build_vld_tree_preview.py`` and
``build_vld_chunk_preview.py`` while streaming content parquet files so the
full business subset can be built on Kaggle.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterator

try:
    import pyarrow.parquet as pq
except ImportError as exc:  # pragma: no cover - environment guard.
    raise RuntimeError("pyarrow is required. Run: uv sync --extra data") from exc

from build_vld_business_collection import classify_metadata, effect_status, load_effect_status_by_id, merge_effect_status, normalize_metadata_row
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
    effect_metadata_path: Path | None,
) -> tuple[dict[int, dict[str, Any]], dict[str, Any]]:
    allowed_ids = load_ids(ids_file) if ids_file else None
    metadata_path = vld_root / "metadata" / "data-00000-of-00001.parquet"
    effect_status_by_id = load_effect_status_by_id(effect_metadata_path)
    rows: dict[int, dict[str, Any]] = {}
    total = 0
    kept = 0
    tier_counts: Counter[str] = Counter()
    excluded_reason_counts: Counter[str] = Counter()
    type_counts: Counter[str] = Counter()
    sector_counts: Counter[str] = Counter()
    effect_status_counts: Counter[str] = Counter()

    for row in iter_parquet_rows(metadata_path):
        total += 1
        row = normalize_metadata_row(merge_effect_status(row, effect_status_by_id))
        doc_id = int(row["id"])
        if allowed_ids is not None and doc_id not in allowed_ids:
            continue
        keep, tier = classify_metadata(row, min_year=min_year)
        if not keep:
            excluded_reason_counts[tier] += 1
            continue
        row["_business_scope_tier"] = tier
        rows[doc_id] = row
        kept += 1
        tier_counts[tier] += 1
        type_counts[str(row.get("legal_type") or "<missing>").strip()] += 1
        for sector in split_sectors(row.get("legal_sectors")):
            sector_counts[sector] += 1
        effect_status_counts[effect_status(row) or "<missing>"] += 1
        if limit > 0 and kept >= limit:
            break

    return rows, {
        "total_metadata_rows_scanned": total,
        "kept_documents": len(rows),
        "tier_counts": dict(tier_counts),
        "excluded_reason_counts": dict(excluded_reason_counts),
        "effect_metadata_path": str(effect_metadata_path) if effect_metadata_path else "",
        "effect_metadata_loaded": bool(effect_status_by_id),
        "top_legal_types": dict(type_counts.most_common(25)),
        "top_sectors": dict(sector_counts.most_common(25)),
        "top_effect_status": dict(effect_status_counts.most_common(25)),
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


def is_article_embedding_chunk(chunk: dict[str, Any]) -> bool:
    return (
        chunk.get("node_type") == "article"
        and chunk.get("chunk_type") in {"article_text_chunk", "article_split_chunk", "article_title_chunk"}
        and bool(chunk.get("article_no_normalized"))
        and not chunk.get("contains_table")
        and not chunk.get("appendix")
    )


def build_artifacts(args: argparse.Namespace) -> dict[str, Any]:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metadata_by_id, metadata_report = load_business_metadata(
        args.vld_root,
        min_year=args.min_year,
        ids_file=args.ids_file,
        limit=args.limit,
        effect_metadata_path=args.effect_metadata_path,
    )
    wanted_ids = set(metadata_by_id)

    metadata_out_path = args.output_dir / "metadata.jsonl"
    ids_out_path = args.output_dir / "document_ids.txt"
    articles_path = args.output_dir / "articles.jsonl"
    chunks_path = args.output_dir / "retrieval_units.jsonl"
    preview_path = args.output_dir / "qdrant_payload_preview.jsonl"
    errors_path = args.output_dir / "build_errors.jsonl"
    preview_parts_dir = args.output_dir / "qdrant_payload_preview.parts"
    article_parts_dir = args.output_dir / "articles.parts"
    chunk_parts_dir = args.output_dir / "retrieval_units.parts"

    if args.force_rebuild:
        for path in (preview_parts_dir, article_parts_dir, chunk_parts_dir):
            if path.exists():
                shutil.rmtree(path)
        for path in (articles_path, chunks_path, preview_path, errors_path):
            if path.exists():
                path.unlink()
    for path in (preview_parts_dir, article_parts_dir, chunk_parts_dir):
        path.mkdir(parents=True, exist_ok=True)

    ids_out_path.write_text("\n".join(str(doc_id) for doc_id in sorted(wanted_ids)) + "\n", encoding="utf-8")
    with metadata_out_path.open("w", encoding="utf-8", newline="\n") as handle:
        for doc_id in sorted(wanted_ids):
            handle.write(json.dumps(clean_metadata(metadata_by_id[doc_id]), ensure_ascii=False) + "\n")

    resumed_documents = count_part_files(preview_parts_dir)
    if resumed_documents:
        print(f"Resuming VLD build from {resumed_documents} completed document parts", flush=True)

    documents_built_this_run = 0
    non_article_chunks_dropped = 0
    documents_without_article_chunks = 0
    for parquet_path in sorted((args.vld_root / "content").glob("*.parquet")):
        for content_row in iter_parquet_rows(parquet_path, columns=["id", "content"]):
            doc_id = int(content_row["id"])
            metadata = metadata_by_id.get(doc_id)
            if metadata is None:
                continue
            preview_part_path = document_part_path(preview_parts_dir, doc_id)
            if preview_part_path.exists():
                continue
            try:
                document = build_tree(metadata, str(content_row.get("content") or ""))
                chunks = build_chunks_for_tree(
                    document,
                    max_text_tokens=args.max_text_tokens,
                    table_rows_per_chunk=args.table_rows_per_chunk,
                    overlap_tokens=args.overlap_tokens,
                )
                if args.article_only:
                    original_chunk_count = len(chunks)
                    chunks = [chunk for chunk in chunks if is_article_embedding_chunk(chunk)]
                    non_article_chunks_dropped += original_chunk_count - len(chunks)
                    if not chunks:
                        documents_without_article_chunks += 1
                        continue
                    chunk_count = len(chunks)
                    for chunk_index, chunk in enumerate(chunks):
                        chunk["chunk_index"] = chunk_index
                        chunk["chunk_count"] = chunk_count
            except Exception as exc:  # Keep long Kaggle jobs moving.
                append_jsonl(errors_path, [{"document_id": doc_id, "error": str(exc)}])
                continue

            for chunk in chunks:
                chunk["business_scope_tier"] = metadata.get("_business_scope_tier")
            preview_rows = [make_qdrant_preview(chunk) for chunk in chunks]
            write_jsonl_atomic(preview_part_path, preview_rows)
            if not args.preview_only:
                write_jsonl_atomic(
                    document_part_path(article_parts_dir, doc_id),
                    [
                        {
                            **document["metadata"],
                            "business_scope_tier": metadata.get("_business_scope_tier"),
                            "chunk_count": len(chunks),
                        }
                    ],
                )
                write_jsonl_atomic(document_part_path(chunk_parts_dir, doc_id), chunks)

            documents_built_this_run += 1
            done = resumed_documents + documents_built_this_run
            if args.progress_every > 0 and (documents_built_this_run == 1 or done % args.progress_every == 0):
                print(f"Built/resumed {done} documents", flush=True)

    merge_jsonl_parts(preview_parts_dir, preview_path)
    if not args.preview_only:
        merge_jsonl_parts(article_parts_dir, articles_path)
        merge_jsonl_parts(chunk_parts_dir, chunks_path)

    documents_built, chunk_count, chunk_type_counts, table_chunk_count = summarize_preview_parts(preview_parts_dir)
    error_count, errors_preview = summarize_error_file(errors_path)

    report = {
        "dataset": "vietnamese-legal-documents",
        "vld_root": str(args.vld_root),
        "output_dir": str(args.output_dir),
        "filter": {
            "min_year": args.min_year,
            "limit": args.limit,
            "ids_file": str(args.ids_file) if args.ids_file else "",
            "effect_metadata_path": str(args.effect_metadata_path) if args.effect_metadata_path else "",
        },
        "chunking": {
            "tokenizer": "tiktoken/cl100k_base",
            "max_text_tokens": args.max_text_tokens,
            "overlap_tokens": args.overlap_tokens,
            "table_rows_per_chunk": args.table_rows_per_chunk,
            "article_only": args.article_only,
        },
        "metadata": metadata_report,
        "documents_built": documents_built,
        "documents_built_this_run": documents_built_this_run,
        "documents_resumed_from_checkpoint": resumed_documents,
        "documents_without_article_chunks": documents_without_article_chunks,
        "non_article_chunks_dropped": non_article_chunks_dropped,
        "chunk_count": chunk_count,
        "chunk_type_counts": dict(chunk_type_counts),
        "table_chunk_count": table_chunk_count,
        "error_count": error_count,
        "errors_preview": errors_preview,
        "checkpoint": {
            "mode": "document_part_files",
            "qdrant_payload_preview_parts": str(preview_parts_dir),
            "articles_parts": "" if args.preview_only else str(article_parts_dir),
            "retrieval_units_parts": "" if args.preview_only else str(chunk_parts_dir),
        },
        "outputs": {
            "metadata": str(metadata_out_path),
            "document_ids": str(ids_out_path),
            "articles": "" if args.preview_only else str(articles_path),
            "retrieval_units": "" if args.preview_only else str(chunks_path),
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


def document_part_path(parts_dir: Path, doc_id: int) -> Path:
    return parts_dir / f"{doc_id}.jsonl"


def write_jsonl_atomic(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()
    tmp_path.replace(path)


def append_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()


def merge_jsonl_parts(parts_dir: Path, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8", newline="\n") as output:
        for part_path in sorted_part_files(parts_dir):
            with part_path.open("r", encoding="utf-8") as part:
                shutil.copyfileobj(part, output)
        output.flush()
    tmp_path.replace(output_path)


def summarize_preview_parts(parts_dir: Path) -> tuple[int, int, Counter[str], int]:
    chunk_type_counts: Counter[str] = Counter()
    chunk_count = 0
    table_chunk_count = 0
    document_count = 0
    for part_path in sorted_part_files(parts_dir):
        document_count += 1
        with part_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                payload = row.get("payload") or {}
                chunk_count += 1
                chunk_type_counts[payload.get("chunk_type") or "<missing>"] += 1
                table_chunk_count += int(bool(payload.get("contains_table")))
    return document_count, chunk_count, chunk_type_counts, table_chunk_count


def summarize_error_file(path: Path) -> tuple[int, list[dict[str, Any]]]:
    if not path.exists():
        return 0, []
    count = 0
    preview: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            count += 1
            if len(preview) < 20:
                preview.append(row)
    return count, preview


def count_part_files(parts_dir: Path) -> int:
    return sum(1 for _ in sorted_part_files(parts_dir))


def sorted_part_files(parts_dir: Path) -> list[Path]:
    def sort_key(path: Path) -> tuple[int, str]:
        try:
            return int(path.stem), path.stem
        except ValueError:
            return sys.maxsize, path.stem

    return sorted(parts_dir.glob("*.jsonl"), key=sort_key)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vld-root", type=Path, default=Path("data/vietnamese-legal-documents"))
    parser.add_argument("--output-dir", type=Path, default=Path("build/vld_business_scope"))
    parser.add_argument("--min-year", type=int, default=2010)
    parser.add_argument("--max-text-tokens", type=int, default=2048)
    parser.add_argument("--overlap-tokens", type=int, default=256)
    parser.add_argument("--table-rows-per-chunk", type=int, default=8)
    parser.add_argument("--article-only", action=argparse.BooleanOptionalAction, default=True, help="Only write article text chunks for embedding; drop tables, appendices, and lower-level fallback chunks.")
    parser.add_argument("--ids-file", type=Path)
    parser.add_argument(
        "--effect-metadata-path",
        type=Path,
        default=Path("data/vietnam-legal-documentv2/legacy/metadata.parquet"),
        help="Optional v2 legacy metadata parquet used to enrich effect_status by document id.",
    )
    parser.add_argument("--limit", type=int, default=0, help="Optional document limit for smoke tests.")
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument("--preview-only", action="store_true", help="Only write qdrant_payload_preview.jsonl.")
    parser.add_argument("--force-rebuild", action="store_true", help="Ignore existing document checkpoints.")
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
