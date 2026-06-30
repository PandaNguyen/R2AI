"""Find VLD documents newly admitted by extra business-law keywords.

The current business-law filter lives in ``r2ai.data_ingest.vld.business_collection``.
This helper keeps that filter untouched, then checks whether extra support
keywords would admit additional metadata rows. It writes a delta ids file that
can be passed to ``r2ai.data_ingest.vld.business_qdrant --ids-file`` for a
separate upsert.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from r2ai.data_ingest.vld.business_collection import (  # noqa: E402
    CORE_KEYWORDS,
    SUPPORT_KEYWORDS,
    SUPPORT_LEGAL_TYPES,
    classify_metadata,
    compact,
    effect_status,
    is_expired_document,
    is_local_document,
    iter_parquet_rows,
    load_effect_status_by_id,
    lower,
    merge_effect_status,
    normalize_metadata_row,
    parse_year,
)


DEFAULT_EXTRA_KEYWORDS = [
    "quyền",
    "thực hiện",
    "cấp",
    "nộp",
    "sử dụng",
    "tiền",
    "phạt",
    "phí",
    "cung cấp",
    "sở hữu",
]


def read_ids(path: Path | None) -> set[int]:
    if path is None or not path.exists():
        return set()
    ids: set[int] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        for part in line.replace(",", " ").split():
            if part:
                ids.add(int(part))
    return ids


def split_sectors(value: Any) -> list[str]:
    text = compact(value)
    if not text:
        return []
    return [part.strip() for part in text.replace("|", ",").split(",") if part.strip()]


def search_text_for_row(row: dict[str, Any]) -> str:
    title = lower(row.get("title"))
    sectors = lower(row.get("legal_sectors"))
    document_number = compact(row.get("document_number"))
    return f"{document_number} {title} {sectors}"


def metadata_delta(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    metadata_path = args.vld_root / "metadata" / "data-00000-of-00001.parquet"
    current_ids = read_ids(args.current_ids_file)
    effect_status_by_id = load_effect_status_by_id(args.effect_metadata_path)
    extra_keywords = [keyword.lower() for keyword in args.extra_keyword]
    active_keywords = {keyword.lower() for keyword in CORE_KEYWORDS + SUPPORT_KEYWORDS}

    delta: list[dict[str, Any]] = []
    matched_keyword_counts: Counter[str] = Counter()
    matched_new_keyword_counts: Counter[str] = Counter()
    legal_type_counts: Counter[str] = Counter()
    sector_counts: Counter[str] = Counter()
    excluded_reason_counts: Counter[str] = Counter()
    rows_scanned = 0
    current_filter_keep_count = 0

    for raw_row in iter_parquet_rows(metadata_path):
        rows_scanned += 1
        row = normalize_metadata_row(merge_effect_status(raw_row, effect_status_by_id))
        doc_id = int(row["id"])
        keep, _tier = classify_metadata(row, min_year=args.min_year)
        if keep:
            current_filter_keep_count += 1
            continue
        if doc_id in current_ids:
            excluded_reason_counts["already_in_current_ids"] += 1
            continue
        if is_local_document(row):
            excluded_reason_counts["local"] += 1
            continue
        if is_expired_document(row):
            excluded_reason_counts["expired"] += 1
            continue

        hits = [keyword for keyword in extra_keywords if keyword in search_text_for_row(row)]
        if not hits:
            excluded_reason_counts["no_extra_keyword"] += 1
            continue

        legal_type = lower(row.get("legal_type"))
        if legal_type not in SUPPORT_LEGAL_TYPES:
            excluded_reason_counts["keyword_but_bad_type"] += 1
            continue

        year = parse_year(row.get("issuance_date"))
        if year is not None and year < args.min_year:
            excluded_reason_counts["keyword_but_old"] += 1
            continue

        new_hits = [keyword for keyword in hits if keyword not in active_keywords]
        if args.require_new_keyword_hit and not new_hits:
            excluded_reason_counts["only_existing_keyword"] += 1
            continue

        record = {
            "id": doc_id,
            "document_number": compact(row.get("document_number")),
            "title": compact(row.get("title")),
            "legal_type": compact(row.get("legal_type")),
            "legal_sectors": compact(row.get("legal_sectors")),
            "issuance_date": compact(row.get("issuance_date")),
            "effect_status": effect_status(row),
            "matched_extra_keywords": hits,
            "matched_new_keywords": new_hits,
        }
        delta.append(record)
        matched_keyword_counts.update(hits)
        matched_new_keyword_counts.update(new_hits)
        legal_type_counts[record["legal_type"] or "<missing>"] += 1
        sector_counts.update(split_sectors(row.get("legal_sectors")))

    report = {
        "mode": "extra_support_keywords_delta",
        "min_year": args.min_year,
        "vld_root": str(args.vld_root),
        "current_ids_file": str(args.current_ids_file) if args.current_ids_file else "",
        "effect_metadata_path": str(args.effect_metadata_path) if args.effect_metadata_path else "",
        "rows_scanned": rows_scanned,
        "current_filter_keep_count": current_filter_keep_count,
        "current_ids_count": len(current_ids),
        "extra_keywords": args.extra_keyword,
        "keywords_already_present_in_filter": [
            keyword for keyword in args.extra_keyword if keyword.lower() in active_keywords
        ],
        "require_new_keyword_hit": args.require_new_keyword_hit,
        "delta_documents": len(delta),
        "matched_keyword_counts_multi_hit": dict(matched_keyword_counts),
        "matched_new_keyword_counts_multi_hit": dict(matched_new_keyword_counts),
        "top_legal_types": dict(legal_type_counts.most_common(25)),
        "top_sectors": dict(sector_counts.most_common(25)),
        "excluded_reason_counts_after_current_filter": dict(excluded_reason_counts),
        "sample": sorted(delta, key=lambda item: item["id"])[: args.sample_size],
    }
    return sorted(delta, key=lambda item: item["id"]), report


def write_outputs(args: argparse.Namespace, delta: list[dict[str, Any]], report: dict[str, Any]) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    ids_path = args.output_dir / "document_ids.txt"
    metadata_path = args.output_dir / "metadata_delta.jsonl"
    report_path = args.output_dir / "keyword_delta_report.json"

    ids_text = "\n".join(str(row["id"]) for row in delta)
    ids_path.write_text(ids_text + ("\n" if ids_text else ""), encoding="utf-8")
    with metadata_path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in delta:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    report["outputs"] = {
        "document_ids": str(ids_path),
        "metadata_delta": str(metadata_path),
        "report": str(report_path),
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vld-root", type=Path, default=Path("data/vietnamese-legal-documents"))
    parser.add_argument("--output-dir", type=Path, default=Path("build/vld_business_keyword_delta"))
    parser.add_argument("--min-year", type=int, default=2000)
    parser.add_argument(
        "--current-ids-file",
        type=Path,
        default=Path("build/vld_business_min2000_tok2048_ov256_tbl8/document_ids.txt"),
        help="Existing business-law ids to exclude from the delta.",
    )
    parser.add_argument(
        "--effect-metadata-path",
        type=Path,
        default=Path("data/vietnam-legal-documentv2/legacy/metadata.parquet"),
    )
    parser.add_argument("--extra-keyword", action="append", default=None)
    parser.add_argument("--sample-size", type=int, default=20)
    parser.add_argument(
        "--allow-existing-keyword-only",
        dest="require_new_keyword_hit",
        action="store_false",
        default=True,
        help="Also include rows that only hit keywords already present in the active filter.",
    )
    return parser


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except AttributeError:
        pass
    args = build_parser().parse_args()
    if args.extra_keyword is None:
        args.extra_keyword = DEFAULT_EXTRA_KEYWORDS
    delta, report = metadata_delta(args)
    write_outputs(args, delta, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
