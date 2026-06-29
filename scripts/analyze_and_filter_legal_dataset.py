"""Analyze and narrow Vietnamese legal datasets for business-law retrieval.

Outputs are UTF-8 JSON/JSONL/TXT artifacts under ``build/business_scope`` by
default. The script keeps metadata filtering cheap and streams large JSONL files
line by line.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

try:
    import pyarrow.parquet as pq
except ImportError as exc:  # pragma: no cover - exercised in local setup.
    raise RuntimeError("pyarrow is required. Run: uv sync --extra data") from exc


CORE_KEYWORDS = [
    "doanh nghiệp",
    "dnnvv",
    "nhỏ và vừa",
    "sme",
    "công ty",
    "đăng ký",
    "kinh doanh",
    "công ty cổ phần",
    "công ty trách nhiệm hữu hạn",
    "tnhh",
    "hộ kinh doanh",
    "hợp tác xã",
    "đăng ký kinh doanh",
    "đăng ký doanh nghiệp",
    "thành lập doanh nghiệp",
    "giải thể doanh nghiệp",
]

SUPPORT_KEYWORDS = [
    "thuế",
    "phí",
    "lệ phí",
    "hóa đơn",
    "xử lý",
    "hồ sơ",
    "nhân viên",
    "cơ quan",
    "yêu cầu",
    "quy định",
    "nội dung",
    "điều kiện",
    "thời hạn",
    "thông tin",
    "hỗ trợ",
    "trách nhiệm",
    "nghĩa vụ",
    "thông báo",
    "hàng hóa",
    "khách hàng",
    "quỹ",
    "giá trị gia tăng",
    "gtgt",
    "vat",
    "thu nhập doanh nghiệp",
    "kế toán",
    "kiểm toán",
    "lao động",
    "người lao động",
    "hợp đồng lao động",
    "bảo hiểm xã hội",
    "bhxh",
    "tiền lương",
    "hợp đồng",
    "giao dịch",
    "đấu thầu",
    "đầu tư",
    "thương mại",
    "chứng khoán",
    "báo cáo tài chính",
    "tài chính doanh nghiệp",
    "tín dụng",
    "ngân hàng",
    "sở hữu trí tuệ",
    "nhãn hiệu",
    "chuyển giao công nghệ",
    "đất đai",
    "thuê đất",
    "mặt bằng",
    "nhà xưởng",
    "phá sản",
    "cạnh tranh",
    "bảo vệ quyền lợi người tiêu dùng",
]

IMPORTANT_LEGAL_TYPES = {
    "bộ luật",
    "luật",
    "pháp lệnh",
    "nghị quyết",
    "nghị định",
    "thông tư",
    "thông tư liên tịch",
    "quyết định",
    "văn bản hợp nhất",
    "code",
    "law",
    "ordinance",
    "resolution",
    "decree",
    "circular",
    "joint circular",
    "decision",
    "integrated document",
    "consolidated document",
}
SUPPORT_LEGAL_TYPES = IMPORTANT_LEGAL_TYPES - {"quyết định", "nghị quyết", "decision", "resolution"}

EXPIRED_EFFECT_STATUS_VALUES = {"expired", "no longer applicable"}
EXPIRED_EFFECT_STATUS_MARKERS = ("hết hiệu lực", "ngưng hiệu lực", "không còn phù hợp")

LOCAL_AUTHORITY_MARKERS = (
    "ủy ban nhân dân",
    "uỷ ban nhân dân",
    "hội đồng nhân dân",
    "ubnd",
    "hđnd",
    "hdnd",
)
LOCAL_TITLE_MARKERS = (
    "do ủy ban nhân dân",
    "do uỷ ban nhân dân",
    "do hội đồng nhân dân",
    "của ủy ban nhân dân",
    "của uỷ ban nhân dân",
    "của hội đồng nhân dân",
)
LOCAL_DOCUMENT_NUMBER_MARKERS = (
    "QĐ-UBND",
    "QD-UBND",
    "QĐ-CTUBND",
    "QD-CTUBND",
    "NQ-HĐND",
    "NQ-HDND",
    "QĐ-HĐND",
    "QD-HDND",
    "CT-UBND",
    "TB-UBND",
    "KH-UBND",
    "CV-UBND",
    "QĐ-UB",
    "QD-UB",
)

CORE_LAW_IDS = {
    "59/2020/QH14",
    "01/2021/NĐ-CP",
    "04/2017/QH14",
    "80/2021/NĐ-CP",
    "07/VBHN-VPQH",
}

SUPPORT_LAW_IDS = {
    "38/2019/QH14",
    "126/2020/NĐ-CP",
    "123/2020/NĐ-CP",
    "125/2020/NĐ-CP",
    "45/2019/QH14",
    "145/2020/NĐ-CP",
    "12/2022/NĐ-CP",
    "58/2014/QH13",
    "91/2015/QH13",
    "36/2005/QH11",
    "22/2023/QH15",
    "24/2024/NĐ-CP",
    "200/2014/TT-BTC",
    "133/2016/TT-BTC",
    "50/2005/QH11",
    "65/2023/NĐ-CP",
    "07/2022/QH15",
    "31/2021/NĐ-CP",
    "10/2024/NĐ-CP",
    "35/2022/NĐ-CP",
}

CORE_PHAPDIEN_TOPICS = {"Doanh nghiệp, hợp tác xã"}
SUPPORT_PHAPDIEN_TOPICS = {
    "Thuế, phí, lệ phí, các khoản thu khác",
    "Kế toán, kiểm toán",
    "Lao động",
    "Bảo hiểm",
    "Dân sự",
    "Thương mại, đầu tư, chứng khoán",
    "Ngân hàng, tiền tệ",
    "Khoa học, công nghệ",
    "Đất đai",
}

YEAR_PATTERN = re.compile(r"(\d{4})")


def normalize(value: Any) -> str:
    return "" if value is None else str(value).strip()


def lower(value: Any) -> str:
    return normalize(value).lower()


def first_value(row: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        value = row.get(key)
        if value is not None and str(value).strip():
            return value
    return None


def joined_values(row: dict[str, Any], *keys: str) -> str:
    return " | ".join(normalize(row.get(key)) for key in keys if normalize(row.get(key)))


def normalize_vld_row(row: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(row)
    aliases = {
        "document_number": ("document_number", "so_ky_hieu"),
        "legal_type": ("legal_type", "loai_van_ban"),
        "legal_sectors": ("legal_sectors", "nganh", "linh_vuc"),
        "issuing_authority": ("issuing_authority", "co_quan_ban_hanh"),
        "issuance_date": ("issuance_date", "ngay_ban_hanh"),
        "signers": ("signers", "nguoi_ky"),
        "effect_status": ("effect_status", "tinh_trang_hieu_luc"),
        "effectless_date": ("effectless_date", "ngay_het_hieu_luc"),
    }
    for target, keys in aliases.items():
        if target == "legal_sectors":
            value = joined_values(row, *keys)
        else:
            value = first_value(row, *keys)
        if value is not None and str(value).strip() and not normalize(normalized.get(target)):
            normalized[target] = value
    return normalized


def effect_status(row: dict[str, Any]) -> str:
    return normalize(first_value(row, "effect_status", "tinh_trang_hieu_luc"))


def is_expired_document(row: dict[str, Any]) -> bool:
    status = lower(effect_status(row))
    if not status:
        return False
    return status in EXPIRED_EFFECT_STATUS_VALUES or any(marker in status for marker in EXPIRED_EFFECT_STATUS_MARKERS)


def is_local_document(row: dict[str, Any]) -> bool:
    authority = lower(row.get("issuing_authority"))
    title = lower(row.get("title"))
    document_number = normalize(row.get("document_number")).upper()
    return (
        any(marker in authority for marker in LOCAL_AUTHORITY_MARKERS)
        or any(marker in title for marker in LOCAL_TITLE_MARKERS)
        or any(marker in document_number for marker in LOCAL_DOCUMENT_NUMBER_MARKERS)
    )


def split_sectors(value: Any) -> list[str]:
    text = normalize(value)
    return [part.strip() for part in text.split("|") if part.strip()]


def parse_year(value: Any) -> int | None:
    matches = YEAR_PATTERN.findall(normalize(value))
    if not matches:
        return None
    year = int(matches[-1])
    return year if 1800 <= year <= 2100 else None


def contains_any(text: str, keywords: Iterable[str]) -> bool:
    return any(keyword.lower() in text for keyword in keywords)


def law_id_hit(text: str, law_ids: set[str]) -> bool:
    folded = text.upper().replace("ND-CP", "NĐ-CP").replace("QD-", "QĐ-")
    return any(law_id.upper() in folded for law_id in law_ids)


def top(counter: Counter[str], limit: int = 25) -> dict[str, int]:
    return {key: value for key, value in counter.most_common(limit)}


def pct(part: int, total: int) -> float:
    return round(part * 100 / total, 4) if total else 0.0


def iter_parquet_rows(path: Path, columns: list[str] | None = None) -> Iterable[dict[str, Any]]:
    pf = pq.ParquetFile(path)
    for batch in pf.iter_batches(columns=columns):
        table = batch.to_pydict()
        keys = list(table)
        for values in zip(*(table[key] for key in keys), strict=True):
            yield dict(zip(keys, values, strict=True))


def load_effect_status_by_id(effect_metadata_path: Path | None) -> dict[int, str]:
    if effect_metadata_path is None or not effect_metadata_path.exists():
        return {}

    pf = pq.ParquetFile(effect_metadata_path)
    names = set(pf.schema_arrow.names)
    status_column = "effect_status" if "effect_status" in names else "tinh_trang_hieu_luc" if "tinh_trang_hieu_luc" in names else ""
    if not status_column or "id" not in names:
        return {}

    statuses: dict[int, str] = {}
    for row in iter_parquet_rows(effect_metadata_path, columns=["id", status_column]):
        status = normalize(row.get(status_column))
        if status:
            statuses[int(row["id"])] = status
    return statuses


def merge_effect_status(row: dict[str, Any], effect_status_by_id: dict[int, str]) -> dict[str, Any]:
    if not effect_status_by_id or effect_status(row):
        return row
    status = effect_status_by_id.get(int(row["id"]))
    if not status:
        return row
    merged = dict(row)
    merged["effect_status"] = status
    return merged


def resolve_vld_metadata_path(vld_root: Path) -> Path:
    if vld_root.is_file():
        return vld_root
    candidates = [
        vld_root / "metadata" / "data-00000-of-00001.parquet",
        vld_root / "data" / "metadata.parquet",
        vld_root / "legacy" / "metadata.parquet",
        vld_root / "metadata.parquet",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Could not find VLD metadata parquet under {vld_root}")


def classify_vld_row(row: dict[str, Any], min_year: int) -> tuple[bool, str]:
    row = normalize_vld_row(row)
    if is_local_document(row):
        return False, "local"
    if is_expired_document(row):
        return False, "expired"

    title = lower(row.get("title"))
    sectors = lower(row.get("legal_sectors"))
    document_number = normalize(row.get("document_number"))
    legal_type = lower(row.get("legal_type"))
    year = parse_year(row.get("issuance_date"))
    search_text = f"{document_number} {title} {sectors}"

    core_hit = contains_any(search_text, CORE_KEYWORDS)
    support_hit = contains_any(search_text, SUPPORT_KEYWORDS)
    direct_core_law = law_id_hit(document_number, CORE_LAW_IDS)
    direct_support_law = law_id_hit(document_number, SUPPORT_LAW_IDS)
    important_type = legal_type in IMPORTANT_LEGAL_TYPES
    recent_enough = year is None or year >= min_year

    support_type = legal_type in SUPPORT_LEGAL_TYPES

    if direct_core_law or (core_hit and important_type and recent_enough):
        return True, "core"
    if direct_support_law or (support_hit and support_type and recent_enough):
        return True, "support"
    return False, "excluded"


def analyze_vld_metadata(
    metadata_path: Path,
    output_dir: Path,
    min_year: int,
    effect_metadata_path: Path | None = None,
) -> dict[str, Any]:
    pf = pq.ParquetFile(metadata_path)
    total = pf.metadata.num_rows
    effect_status_by_id = load_effect_status_by_id(effect_metadata_path)
    type_counts: Counter[str] = Counter()
    sector_counts: Counter[str] = Counter()
    year_counts: Counter[str] = Counter()
    authority_counts: Counter[str] = Counter()
    kept_type_counts: Counter[str] = Counter()
    kept_sector_counts: Counter[str] = Counter()
    kept_year_counts: Counter[str] = Counter()
    effect_status_counts: Counter[str] = Counter()
    kept_effect_status_counts: Counter[str] = Counter()
    excluded_reason_counts: Counter[str] = Counter()
    tier_counts: Counter[str] = Counter()
    kept_ids: list[int] = []

    metadata_jsonl = output_dir / "vld_business_metadata.jsonl"
    ids_txt = output_dir / "vld_business_ids.txt"
    with metadata_jsonl.open("w", encoding="utf-8", newline="\n") as meta_out:
        for batch in pf.iter_batches():
            table = batch.to_pydict()
            keys = list(table)
            for values in zip(*(table[key] for key in keys), strict=True):
                row = normalize_vld_row(merge_effect_status(dict(zip(keys, values, strict=True)), effect_status_by_id))
                legal_type = normalize(row.get("legal_type")) or "<missing>"
                type_counts[legal_type] += 1
                for sector in split_sectors(row.get("legal_sectors")):
                    sector_counts[sector] += 1
                year = parse_year(row.get("issuance_date"))
                year_counts[str(year) if year is not None else "<missing>"] += 1
                authority = normalize(row.get("issuing_authority")) or "<missing>"
                authority_counts[authority] += 1
                status_value = effect_status(row) or "<missing>"
                effect_status_counts[status_value] += 1

                keep, tier = classify_vld_row(row, min_year=min_year)
                if not keep:
                    excluded_reason_counts[tier] += 1
                    continue
                tier_counts[tier] += 1
                kept_id = int(row["id"])
                kept_ids.append(kept_id)
                kept_type_counts[legal_type] += 1
                for sector in split_sectors(row.get("legal_sectors")):
                    kept_sector_counts[sector] += 1
                kept_year_counts[str(year) if year is not None else "<missing>"] += 1
                kept_effect_status_counts[status_value] += 1
                meta_out.write(json.dumps(row, ensure_ascii=False) + "\n")

    ids_txt.write_text("\n".join(str(doc_id) for doc_id in sorted(kept_ids)) + "\n", encoding="utf-8")

    return {
        "source": str(metadata_path),
        "total_documents": total,
        "kept_documents": len(kept_ids),
        "reduction": {
            "kept_percent": pct(len(kept_ids), total),
            "excluded_percent": round(100 - pct(len(kept_ids), total), 4),
        },
        "filter": {
            "min_year": min_year,
            "important_legal_types": sorted(IMPORTANT_LEGAL_TYPES),
            "core_keywords": CORE_KEYWORDS,
            "support_keywords": SUPPORT_KEYWORDS,
            "core_law_ids": sorted(CORE_LAW_IDS),
            "support_law_ids": sorted(SUPPORT_LAW_IDS),
            "effect_metadata_path": str(effect_metadata_path) if effect_metadata_path else "",
            "effect_metadata_loaded": bool(effect_status_by_id),
            "drop_expired_status_values": sorted(EXPIRED_EFFECT_STATUS_VALUES),
            "drop_expired_status_markers": list(EXPIRED_EFFECT_STATUS_MARKERS),
            "drop_local_authority_markers": list(LOCAL_AUTHORITY_MARKERS),
            "drop_local_title_markers": list(LOCAL_TITLE_MARKERS),
            "drop_local_document_number_markers": list(LOCAL_DOCUMENT_NUMBER_MARKERS),
        },
        "tier_counts": dict(tier_counts),
        "excluded_reason_counts": dict(excluded_reason_counts),
        "top_legal_types_all": top(type_counts),
        "top_legal_types_kept": top(kept_type_counts),
        "top_sectors_all": top(sector_counts),
        "top_sectors_kept": top(kept_sector_counts),
        "top_years_all": top(year_counts),
        "top_years_kept": top(kept_year_counts),
        "top_authorities_all": top(authority_counts),
        "top_effect_status_all": top(effect_status_counts),
        "top_effect_status_kept": top(kept_effect_status_counts),
        "outputs": {
            "metadata_jsonl": str(metadata_jsonl),
            "ids_txt": str(ids_txt),
        },
    }


def analyze_vld_content(content_dir: Path, kept_ids_path: Path) -> dict[str, Any]:
    kept_ids = {int(line) for line in kept_ids_path.read_text(encoding="utf-8").splitlines() if line.strip()}
    if not kept_ids:
        return {"scanned": False, "reason": "no kept ids"}

    total_chars = 0
    max_chars = 0
    docs_found = 0
    estimated_1600_char_chunks = 0
    length_buckets: Counter[str] = Counter()

    for parquet_path in sorted(content_dir.glob("*.parquet")):
        pf = pq.ParquetFile(parquet_path)
        for batch in pf.iter_batches(columns=["id", "content"]):
            table = batch.to_pydict()
            for doc_id, content in zip(table["id"], table["content"], strict=True):
                if int(doc_id) not in kept_ids:
                    continue
                text = normalize(content)
                char_len = len(text)
                docs_found += 1
                total_chars += char_len
                max_chars = max(max_chars, char_len)
                estimated_1600_char_chunks += max(1, math.ceil(char_len / 1600))
                if char_len < 2_000:
                    length_buckets["<2k"] += 1
                elif char_len < 10_000:
                    length_buckets["2k-10k"] += 1
                elif char_len < 50_000:
                    length_buckets["10k-50k"] += 1
                else:
                    length_buckets[">=50k"] += 1

    return {
        "scanned": True,
        "docs_found": docs_found,
        "total_chars": total_chars,
        "avg_chars_per_doc": round(total_chars / docs_found, 2) if docs_found else 0,
        "max_chars": max_chars,
        "estimated_chunks_at_1600_chars": estimated_1600_char_chunks,
        "estimated_points_per_doc": round(estimated_1600_char_chunks / docs_found, 2) if docs_found else 0,
        "length_buckets": dict(length_buckets),
    }


def classify_phapdien_payload(payload: dict[str, Any]) -> tuple[bool, str]:
    topic = normalize(payload.get("topic_title"))
    subject = normalize(payload.get("subject_title"))
    source_note = lower(payload.get("source_note_text"))
    doc_titles = " ".join(str(item) for item in payload.get("source_doc_title_candidates") or [])
    content_tree_path = " ".join(str(item) for item in payload.get("content_tree_path") or [])
    article_title = lower(payload.get("article_title"))
    law_ids = payload.get("source_law_id_candidates") or []
    law_id_text = " ".join(str(item) for item in law_ids)
    metadata_scope = f"{lower(topic)} {lower(subject)} {source_note} {lower(doc_titles)} {lower(content_tree_path)} {article_title}"

    core_hit = topic in CORE_PHAPDIEN_TOPICS or contains_any(metadata_scope, CORE_KEYWORDS) or law_id_hit(law_id_text, CORE_LAW_IDS)
    support_hit = topic in SUPPORT_PHAPDIEN_TOPICS or contains_any(metadata_scope, SUPPORT_KEYWORDS) or law_id_hit(law_id_text, SUPPORT_LAW_IDS)
    if core_hit:
        return True, "core"
    if support_hit:
        return True, "support"
    return False, "excluded"


def filter_jsonl_by_payload(
    input_path: Path,
    output_path: Path,
    payload_key: str | None,
) -> dict[str, Any]:
    if not input_path.exists():
        return {"source": str(input_path), "exists": False}

    total = 0
    kept = 0
    tier_counts: Counter[str] = Counter()
    topic_counts: Counter[str] = Counter()
    subject_counts: Counter[str] = Counter()
    law_id_counts: Counter[str] = Counter()

    with input_path.open("r", encoding="utf-8") as src, output_path.open("w", encoding="utf-8", newline="\n") as dst:
        for line in src:
            if not line.strip():
                continue
            total += 1
            row = json.loads(line)
            payload = row[payload_key] if payload_key else row
            keep, tier = classify_phapdien_payload(payload)
            if not keep:
                continue
            kept += 1
            tier_counts[tier] += 1
            topic_counts[normalize(payload.get("topic_title")) or "<missing>"] += 1
            subject_counts[normalize(payload.get("subject_title")) or "<missing>"] += 1
            for law_id in payload.get("source_law_id_candidates") or []:
                law_id_counts[str(law_id)] += 1
            dst.write(json.dumps(row, ensure_ascii=False) + "\n")

    return {
        "source": str(input_path),
        "output": str(output_path),
        "exists": True,
        "total_rows": total,
        "kept_rows": kept,
        "kept_percent": pct(kept, total),
        "tier_counts": dict(tier_counts),
        "top_topics_kept": top(topic_counts),
        "top_subjects_kept": top(subject_counts),
        "top_law_ids_kept": top(law_id_counts),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vld-root", type=Path, default=Path("data/vietnamese-legal-documents"))
    parser.add_argument("--phapdien-build-dir", type=Path, default=Path("build"))
    parser.add_argument("--output-dir", type=Path, default=Path("build/business_scope"))
    parser.add_argument("--min-year", type=int, default=2010)
    parser.add_argument(
        "--effect-metadata-path",
        type=Path,
        default=Path("data/vietnam-legal-documentv2/legacy/metadata.parquet"),
        help="Optional v2 metadata parquet used to enrich effect_status by document id.",
    )
    parser.add_argument("--scan-content", action="store_true")
    return parser


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except AttributeError:
        pass

    args = build_parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    metadata_path = resolve_vld_metadata_path(args.vld_root)
    report: dict[str, Any] = {
        "vietnamese_legal_documents": analyze_vld_metadata(
            metadata_path,
            args.output_dir,
            min_year=args.min_year,
            effect_metadata_path=args.effect_metadata_path,
        ),
        "phapdien_business_scope": {},
    }
    if args.scan_content:
        report["vietnamese_legal_documents"]["content_scan"] = analyze_vld_content(
            args.vld_root / "content",
            Path(report["vietnamese_legal_documents"]["outputs"]["ids_txt"]),
        )

    retrieval_units_out = args.output_dir / "retrieval_units.jsonl"
    qdrant_preview_out = args.output_dir / "qdrant_payload_preview.jsonl"
    report["phapdien_business_scope"]["retrieval_units"] = filter_jsonl_by_payload(
        args.phapdien_build_dir / "retrieval_units.jsonl",
        retrieval_units_out,
        payload_key=None,
    )
    report["phapdien_business_scope"]["qdrant_payload_preview"] = filter_jsonl_by_payload(
        args.phapdien_build_dir / "qdrant_payload_preview.jsonl",
        qdrant_preview_out,
        payload_key="payload",
    )

    report_path = args.output_dir / "filter_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
