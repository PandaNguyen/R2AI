"""Build a business-focused Qdrant artifact from vietnamese-legal-documents.

This script is intentionally independent from the phapdien pipeline. It reads
the VLD metadata/content parquet files, filters documents around business law,
chunks the selected content, and writes Qdrant preview rows that can be ingested
as a separate recall-boosting collection.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
import uuid
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator

try:
    import pyarrow.parquet as pq
except ImportError as exc:  # pragma: no cover - local setup guard.
    raise RuntimeError("pyarrow is required. Run: uv sync --extra data") from exc

from r2ai.indexing.config import DEFAULT_DENSE_MODEL, DEFAULT_SPARSE_MODEL


DEFAULT_MAX_CHUNK_TOKENS = 2048
DEFAULT_CHUNK_OVERLAP_TOKENS = 256
CONTENT_PREVIEW_CHARS = 500
POINT_NAMESPACE = uuid.UUID("8fb0e0da-8ba3-44fc-b0a1-4b52bb8f5f78")

CORE_KEYWORDS = [
    "doanh nghiệp",
    "dnnvv",
    "nhỏ và vừa",
    "sme",
    "công ty",
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
}
SUPPORT_LEGAL_TYPES = IMPORTANT_LEGAL_TYPES - {"quyết định", "nghị quyết"}

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

ARTICLE_HEADING_PATTERN = re.compile(r"(?im)^\s*(Điều|Article)\s+\d+[^\n\r]*")
ARTICLE_NO_PATTERN = re.compile(r"\b(?:Điều|Article)\s+\d+[A-Za-z]?", re.IGNORECASE)
SENTENCE_SPLIT_PATTERN = re.compile(r"(?<=[.!?。])\s+")
YEAR_PATTERN = re.compile(r"(\d{4})")
WHITESPACE_PATTERN = re.compile(r"[ \t\r\f\v]+")
VIETNAMESE_MARKER_PATTERN = re.compile(r"[àáảãạăằắẳẵặâầấẩẫậèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵđ]", re.IGNORECASE)
BOILERPLATE_PATTERNS = [
    re.compile(r"Bạn phải đăng nhập hoặc đăng ký Thành Viên TVPL Pro.*?(?=\n|$)", re.IGNORECASE),
    re.compile(r"Mọi chi tiết xin liên hệ:.*?(?=\n|$)", re.IGNORECASE),
    re.compile(r"^\s*\.{3,}\s*$", re.MULTILINE),
]


@dataclass(frozen=True)
class Chunk:
    text: str
    article_no: str
    heading: str
    method: str


def normalize(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).replace("\u00a0", " ")
    text = WHITESPACE_PATTERN.sub(" ", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def compact(value: Any) -> str:
    return re.sub(r"\s+", " ", normalize(value)).strip()


def lower(value: Any) -> str:
    return compact(value).lower()


def parse_year(value: Any) -> int | None:
    matches = YEAR_PATTERN.findall(compact(value))
    if not matches:
        return None
    year = int(matches[-1])
    return year if 1800 <= year <= 2100 else None


def contains_any(text: str, keywords: Iterable[str]) -> bool:
    return any(keyword.lower() in text for keyword in keywords)


def law_id_hit(text: str, law_ids: set[str]) -> bool:
    folded = text.upper().replace("ND-CP", "NĐ-CP").replace("QD-", "QĐ-")
    return any(law_id.upper() in folded for law_id in law_ids)


def split_sectors(value: Any) -> list[str]:
    text = compact(value)
    if not text:
        return []
    parts = re.split(r"\s*\|\s*|\s*,\s*", text)
    return [part.strip() for part in parts if part.strip()]


def classify_metadata(row: dict[str, Any], min_year: int) -> tuple[bool, str]:
    title = lower(row.get("title"))
    sectors = lower(row.get("legal_sectors"))
    document_number = compact(row.get("document_number"))
    legal_type = lower(row.get("legal_type"))
    year = parse_year(row.get("issuance_date"))
    search_text = f"{document_number} {title} {sectors}"

    core_hit = contains_any(search_text, CORE_KEYWORDS)
    support_hit = contains_any(search_text, SUPPORT_KEYWORDS)
    direct_core_law = law_id_hit(document_number, CORE_LAW_IDS)
    direct_support_law = law_id_hit(document_number, SUPPORT_LAW_IDS)
    important_type = legal_type in IMPORTANT_LEGAL_TYPES
    support_type = legal_type in SUPPORT_LEGAL_TYPES
    recent_enough = year is None or year >= min_year

    if direct_core_law or (core_hit and important_type and recent_enough):
        return True, "core"
    if direct_support_law or (support_hit and support_type and recent_enough):
        return True, "support"
    return False, "excluded"


def count_tokens(text: str) -> int:
    text = compact(text)
    if not text:
        return 0
    # A rough estimate is enough for pre-index chunk sizing and keeps the VLD
    # build fast on hundreds of MB of selected legal text.
    return max(1, int(len(text.split()) * 1.35))


def clean_content(text: Any) -> str:
    cleaned = normalize(text)
    for pattern in BOILERPLATE_PATTERNS:
        cleaned = pattern.sub("", cleaned)
    lines = [line.strip() for line in cleaned.splitlines()]
    lines = [line for line in lines if line and line != "..."]
    return normalize("\n".join(lines))


def vietnamese_marker_ratio(text: str) -> float:
    letters = sum(1 for char in text if char.isalpha())
    if not letters:
        return 0.0
    return len(VIETNAMESE_MARKER_PATTERN.findall(text)) / letters


def chunk_document(text: str, max_tokens: int, overlap_tokens: int) -> list[Chunk]:
    text = clean_content(text)
    if not text:
        return []

    matches = list(ARTICLE_HEADING_PATTERN.finditer(text))
    if not matches:
        return [
            Chunk(text=chunk, article_no="", heading="", method="text")
            for chunk in split_text_by_budget(text, max_tokens=max_tokens, overlap_tokens=overlap_tokens)
        ]

    preamble = text[: matches[0].start()].strip()
    chunks: list[Chunk] = []
    if preamble and count_tokens(preamble) > 30:
        chunks.extend(
            Chunk(text=chunk, article_no="", heading="Preamble", method="preamble")
            for chunk in split_text_by_budget(preamble, max_tokens=max_tokens, overlap_tokens=overlap_tokens)
        )

    for index, match in enumerate(matches):
        start = match.start()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        section = text[start:end].strip()
        heading = compact(match.group(0))
        article_no = extract_article_no(heading)
        parts = split_text_by_budget(section, max_tokens=max_tokens, overlap_tokens=overlap_tokens)
        method = "article" if len(parts) == 1 else "article_split"
        chunks.extend(Chunk(text=part, article_no=article_no, heading=heading, method=method) for part in parts)
    return chunks


def split_text_by_budget(text: str, max_tokens: int, overlap_tokens: int) -> list[str]:
    text = compact(text)
    if not text:
        return []
    if count_tokens(text) <= max_tokens:
        return [text]

    sentences = [part.strip() for part in SENTENCE_SPLIT_PATTERN.split(text) if part.strip()]
    if len(sentences) <= 1:
        return split_words_by_budget(text, max_tokens=max_tokens, overlap_tokens=overlap_tokens)

    chunks: list[str] = []
    current: list[str] = []
    current_tokens = 0
    for sentence in sentences:
        sentence_tokens = count_tokens(sentence)
        if current_tokens + sentence_tokens <= max_tokens or not current:
            current.append(sentence)
            current_tokens += sentence_tokens
            continue
        chunks.append(" ".join(current).strip())
        current = overlap_tail(current, overlap_tokens)
        current_tokens = sum(count_tokens(part) for part in current)
        current.append(sentence)
        current_tokens += sentence_tokens
    if current:
        chunks.append(" ".join(current).strip())
    return chunks


def split_words_by_budget(text: str, max_tokens: int, overlap_tokens: int) -> list[str]:
    words = text.split()
    max_words = max(1, int(max_tokens / 1.35))
    overlap_words = max(0, int(overlap_tokens / 1.35))
    if len(words) <= max_words:
        return [" ".join(words)]

    chunks: list[str] = []
    step = max(1, max_words - overlap_words)
    for start in range(0, len(words), step):
        chunk_words = words[start : start + max_words]
        if not chunk_words:
            break
        chunks.append(" ".join(chunk_words).strip())
        if start + max_words >= len(words):
            break
    return chunks


def overlap_tail(parts: list[str], overlap_tokens: int) -> list[str]:
    if overlap_tokens <= 0:
        return []
    tail: list[str] = []
    tail_tokens = 0
    for part in reversed(parts):
        part_tokens = count_tokens(part)
        if tail_tokens + part_tokens > overlap_tokens and tail:
            break
        tail = [part] + tail
        tail_tokens += part_tokens
    return tail


def extract_article_no(text: str) -> str:
    match = ARTICLE_NO_PATTERN.search(text)
    if not match:
        return ""
    value = match.group(0)
    return value.replace("article", "Article").replace("điều", "Điều")


def make_point_id(document_id: int, chunk_index: int) -> str:
    return str(uuid.uuid5(POINT_NAMESPACE, f"vld:{document_id}:chunk:{chunk_index:05d}"))


def source_law_ids(row: dict[str, Any]) -> list[str]:
    document_number = compact(row.get("document_number"))
    return [document_number] if document_number else []


def build_retrieval_text(row: dict[str, Any], chunk: Chunk) -> str:
    document_number = compact(row.get("document_number"))
    title = compact(row.get("title"))
    legal_type = compact(row.get("legal_type"))
    sectors = compact(row.get("legal_sectors"))
    authority = compact(row.get("issuing_authority"))
    issuance_date = compact(row.get("issuance_date"))

    header_parts = [part for part in [legal_type, document_number, title, chunk.article_no] if part]
    header = " ".join(header_parts)
    metadata = [
        f"Lĩnh vực: {sectors}" if sectors else "",
        f"Cơ quan ban hành: {authority}" if authority else "",
        f"Ngày ban hành: {issuance_date}" if issuance_date else "",
        f"Tiêu đề đoạn: {chunk.heading}" if chunk.heading and chunk.heading != chunk.article_no else "",
    ]
    metadata_text = "\n".join(line for line in metadata if line)
    return f"{header}:\n{metadata_text}\n\n{chunk.text}".strip()


def make_unit(row: dict[str, Any], chunk: Chunk, chunk_index: int, chunk_count: int, tier: str) -> dict[str, Any]:
    document_id = int(row["id"])
    chunk_id = f"vld:{document_id}:chunk:{chunk_index:05d}"
    sectors = split_sectors(row.get("legal_sectors"))
    retrieval_text = build_retrieval_text(row, chunk)
    return {
        "point_id": make_point_id(document_id, chunk_index),
        "dataset": "vietnamese-legal-documents",
        "document_id": document_id,
        "document_number": compact(row.get("document_number")),
        "chunk_id": chunk_id,
        "chunk_index": chunk_index,
        "chunk_count": chunk_count,
        "chunk_method": chunk.method,
        "business_scope_tier": tier,
        "title": compact(row.get("title")),
        "legal_type": compact(row.get("legal_type")),
        "legal_sectors": sectors,
        "issuing_authority": compact(row.get("issuing_authority")),
        "issuance_date": compact(row.get("issuance_date")),
        "signers": compact(row.get("signers")),
        "source_url": compact(row.get("url")),
        "article_title": chunk.heading,
        "article_no_normalized": chunk.article_no,
        "source_law_id_candidates": source_law_ids(row),
        "source_article_no_candidates": [chunk.article_no] if chunk.article_no else [],
        "source_doc_title_candidates": [compact(row.get("title"))] if compact(row.get("title")) else [],
        "content_preview": compact(chunk.text)[:CONTENT_PREVIEW_CHARS],
        "retrieval_text": retrieval_text,
        "retrieval_text_sha1": hashlib.sha1(retrieval_text.encode("utf-8")).hexdigest(),
        # Compatibility fields for the existing search and payload-index code.
        "canonical_article_id": f"vld:{document_id}",
        "content_tree_path": [chunk.article_no] if chunk.article_no else [],
        "chapter_title": "",
        "topic_number": None,
        "topic_title": sectors[0] if sectors else "",
        "subject_title": compact(row.get("legal_type")),
        "source_note_text": compact(row.get("title")),
        "related_note_text": "",
        "citation_confidence": "metadata",
    }


def make_qdrant_preview(unit: dict[str, Any]) -> dict[str, Any]:
    payload = {key: value for key, value in unit.items() if key != "point_id"}
    return {
        "id": unit["point_id"],
        "payload": payload,
        "vectors": {
            "dense": {"model": DEFAULT_DENSE_MODEL, "source": "local", "status": "not_embedded"},
            "sparse": {"model": DEFAULT_SPARSE_MODEL, "source": "qdrant", "field": "retrieval_text"},
        },
    }


def iter_parquet_rows(path: Path, columns: list[str] | None = None) -> Iterator[dict[str, Any]]:
    pf = pq.ParquetFile(path)
    for batch in pf.iter_batches(columns=columns):
        table = batch.to_pydict()
        keys = list(table)
        for values in zip(*(table[key] for key in keys), strict=True):
            yield dict(zip(keys, values, strict=True))


def load_filtered_metadata(metadata_path: Path, min_year: int) -> tuple[dict[int, dict[str, Any]], dict[str, Any]]:
    total = 0
    kept: dict[int, dict[str, Any]] = {}
    tier_counts: Counter[str] = Counter()
    type_counts: Counter[str] = Counter()
    sector_counts: Counter[str] = Counter()

    for row in iter_parquet_rows(metadata_path):
        total += 1
        keep, tier = classify_metadata(row, min_year=min_year)
        if not keep:
            continue
        doc_id = int(row["id"])
        row["_business_scope_tier"] = tier
        kept[doc_id] = row
        tier_counts[tier] += 1
        type_counts[compact(row.get("legal_type")) or "<missing>"] += 1
        for sector in split_sectors(row.get("legal_sectors")):
            sector_counts[sector] += 1

    return kept, {
        "total_documents": total,
        "kept_documents": len(kept),
        "kept_percent": round(len(kept) * 100 / total, 4) if total else 0.0,
        "tier_counts": dict(tier_counts),
        "top_legal_types_kept": dict(type_counts.most_common(25)),
        "top_sectors_kept": dict(sector_counts.most_common(25)),
    }


def write_vld_artifacts(args: argparse.Namespace) -> dict[str, Any]:
    metadata_path = args.vld_root / "metadata" / "data-00000-of-00001.parquet"
    content_dir = args.vld_root / "content"
    args.output_dir.mkdir(parents=True, exist_ok=True)

    metadata_by_id, metadata_report = load_filtered_metadata(metadata_path, min_year=args.min_year)
    wanted_ids = set(metadata_by_id)

    articles_path = args.output_dir / "articles.jsonl"
    units_path = args.output_dir / "retrieval_units.jsonl"
    preview_path = args.output_dir / "qdrant_payload_preview.jsonl"
    metadata_path_out = args.output_dir / "metadata.jsonl"
    ids_path = args.output_dir / "document_ids.txt"

    chunk_count_total = 0
    documents_with_content = 0
    skipped_non_vietnamese = 0
    skipped_empty = 0
    truncated_documents = 0
    method_counts: Counter[str] = Counter()
    tier_unit_counts: Counter[str] = Counter()
    chunk_count_by_doc: list[int] = []

    ids_path.write_text("\n".join(str(doc_id) for doc_id in sorted(wanted_ids)) + "\n", encoding="utf-8")
    with metadata_path_out.open("w", encoding="utf-8", newline="\n") as metadata_out:
        for doc_id in sorted(wanted_ids):
            metadata_out.write(json.dumps(metadata_by_id[doc_id], ensure_ascii=False) + "\n")

    with (
        articles_path.open("w", encoding="utf-8", newline="\n") as articles_out,
        units_path.open("w", encoding="utf-8", newline="\n") as units_out,
        preview_path.open("w", encoding="utf-8", newline="\n") as preview_out,
    ):
        for parquet_path in sorted(content_dir.glob("*.parquet")):
            for content_row in iter_parquet_rows(parquet_path, columns=["id", "content"]):
                document_id = int(content_row["id"])
                row = metadata_by_id.get(document_id)
                if row is None:
                    continue

                content = clean_content(content_row.get("content"))
                if not content:
                    skipped_empty += 1
                    continue
                vi_ratio = vietnamese_marker_ratio(content)
                if args.require_vietnamese_content and vi_ratio < args.min_vietnamese_marker_ratio:
                    skipped_non_vietnamese += 1
                    continue

                chunks = chunk_document(content, max_tokens=args.max_chunk_tokens, overlap_tokens=args.chunk_overlap_tokens)
                if not chunks:
                    skipped_empty += 1
                    continue
                if args.max_chunks_per_document and len(chunks) > args.max_chunks_per_document:
                    chunks = chunks[: args.max_chunks_per_document]
                    truncated_documents += 1

                documents_with_content += 1
                chunk_count_by_doc.append(len(chunks))
                tier = row["_business_scope_tier"]
                article_record = {
                    **{key: value for key, value in row.items() if not key.startswith("_")},
                    "dataset": "vietnamese-legal-documents",
                    "business_scope_tier": tier,
                    "content_char_len": len(content),
                    "content_token_estimate": count_tokens(content),
                    "chunk_count": len(chunks),
                    "vietnamese_marker_ratio": round(vi_ratio, 6),
                }
                articles_out.write(json.dumps(article_record, ensure_ascii=False) + "\n")

                for chunk_index, chunk in enumerate(chunks):
                    unit = make_unit(row, chunk, chunk_index=chunk_index, chunk_count=len(chunks), tier=tier)
                    method_counts[unit["chunk_method"]] += 1
                    tier_unit_counts[tier] += 1
                    chunk_count_total += 1
                    units_out.write(json.dumps(unit, ensure_ascii=False) + "\n")
                    preview_out.write(json.dumps(make_qdrant_preview(unit), ensure_ascii=False) + "\n")

    report = {
        "dataset": "vietnamese-legal-documents",
        "output_dir": str(args.output_dir),
        "filter": {
            "min_year": args.min_year,
            "core_keywords": CORE_KEYWORDS,
            "support_keywords": SUPPORT_KEYWORDS,
            "core_law_ids": sorted(CORE_LAW_IDS),
            "support_law_ids": sorted(SUPPORT_LAW_IDS),
            "important_legal_types": sorted(IMPORTANT_LEGAL_TYPES),
            "support_legal_types": sorted(SUPPORT_LEGAL_TYPES),
            "require_vietnamese_content": args.require_vietnamese_content,
            "min_vietnamese_marker_ratio": args.min_vietnamese_marker_ratio,
        },
        "metadata": metadata_report,
        "content": {
            "documents_with_content": documents_with_content,
            "skipped_empty": skipped_empty,
            "skipped_non_vietnamese": skipped_non_vietnamese,
            "truncated_documents": truncated_documents,
        },
        "chunking": {
            "max_chunk_tokens": args.max_chunk_tokens,
            "chunk_overlap_tokens": args.chunk_overlap_tokens,
            "max_chunks_per_document": args.max_chunks_per_document,
            "retrieval_units": chunk_count_total,
            "method_distribution": dict(method_counts.most_common()),
            "tier_distribution": dict(tier_unit_counts),
            "avg_chunks_per_document": round(chunk_count_total / documents_with_content, 2) if documents_with_content else 0,
            "max_chunk_count_per_document": max(chunk_count_by_doc, default=0),
        },
        "outputs": {
            "articles": str(articles_path),
            "retrieval_units": str(units_path),
            "qdrant_payload_preview": str(preview_path),
            "metadata": str(metadata_path_out),
            "document_ids": str(ids_path),
        },
    }
    report_path = args.output_dir / "build_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vld-root", type=Path, default=Path("data/vietnamese-legal-documents"))
    parser.add_argument("--output-dir", type=Path, default=Path("build/vld_business_scope"))
    parser.add_argument("--min-year", type=int, default=2010)
    parser.add_argument("--max-chunk-tokens", type=int, default=DEFAULT_MAX_CHUNK_TOKENS)
    parser.add_argument("--chunk-overlap-tokens", type=int, default=DEFAULT_CHUNK_OVERLAP_TOKENS)
    parser.add_argument("--max-chunks-per-document", type=int, default=0)
    parser.add_argument("--require-vietnamese-content", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--min-vietnamese-marker-ratio", type=float, default=0.01)
    return parser


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except AttributeError:
        pass
    args = build_parser().parse_args()
    report = write_vld_artifacts(args)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
