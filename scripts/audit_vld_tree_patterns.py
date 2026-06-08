"""Audit VLD tree parsing across a broader document sample.

The goal is to find suspicious structures before we trust tree-based chunking:
missed articles, missed appendices, tables under footer, unknown headings, and
large unstructured text blocks.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterator

try:
    import pyarrow.parquet as pq
except ImportError as exc:  # pragma: no cover
    raise RuntimeError("pyarrow is required. Run: uv sync --extra data") from exc

from build_vld_business_collection import classify_metadata
from build_vld_tree_preview import build_tree


ARTICLE_RAW_RE = re.compile(r"^\s*(?:Điều|ĐIỀU)\s+\d+[A-Za-z]?\b", re.MULTILINE)
CHAPTER_RAW_RE = re.compile(r"^\s*(?:Chương|CHƯƠNG)\s+[IVXLCDM\d]+\b", re.MULTILINE)
SECTION_RAW_RE = re.compile(r"^\s*(?:Mục|MỤC)\s+\d+[A-Za-z]?\b", re.MULTILINE)
APPENDIX_RAW_RE = re.compile(r"^\s*(?:PHỤ LỤC|Phụ lục)\b", re.MULTILINE)
APPENDIX_HEADING_RAW_RE = re.compile(r"^\s*(?:DANH MỤC|Danh mục|MẪU SỐ|Mẫu số)\b", re.MULTILINE)
TABLE_LINE_RE = re.compile(r"^\s*\|.*\|\s*$")
ATTACHMENT_TABLE_RE = re.compile(r"FILE ĐƯỢC ĐÍNH KÈM|Tải về|Download|Chữ ký số", re.IGNORECASE)
FOOTER_RAW_RE = re.compile(r"^\s*(?:Nơi nhận|TM\.|KT\.|CHỦ TỊCH|BỘ TRƯỞNG|THỦ TƯỚNG|TỔNG THƯ KÝ)\b", re.IGNORECASE)

UNKNOWN_HEADING_PATTERNS = {
    "letter_heading": re.compile(r"^\s*[A-Z]\.\s+[A-ZÀÁẢÃẠĂẰẮẲẴẶÂẦẤẨẪẬÈÉẺẼẸÊỀẾỂỄỆÌÍỈĨỊÒÓỎÕỌÔỒỐỔỖỘƠỜỚỞỠỢÙÚỦŨỤƯỪỨỬỮỰỲÝỶỸỴĐ].+"),
    "roman_heading": re.compile(r"^\s*[IVXLCDM]+\.\s+[A-ZÀÁẢÃẠĂẰẮẲẴẶÂẦẤẨẪẬÈÉẺẼẸÊỀẾỂỄỆÌÍỈĨỊÒÓỎÕỌÔỒỐỔỖỘƠỜỚỞỠỢÙÚỦŨỤƯỪỨỬỮỰỲÝỶỸỴĐ].+"),
    "decimal_heading": re.compile(r"^\s*\d+(?:\.\d+)+\.?\s+.+"),
    "article_word_ordinal": re.compile(r"^\s*(?:Điều|ĐIỀU)\s+thứ\s+.+", re.IGNORECASE),
    "chapter_word_ordinal": re.compile(r"^\s*(?:Chương|CHƯƠNG)\s+thứ\s+.+", re.IGNORECASE),
    "form_heading": re.compile(r"^\s*(?:Mẫu|MẪU)\s+(?:số\s*)?[\w./-]+", re.IGNORECASE),
    "attachment_notice": re.compile(r"FILE ĐƯỢC ĐÍNH KÈM|Tải về|Download", re.IGNORECASE),
}


def iter_parquet_rows(path: Path, columns: list[str] | None = None) -> Iterator[dict[str, Any]]:
    pf = pq.ParquetFile(path)
    for batch in pf.iter_batches(columns=columns):
        table = batch.to_pydict()
        keys = list(table)
        for values in zip(*(table[key] for key in keys), strict=True):
            yield dict(zip(keys, values, strict=True))


def select_metadata(vld_root: Path, limit: int, min_year: int) -> dict[int, dict[str, Any]]:
    metadata_path = vld_root / "metadata" / "data-00000-of-00001.parquet"
    selected: dict[int, dict[str, Any]] = {}
    buckets: dict[str, int] = defaultdict(int)
    per_bucket = max(10, limit // 8)
    keyword_patterns = {
        "business": lambda row: classify_metadata(row, min_year=min_year)[0],
        "appendix": lambda row: has_text(row, ["phụ lục", "danh mục", "mẫu", "biểu mẫu"]),
        "procedure": lambda row: has_text(row, ["thủ tục hành chính", "đăng ký doanh nghiệp", "hộ kinh doanh"]),
        "tax_accounting": lambda row: has_text(row, ["thuế", "kế toán", "hóa đơn", "báo cáo tài chính"]),
        "law_decree": lambda row: str(row.get("legal_type") or "").strip() in {"Luật", "Nghị định", "Thông tư"},
        "decision": lambda row: str(row.get("legal_type") or "").strip() == "Quyết định",
    }

    for row in iter_parquet_rows(metadata_path):
        if len(selected) >= limit:
            break
        doc_id = int(row["id"])
        for bucket, predicate in keyword_patterns.items():
            if buckets[bucket] >= per_bucket:
                continue
            if predicate(row):
                selected[doc_id] = row
                buckets[bucket] += 1
                break
    return selected


def has_text(row: dict[str, Any], keywords: list[str]) -> bool:
    text = f"{row.get('title') or ''} {row.get('legal_sectors') or ''}".lower()
    return any(keyword in text for keyword in keywords)


def load_contents(vld_root: Path, ids: set[int]) -> dict[int, str]:
    contents: dict[int, str] = {}
    for parquet_path in sorted((vld_root / "content").glob("*.parquet")):
        for row in iter_parquet_rows(parquet_path, columns=["id", "content"]):
            doc_id = int(row["id"])
            if doc_id in ids:
                contents[doc_id] = str(row.get("content") or "")
                if len(contents) == len(ids):
                    return contents
    return contents


def count_nodes(node: dict[str, Any], node_counts: Counter[str], block_counts: Counter[str], issues: list[dict[str, Any]]) -> None:
    node_counts[node["type"]] += 1
    for block in node.get("blocks", []):
        block_counts[block["type"]] += 1
        if block["type"] == "table" and node["type"] == "footer":
            issues.append(
                {
                    "issue": "table_under_footer",
                    "path": node.get("path", []),
                    "line_start": block.get("line_start"),
                    "line_end": block.get("line_end"),
                    "sample": str(block.get("raw_markdown") or "")[:500],
                }
            )
        if block["type"] == "text" and node["type"] in {"document", "footer"} and len(str(block.get("text") or "")) > 1000:
            issues.append(
                {
                    "issue": "large_text_on_context_only_node",
                    "node_type": node["type"],
                    "path": node.get("path", []),
                    "line_start": block.get("line_start"),
                    "line_end": block.get("line_end"),
                    "sample": str(block.get("text") or "")[:500],
                }
            )
    for child in node.get("children", []):
        count_nodes(child, node_counts, block_counts, issues)


def raw_table_groups(content: str) -> tuple[int, int]:
    count = 0
    attachment_count = 0
    in_table = False
    table_lines: list[str] = []

    def finish_group() -> None:
        nonlocal count, attachment_count, table_lines
        if not table_lines:
            return
        raw = "\n".join(table_lines)
        if ATTACHMENT_TABLE_RE.search(raw):
            attachment_count += 1
        else:
            count += 1
        table_lines = []

    for line in content.splitlines():
        is_table = bool(TABLE_LINE_RE.match(line.strip())) and line.count("|") >= 2
        if is_table and not in_table:
            table_lines = []
        if is_table:
            table_lines.append(line.strip())
        if not is_table and in_table:
            finish_group()
        in_table = is_table
    finish_group()
    return count, attachment_count


def find_unknown_lines(content: str, limit: int = 20) -> list[dict[str, Any]]:
    rows = []
    known = [
        ARTICLE_RAW_RE,
        CHAPTER_RAW_RE,
        SECTION_RAW_RE,
        APPENDIX_RAW_RE,
        APPENDIX_HEADING_RAW_RE,
    ]
    for line_no, line in enumerate(content.splitlines(), start=1):
        stripped = line.strip()
        if not stripped or any(pattern.match(stripped) for pattern in known):
            continue
        for name, pattern in UNKNOWN_HEADING_PATTERNS.items():
            if pattern.search(stripped):
                rows.append({"pattern": name, "line": line_no, "text": stripped[:300]})
                break
        if len(rows) >= limit:
            return rows
    return rows


def audit_document(row: dict[str, Any], content: str) -> dict[str, Any]:
    tree = build_tree(row, content)
    node_counts: Counter[str] = Counter()
    block_counts: Counter[str] = Counter()
    issues: list[dict[str, Any]] = []
    count_nodes(tree["tree"], node_counts, block_counts, issues)

    raw_table_count, raw_attachment_count = raw_table_groups(content)
    raw_counts = {
        "chapter": len(CHAPTER_RAW_RE.findall(content)),
        "section": len(SECTION_RAW_RE.findall(content)),
        "article": len(ARTICLE_RAW_RE.findall(content)),
        "appendix": len(APPENDIX_RAW_RE.findall(content)),
        "appendix_heading": len(APPENDIX_HEADING_RAW_RE.findall(content)),
        "table_group": raw_table_count,
        "attachment_group": raw_attachment_count,
        "footer_marker": sum(1 for line in content.splitlines() if FOOTER_RAW_RE.match(line.strip())),
    }

    if raw_counts["article"] != node_counts.get("article", 0):
        issues.append(
            {
                "issue": "article_count_mismatch",
                "raw": raw_counts["article"],
                "tree": node_counts.get("article", 0),
            }
        )
    if raw_counts["table_group"] != block_counts.get("table", 0):
        issues.append(
            {
                "issue": "table_count_mismatch",
                "raw": raw_counts["table_group"],
                "tree": block_counts.get("table", 0),
            }
        )
    if raw_counts["appendix"] + raw_counts["appendix_heading"] > 0 and node_counts.get("appendix", 0) == 0:
        issues.append(
            {
                "issue": "appendix_markers_without_appendix_node",
                "raw_appendix": raw_counts["appendix"],
                "raw_appendix_heading": raw_counts["appendix_heading"],
            }
        )

    unknown_lines = find_unknown_lines(content)
    if unknown_lines:
        issues.append({"issue": "unknown_heading_patterns", "examples": unknown_lines})

    return {
        "document_id": int(row["id"]),
        "document_number": str(row.get("document_number") or ""),
        "title": str(row.get("title") or ""),
        "legal_type": str(row.get("legal_type") or ""),
        "legal_sectors": str(row.get("legal_sectors") or ""),
        "raw_counts": raw_counts,
        "tree_node_counts": dict(node_counts),
        "tree_block_counts": dict(block_counts),
        "issue_count": len(issues),
        "issues": issues,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vld-root", type=Path, default=Path("data/vietnamese-legal-documents"))
    parser.add_argument("--output-dir", type=Path, default=Path("build/vld_tree_audit"))
    parser.add_argument("--limit", type=int, default=160)
    parser.add_argument("--min-year", type=int, default=2010)
    return parser


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except AttributeError:
        pass
    args = build_parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    metadata = select_metadata(args.vld_root, args.limit, args.min_year)
    contents = load_contents(args.vld_root, set(metadata))
    reports = []
    for doc_id, row in metadata.items():
        content = contents.get(doc_id)
        if content is None:
            continue
        reports.append(audit_document(row, content))

    aggregate = {
        "sampled_documents": len(reports),
        "documents_with_issues": sum(1 for report in reports if report["issue_count"]),
        "issue_distribution": dict(
            Counter(issue["issue"] for report in reports for issue in report["issues"]).most_common()
        ),
        "node_distribution": dict(
            Counter(
                node_type
                for report in reports
                for node_type, count in report["tree_node_counts"].items()
                for _ in range(count)
            ).most_common()
        ),
        "block_distribution": dict(
            Counter(
                block_type
                for report in reports
                for block_type, count in report["tree_block_counts"].items()
                for _ in range(count)
            ).most_common()
        ),
    }

    suspicious = [report for report in reports if report["issue_count"]]
    (args.output_dir / "audit_summary.json").write_text(
        json.dumps(aggregate, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "audit_reports.json").write_text(
        json.dumps(reports, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "suspicious_documents.json").write_text(
        json.dumps(suspicious[:80], ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(aggregate, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
