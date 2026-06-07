"""Build preview trees from vietnamese-legal-documents markdown content.

The output is for human review only: it preserves legal hierarchy as
``children`` and keeps text/table/heading material as ordered ``blocks`` inside
the current legal node.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import uuid
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator

try:
    import pyarrow.parquet as pq
except ImportError as exc:  # pragma: no cover - local environment guard.
    raise RuntimeError("pyarrow is required. Run: uv sync --extra data") from exc


DEFAULT_SAMPLE_IDS = [
    427301,  # Luật Doanh nghiệp 2020
    283247,  # Nghị định 01/2021/NĐ-CP về đăng ký doanh nghiệp
    677282,  # QĐ with long administrative-procedure table
    668308,  # QĐ with PHỤ LỤC and tables
    484451,  # QĐ with multiple appendices/tables
    694605,  # QĐ with DANH MỤC table
]

NODE_PARENT_TYPES: dict[str, tuple[str, ...]] = {
    "part": ("document",),
    "chapter": ("part", "document"),
    "section": ("chapter", "part", "document", "appendix"),
    "subsection": ("section", "chapter", "part", "document", "appendix"),
    "article": ("subsection", "section", "chapter", "part", "document"),
    "clause": ("article",),
    "point": ("clause", "article"),
    "subpoint": ("point", "clause"),
    "appendix": ("document",),
}

PART_RE = re.compile(r"^(?:PHẦN|Phần)\s+(?P<num>[IVXLCDM\d]+)(?:[.:]?\s*(?P<title>.*))?$")
CHAPTER_RE = re.compile(r"^(?:CHƯƠNG|Chương)\s+(?P<num>[IVXLCDM\d]+)(?:[.:]?\s*(?P<title>.*))?$")
SECTION_RE = re.compile(r"^(?:MỤC|Mục)\s+(?P<num>\d+[A-Za-z]?)\.?\s*(?P<title>.*)$")
SUBSECTION_RE = re.compile(r"^(?:TIỂU MỤC|Tiểu mục)\s+(?P<num>\d+[A-Za-z]?)\.?\s*(?P<title>.*)$")
ARTICLE_RE = re.compile(r"^(?:ĐIỀU|Điều)\s+(?P<num>\d+[A-Za-z]?)\.?\s*(?P<title>.*)$")
CLAUSE_RE = re.compile(r"^(?P<num>\d+)\.\s+(?P<body>.+)$")
POINT_RE = re.compile(r"^(?P<num>[a-zđ])\)\s+(?P<body>.+)$", re.IGNORECASE)
BULLET_RE = re.compile(r"^(?:[-–•])\s+(?P<body>.+)$")
APPENDIX_RE = re.compile(r"^(?:PHỤ LỤC|Phụ lục)(?:\s+(?P<num>[IVXLCDM\d]+))?(?:[.:]?\s*(?P<title>.*))?$")
APPENDIX_HEADING_RE = re.compile(r"^(?:DANH MỤC|Danh mục|MẪU SỐ|Mẫu số)\b(?P<title>.*)$")
FOOTER_RE = re.compile(r"^(?:Nơi nhận|TM\.|KT\.|CHỦ TỊCH|BỘ TRƯỞNG|THỦ TƯỚNG|TỔNG THƯ KÝ)\b", re.IGNORECASE)
TABLE_LINE_RE = re.compile(r"^\s*\|.*\|\s*$")
UPPER_HINT_RE = re.compile(r"[A-ZÀÁẢÃẠĂẰẮẲẴẶÂẦẤẨẪẬÈÉẺẼẸÊỀẾỂỄỆÌÍỈĨỊÒÓỎÕỌÔỒỐỔỖỘƠỜỚỞỠỢÙÚỦŨỤƯỪỨỬỮỰỲÝỶỸỴĐ]{3,}")
WHITESPACE_RE = re.compile(r"[ \t\r\f\v]+")


@dataclass
class TreeNode:
    type: str
    label: str
    number: str = ""
    title: str = ""
    line_start: int = 0
    line_end: int = 0
    blocks: list[dict[str, Any]] = field(default_factory=list)
    children: list["TreeNode"] = field(default_factory=list)


def normalize(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).replace("\u00a0", " ")
    text = WHITESPACE_RE.sub(" ", text)
    return text.strip()


def is_table_line(line: str) -> bool:
    stripped = line.strip()
    return bool(TABLE_LINE_RE.match(stripped)) and stripped.count("|") >= 2


def is_title_like(line: str) -> bool:
    stripped = normalize(line)
    if not stripped or len(stripped) > 260:
        return False
    if stripped.endswith((".", ";", ",")):
        return False
    letters = [char for char in stripped if char.isalpha()]
    if not letters:
        return False
    uppercase_letters = [char for char in letters if char.upper() == char]
    return len(uppercase_letters) / max(1, len(letters)) > 0.72 or bool(UPPER_HINT_RE.search(stripped))


def make_node(node_type: str, label: str, line_no: int, number: str = "", title: str = "") -> TreeNode:
    return TreeNode(
        type=node_type,
        label=normalize(label),
        number=normalize(number),
        title=normalize(title),
        line_start=line_no,
        line_end=line_no,
    )


def node_display(node: TreeNode) -> str:
    parts = [node.label]
    if node.title and node.title not in node.label:
        parts.append(node.title)
    return ". ".join(part for part in parts if part)


def can_parent(parent_type: str, child_type: str) -> bool:
    return parent_type in NODE_PARENT_TYPES.get(child_type, ())


def attach_node(root: TreeNode, active_path: list[TreeNode], node: TreeNode) -> list[TreeNode]:
    """Attach by type-specific parent rules and return the new active path."""
    for index in range(len(active_path) - 1, -1, -1):
        candidate = active_path[index]
        if can_parent(candidate.type, node.type):
            candidate.children.append(node)
            return active_path[: index + 1] + [node]

    root.children.append(node)
    return [root, node]


def current_node(active_path: list[TreeNode]) -> TreeNode:
    return active_path[-1]


def append_block(node: TreeNode, block: dict[str, Any]) -> None:
    if block["type"] == "text" and node.blocks and node.blocks[-1]["type"] == "text":
        previous = node.blocks[-1]
        previous["text"] = f"{previous['text']}\n{block['text']}".strip()
        previous["line_end"] = block["line_end"]
        return
    node.blocks.append(block)


def flush_text_buffer(active_path: list[TreeNode], buffer: list[tuple[int, str]]) -> None:
    if not buffer:
        return
    text = "\n".join(line for _, line in buffer).strip()
    if text:
        append_block(
            current_node(active_path),
            {
                "type": "text",
                "text": text,
                "line_start": buffer[0][0],
                "line_end": buffer[-1][0],
            },
        )
    buffer.clear()


def parse_structural_line(line: str, line_no: int, active_path: list[TreeNode]) -> tuple[TreeNode | None, str]:
    stripped = normalize(line)
    for node_type, pattern, label_prefix in [
        ("part", PART_RE, "Phần"),
        ("chapter", CHAPTER_RE, "Chương"),
        ("section", SECTION_RE, "Mục"),
        ("subsection", SUBSECTION_RE, "Tiểu mục"),
        ("article", ARTICLE_RE, "Điều"),
    ]:
        match = pattern.match(stripped)
        if match:
            number = normalize(match.group("num"))
            title = normalize(match.groupdict().get("title"))
            return make_node(node_type, f"{label_prefix} {number}", line_no, number=number, title=title), ""

    match = APPENDIX_RE.match(stripped)
    if match:
        number = normalize(match.group("num"))
        title = normalize(match.groupdict().get("title"))
        label = f"Phụ lục {number}".strip()
        return make_node("appendix", label, line_no, number=number, title=title), ""

    match = APPENDIX_HEADING_RE.match(stripped)
    if match and not has_ancestor(active_path, "article"):
        title = normalize(match.groupdict().get("title"))
        label = "Danh mục" if stripped.lower().startswith("danh mục") else "Mẫu số"
        return make_node("appendix", label, line_no, title=title), ""

    match = CLAUSE_RE.match(stripped)
    if match and has_ancestor(active_path, "article"):
        number = normalize(match.group("num"))
        body = normalize(match.group("body"))
        return make_node("clause", f"Khoản {number}", line_no, number=number), body

    match = POINT_RE.match(stripped)
    if match and (has_ancestor(active_path, "clause") or has_ancestor(active_path, "article")):
        number = normalize(match.group("num")).lower()
        body = normalize(match.group("body"))
        return make_node("point", f"Điểm {number}", line_no, number=number), body

    match = BULLET_RE.match(stripped)
    if match and (has_ancestor(active_path, "point") or has_ancestor(active_path, "clause")):
        return make_node("subpoint", "Ý", line_no), normalize(match.group("body"))

    return None, ""


def has_ancestor(active_path: list[TreeNode], node_type: str) -> bool:
    return any(node.type == node_type for node in active_path)


def maybe_absorb_title(active_path: list[TreeNode], line: str, line_no: int) -> bool:
    node = current_node(active_path)
    if node.type not in {"part", "chapter", "appendix"} or node.title:
        return False
    if not is_title_like(line):
        return False
    node.title = normalize(line)
    node.line_end = line_no
    return True


def parse_table(lines: list[tuple[int, str]]) -> dict[str, Any]:
    parsed_rows = [split_table_row(line) for _, line in lines]
    useful_rows = [row for row in parsed_rows if any(cell for cell in row) and not is_separator_row(row)]
    headers = useful_rows[0] if useful_rows else []
    rows = useful_rows[1:] if len(useful_rows) > 1 else []
    return {
        "type": "table",
        "caption": "",
        "headers": headers,
        "rows": rows,
        "row_count": len(rows),
        "column_count": max((len(row) for row in useful_rows), default=0),
        "raw_markdown": "\n".join(line for _, line in lines),
        "line_start": lines[0][0],
        "line_end": lines[-1][0],
    }


def split_table_row(line: str) -> list[str]:
    stripped = line.strip()
    if stripped.startswith("|"):
        stripped = stripped[1:]
    if stripped.endswith("|"):
        stripped = stripped[:-1]
    return [normalize(cell) for cell in stripped.split("|")]


def is_separator_row(row: list[str]) -> bool:
    if not row:
        return True
    return all(not cell or set(cell) <= {"-", ":"} for cell in row)


def append_table(active_path: list[TreeNode], table_lines: list[tuple[int, str]]) -> None:
    block = parse_table(table_lines)
    node = current_node(active_path)
    for previous in reversed(node.blocks):
        if previous["type"] in {"heading", "text"}:
            previous_text = normalize(previous.get("text"))
            if is_title_like(previous_text):
                block["caption"] = previous_text
            break
    append_block(node, block)


def build_tree(row: dict[str, Any], content: str) -> dict[str, Any]:
    root = TreeNode(
        type="document",
        label=normalize(row.get("title")),
        number=normalize(row.get("document_number")),
        title=normalize(row.get("title")),
        line_start=1,
        line_end=max(1, len(content.splitlines())),
    )
    active_path = [root]
    text_buffer: list[tuple[int, str]] = []
    table_buffer: list[tuple[int, str]] = []

    for line_no, raw_line in enumerate(content.splitlines(), start=1):
        line = normalize(raw_line)
        if not line:
            flush_text_buffer(active_path, text_buffer)
            continue

        if is_table_line(line):
            flush_text_buffer(active_path, text_buffer)
            table_buffer.append((line_no, line))
            continue
        if table_buffer:
            append_table(active_path, table_buffer)
            table_buffer = []

        structural_node, initial_body = parse_structural_line(line, line_no, active_path)
        if structural_node:
            flush_text_buffer(active_path, text_buffer)
            active_path = attach_node(root, active_path, structural_node)
            if initial_body:
                append_block(
                    structural_node,
                    {"type": "text", "text": initial_body, "line_start": line_no, "line_end": line_no},
                )
            continue

        if maybe_absorb_title(active_path, line, line_no):
            continue

        if APPENDIX_HEADING_RE.match(line):
            flush_text_buffer(active_path, text_buffer)
            append_block(
                current_node(active_path),
                {"type": "heading", "text": line, "line_start": line_no, "line_end": line_no},
            )
            continue

        if FOOTER_RE.match(line):
            flush_text_buffer(active_path, text_buffer)
            footer = make_node("footer", "Footer", line_no)
            active_path = attach_footer(root, active_path, footer)
            append_block(footer, {"type": "text", "text": line, "line_start": line_no, "line_end": line_no})
            continue

        if is_title_like(line) and current_node(active_path).type in {"appendix", "document"}:
            flush_text_buffer(active_path, text_buffer)
            append_block(
                current_node(active_path),
                {"type": "heading", "text": line, "line_start": line_no, "line_end": line_no},
            )
            continue

        text_buffer.append((line_no, line))

    flush_text_buffer(active_path, text_buffer)
    if table_buffer:
        append_table(active_path, table_buffer)

    return {
        "metadata": {
            "document_id": int(row["id"]),
            "document_number": normalize(row.get("document_number")),
            "title": normalize(row.get("title")),
            "legal_type": normalize(row.get("legal_type")),
            "legal_sectors": normalize(row.get("legal_sectors")),
            "issuing_authority": normalize(row.get("issuing_authority")),
            "issuance_date": normalize(row.get("issuance_date")),
            "url": normalize(row.get("url")),
        },
        "tree": materialize_node(root, []),
    }


def attach_footer(root: TreeNode, active_path: list[TreeNode], footer: TreeNode) -> list[TreeNode]:
    root.children.append(footer)
    return [root, footer]


def materialize_node(node: TreeNode, path: list[str]) -> dict[str, Any]:
    display = node_display(node)
    next_path = path + ([display] if display else [])
    node_id_basis = "|".join([node.type, node.number, display, str(node.line_start), str(node.line_end)])
    return {
        "node_id": str(uuid.uuid5(uuid.NAMESPACE_URL, node_id_basis)),
        "type": node.type,
        "label": node.label,
        "number": node.number,
        "title": node.title,
        "display": display,
        "path": next_path,
        "line_start": node.line_start,
        "line_end": node.line_end,
        "blocks": node.blocks,
        "children": [materialize_node(child, next_path) for child in node.children],
    }


def iter_parquet_rows(path: Path, columns: list[str] | None = None) -> Iterator[dict[str, Any]]:
    pf = pq.ParquetFile(path)
    for batch in pf.iter_batches(columns=columns):
        table = batch.to_pydict()
        keys = list(table)
        for values in zip(*(table[key] for key in keys), strict=True):
            yield dict(zip(keys, values, strict=True))


def load_metadata(vld_root: Path, ids: set[int]) -> dict[int, dict[str, Any]]:
    metadata_path = vld_root / "metadata" / "data-00000-of-00001.parquet"
    rows: dict[int, dict[str, Any]] = {}
    for row in iter_parquet_rows(metadata_path):
        doc_id = int(row["id"])
        if doc_id in ids:
            rows[doc_id] = row
            if len(rows) == len(ids):
                break
    return rows


def load_contents(vld_root: Path, ids: set[int]) -> dict[int, str]:
    content_dir = vld_root / "content"
    contents: dict[int, str] = {}
    for parquet_path in sorted(content_dir.glob("*.parquet")):
        for row in iter_parquet_rows(parquet_path, columns=["id", "content"]):
            doc_id = int(row["id"])
            if doc_id in ids:
                contents[doc_id] = str(row.get("content") or "")
                if len(contents) == len(ids):
                    return contents
    return contents


def tree_stats(tree: dict[str, Any]) -> dict[str, Any]:
    node_counts: Counter[str] = Counter()
    block_counts: Counter[str] = Counter()
    table_count = 0

    def visit(node: dict[str, Any]) -> None:
        nonlocal table_count
        node_counts[node["type"]] += 1
        for block in node.get("blocks", []):
            block_counts[block["type"]] += 1
            if block["type"] == "table":
                table_count += 1
        for child in node.get("children", []):
            visit(child)

    visit(tree["tree"])
    return {
        "document_id": tree["metadata"]["document_id"],
        "document_number": tree["metadata"]["document_number"],
        "title": tree["metadata"]["title"],
        "node_counts": dict(node_counts),
        "block_counts": dict(block_counts),
        "table_count": table_count,
    }


def parse_ids(values: Iterable[str]) -> set[int]:
    ids: set[int] = set()
    for value in values:
        for part in re.split(r"[,\s]+", value.strip()):
            if part:
                ids.add(int(part))
    return ids


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vld-root", type=Path, default=Path("data/vietnamese-legal-documents"))
    parser.add_argument("--output-dir", type=Path, default=Path("build/vld_tree_preview"))
    parser.add_argument("--document-id", action="append", default=[], help="Document id(s), repeat or comma-separate.")
    parser.add_argument("--ids-file", type=Path, help="Optional UTF-8 file containing document ids.")
    return parser


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except AttributeError:
        pass
    args = build_parser().parse_args()

    ids = parse_ids(args.document_id)
    if args.ids_file:
        ids.update(parse_ids(args.ids_file.read_text(encoding="utf-8").splitlines()))
    if not ids:
        ids = set(DEFAULT_SAMPLE_IDS)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    metadata = load_metadata(args.vld_root, ids)
    contents = load_contents(args.vld_root, ids)

    trees = []
    missing = sorted(ids - set(metadata) | ids - set(contents))
    for doc_id in sorted(ids):
        if doc_id not in metadata or doc_id not in contents:
            continue
        trees.append(build_tree(metadata[doc_id], contents[doc_id]))

    trees_json = args.output_dir / "trees.json"
    trees_jsonl = args.output_dir / "trees.jsonl"
    summary_json = args.output_dir / "summary.json"
    trees_json.write_text(json.dumps(trees, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with trees_jsonl.open("w", encoding="utf-8", newline="\n") as handle:
        for tree in trees:
            handle.write(json.dumps(tree, ensure_ascii=False) + "\n")
    summary = {
        "requested_ids": sorted(ids),
        "built_count": len(trees),
        "missing_ids": missing,
        "outputs": {
            "trees_json": str(trees_json),
            "trees_jsonl": str(trees_jsonl),
            "summary_json": str(summary_json),
        },
        "documents": [tree_stats(tree) for tree in trees],
    }
    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
