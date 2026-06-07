"""Build preview chunks from VLD legal trees.

This is a review artifact, not an ingestion job. It builds legal trees with
``build_vld_tree_preview.py`` and emits chunks whose retrieval text uses natural
context from ancestors while preserving original surface markers like ``2.``
and ``a)``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import uuid
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from build_vld_tree_preview import DEFAULT_SAMPLE_IDS, build_tree, load_contents, load_metadata


CHUNK_NAMESPACE = uuid.UUID("c1f3536b-2a21-48be-a5f3-e7e266dfbc91")
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?。])\s+")
SPACE_RE = re.compile(r"\s+")
_TIKTOKEN_ENCODING: Any | None = None


def normalize(value: Any) -> str:
    return SPACE_RE.sub(" ", "" if value is None else str(value)).strip()


def join_natural(parts: Iterable[str]) -> str:
    text = ""
    for raw_part in parts:
        part = normalize(raw_part)
        if not part:
            continue
        if not text:
            text = part
            continue
        if text.endswith((".", ":", ";", "?", "!", "\n")):
            text = f"{text} {part}"
        else:
            text = f"{text}. {part}"
    return text


def count_tokens(text: str) -> int:
    text = normalize(text)
    if not text:
        return 0
    return len(tiktoken_encoding().encode(text))


def tiktoken_encoding() -> Any:
    global _TIKTOKEN_ENCODING
    if _TIKTOKEN_ENCODING is None:
        try:
            import tiktoken
        except ImportError as exc:
            raise RuntimeError("tiktoken is required for VLD chunk token counting. Run: uv sync --extra data") from exc
        _TIKTOKEN_ENCODING = tiktoken.get_encoding("cl100k_base")
    return _TIKTOKEN_ENCODING


def parse_ids(values: Iterable[str]) -> set[int]:
    ids: set[int] = set()
    for value in values:
        for part in re.split(r"[,\s]+", value.strip()):
            if part:
                ids.add(int(part))
    return ids


def marker_text(node: dict[str, Any], include_title: bool = True) -> str:
    node_type = node.get("type")
    number = normalize(node.get("number"))
    label = normalize(node.get("label"))
    title = normalize(node.get("title"))
    if node_type == "clause" and number:
        return f"{number}."
    if node_type == "point" and number:
        return f"{number})"
    if node_type == "subpoint":
        return label
    if include_title:
        return normalize(node.get("display")) or label
    return label


def node_text_blocks(node: dict[str, Any]) -> list[dict[str, Any]]:
    return [block for block in node.get("blocks", []) if block.get("type") == "text" and normalize(block.get("text"))]


def first_text_block(node: dict[str, Any]) -> str:
    blocks = node_text_blocks(node)
    return normalize(blocks[0].get("text")) if blocks else ""


def surface_node_text(node: dict[str, Any], text: str) -> str:
    text = normalize(text)
    node_type = node.get("type")
    number = normalize(node.get("number"))
    if node_type == "clause" and number:
        return f"{number}. {text}".strip()
    if node_type == "point" and number:
        return f"{number}) {text}".strip()
    if node_type == "subpoint":
        label = normalize(node.get("label"))
        return f"{label} {text}".strip() if label else text
    return text


def ancestor_context(document: dict[str, Any], ancestors: list[dict[str, Any]], current: dict[str, Any]) -> str:
    metadata = document["metadata"]
    pieces = [
        normalize(metadata.get("document_number")),
        normalize(metadata.get("title")),
    ]
    context_parts = [" ".join(part for part in pieces if part)]

    for node in ancestors[1:]:  # skip document root
        node_type = node.get("type")
        if node_type in {"part", "chapter", "section", "subsection", "article", "appendix"}:
            display = normalize(node.get("display"))
            if display:
                context_parts.append(display)
            continue
        if node_type in {"clause", "point"}:
            lead = first_text_block(node)
            if lead:
                context_parts.append(surface_node_text(node, lead))
            else:
                marker = marker_text(node)
                if marker:
                    context_parts.append(marker)

    if current.get("type") in {"part", "chapter", "section", "subsection", "article", "appendix"}:
        display = normalize(current.get("display"))
        if display and display not in context_parts:
            context_parts.append(display)

    return join_natural(context_parts)


def split_long_text(text: str, max_tokens: int) -> list[str]:
    text = normalize(text)
    if not text or count_tokens(text) <= max_tokens:
        return [text] if text else []

    sentences = [part.strip() for part in SENTENCE_SPLIT_RE.split(text) if part.strip()]
    if len(sentences) <= 1:
        words = text.split()
        chunks: list[str] = []
        current: list[str] = []
        current_tokens = 0
        for word in words:
            word_tokens = count_tokens(word)
            if current and current_tokens + word_tokens > max_tokens:
                chunks.append(" ".join(current))
                current = [word]
                current_tokens = word_tokens
                continue
            current.append(word)
            current_tokens += word_tokens
        if current:
            chunks.append(" ".join(current))
        return chunks

    chunks: list[str] = []
    current: list[str] = []
    current_tokens = 0
    for sentence in sentences:
        sentence_tokens = count_tokens(sentence)
        if current and current_tokens + sentence_tokens > max_tokens:
            chunks.append(" ".join(current).strip())
            current = [sentence]
            current_tokens = sentence_tokens
            continue
        current.append(sentence)
        current_tokens += sentence_tokens
    if current:
        chunks.append(" ".join(current).strip())
    return chunks


def table_markdown(headers: list[str], rows: list[list[str]]) -> str:
    if not headers:
        return "\n".join("| " + " | ".join(row) + " |" for row in rows)
    separator = ["---"] * len(headers)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(separator) + " |",
    ]
    for row in rows:
        padded = row + [""] * max(0, len(headers) - len(row))
        lines.append("| " + " | ".join(padded[: len(headers)]) + " |")
    return "\n".join(lines)


def chunk_table_rows(block: dict[str, Any], max_rows: int, max_tokens: int) -> list[dict[str, Any]]:
    headers = [normalize(cell) for cell in block.get("headers") or []]
    rows = [[normalize(cell) for cell in row] for row in block.get("rows") or []]
    if not rows:
        return [
            {
                "row_start": None,
                "row_end": None,
                "text": normalize(block.get("raw_markdown")),
            }
        ]
    chunks = []
    start = 0
    while start < len(rows):
        group: list[list[str]] = []
        end = start
        while end < len(rows) and len(group) < max_rows:
            candidate = group + [rows[end]]
            if group and count_tokens(table_markdown(headers, candidate)) > max_tokens:
                break
            group = candidate
            end += 1
        if not group:
            group = [rows[start]]
            end = start + 1
        chunks.append(
            {
                "row_start": start + 1,
                "row_end": start + len(group),
                "text": table_markdown(headers, group),
            }
        )
        start = end
    return chunks


def stable_chunk_id(document_id: int, ordinal: int) -> str:
    return f"vld:{document_id}:chunk:{ordinal:05d}"


def stable_point_id(chunk_id: str) -> str:
    return str(uuid.uuid5(CHUNK_NAMESPACE, chunk_id))


def make_chunk(
    document: dict[str, Any],
    node: dict[str, Any],
    ancestors: list[dict[str, Any]],
    chunk_type: str,
    ordinal: int,
    content_text: str,
    *,
    block: dict[str, Any],
    table_info: dict[str, Any] | None = None,
) -> dict[str, Any]:
    metadata = document["metadata"]
    document_id = int(metadata["document_id"])
    chunk_id = stable_chunk_id(document_id, ordinal)
    context = ancestor_context(document, ancestors, node)
    content_text = normalize(content_text)
    retrieval_text = join_natural([context, content_text])
    table_info = table_info or {}
    legal_path = [normalize(item) for item in node.get("path") or [] if normalize(item)]
    source_doc_title = normalize(metadata.get("title"))
    document_number = normalize(metadata.get("document_number"))
    return {
        "id": stable_point_id(chunk_id),
        "chunk_id": chunk_id,
        "chunk_type": chunk_type,
        "dataset": "vietnamese-legal-documents",
        "document_id": document_id,
        "document_number": document_number,
        "document_title": source_doc_title,
        "legal_type": normalize(metadata.get("legal_type")),
        "legal_sectors": normalize(metadata.get("legal_sectors")),
        "issuing_authority": normalize(metadata.get("issuing_authority")),
        "issuance_date": normalize(metadata.get("issuance_date")),
        "source_url": normalize(metadata.get("url")),
        "node_id": node.get("node_id"),
        "node_type": node.get("type"),
        "node_label": normalize(node.get("label")),
        "node_number": normalize(node.get("number")),
        "node_title": normalize(node.get("title")),
        "legal_path": legal_path,
        "legal_path_text": " > ".join(legal_path),
        "article_no": nearest_ancestor_display(ancestors + [node], "article", label_only=True),
        "article_title": nearest_ancestor_title(ancestors + [node], "article"),
        "clause_no": nearest_ancestor_number(ancestors + [node], "clause"),
        "point_no": nearest_ancestor_number(ancestors + [node], "point"),
        "appendix": nearest_ancestor_display(ancestors + [node], "appendix", label_only=False),
        "contains_table": chunk_type.startswith("table"),
        "table_id": table_info.get("table_id", ""),
        "table_caption": table_info.get("caption", ""),
        "table_headers": table_info.get("headers", []),
        "table_row_start": table_info.get("row_start"),
        "table_row_end": table_info.get("row_end"),
        "table_total_rows": table_info.get("total_rows"),
        "line_start": block.get("line_start"),
        "line_end": block.get("line_end"),
        "token_estimate": count_tokens(retrieval_text),
        "content_text": content_text,
        "retrieval_text": retrieval_text,
        "retrieval_text_sha1": hashlib.sha1(retrieval_text.encode("utf-8")).hexdigest(),
        # Compatibility with current Qdrant ingestion/search helpers.
        "point_id": stable_point_id(chunk_id),
        "canonical_article_id": f"vld:{document_id}",
        "chunk_index": ordinal,
        "chunk_count": None,
        "content_tree_path": legal_path,
        "chunk_method": chunk_type,
        "article_title": nearest_ancestor_display(ancestors + [node], "article", label_only=False),
        "article_no_normalized": nearest_ancestor_display(ancestors + [node], "article", label_only=True),
        "chapter_title": nearest_ancestor_display(ancestors + [node], "chapter", label_only=False),
        "topic_number": None,
        "topic_title": normalize(metadata.get("legal_sectors")).split(",")[0].strip() if metadata.get("legal_sectors") else "",
        "subject_title": normalize(metadata.get("legal_type")),
        "source_note_text": source_doc_title,
        "related_note_text": "",
        "source_law_id_candidates": [document_number] if document_number else [],
        "source_article_no_candidates": [nearest_ancestor_display(ancestors + [node], "article", label_only=True)]
        if nearest_ancestor_display(ancestors + [node], "article", label_only=True)
        else [],
        "source_doc_title_candidates": [source_doc_title] if source_doc_title else [],
        "citation_confidence": "vld_tree",
        "content_preview": content_text[:500],
    }


def nearest_ancestor(nodes: list[dict[str, Any]], node_type: str) -> dict[str, Any] | None:
    for node in reversed(nodes):
        if node.get("type") == node_type:
            return node
    return None


def nearest_ancestor_number(nodes: list[dict[str, Any]], node_type: str) -> str:
    node = nearest_ancestor(nodes, node_type)
    return normalize(node.get("number")) if node else ""


def nearest_ancestor_title(nodes: list[dict[str, Any]], node_type: str) -> str:
    node = nearest_ancestor(nodes, node_type)
    return normalize(node.get("title")) if node else ""


def nearest_ancestor_display(nodes: list[dict[str, Any]], node_type: str, *, label_only: bool) -> str:
    node = nearest_ancestor(nodes, node_type)
    if not node:
        return ""
    return normalize(node.get("label") if label_only else node.get("display"))


def build_chunks_for_tree(document: dict[str, Any], max_text_tokens: int, table_rows_per_chunk: int) -> list[dict[str, Any]]:
    chunks: list[dict[str, Any]] = []
    ordinal = 0

    def add_chunk(node: dict[str, Any], ancestors: list[dict[str, Any]], chunk_type: str, text: str, block: dict[str, Any], table_info: dict[str, Any] | None = None) -> None:
        nonlocal ordinal
        chunks.append(
            make_chunk(
                document,
                node,
                ancestors,
                chunk_type,
                ordinal,
                text,
                block=block,
                table_info=table_info,
            )
        )
        ordinal += 1

    def visit(node: dict[str, Any], ancestors: list[dict[str, Any]]) -> None:
        for block_index, block in enumerate(node.get("blocks", [])):
            block_type = block.get("type")
            if block_type == "text":
                if not is_indexable_text_node(node):
                    continue
                surface = surface_node_text(node, block.get("text", ""))
                context_budget = content_budget(document, ancestors, node, max_text_tokens)
                for piece in split_long_text(surface, max_tokens=context_budget):
                    add_chunk(node, ancestors, text_chunk_type(node), piece, block)
            elif block_type == "heading":
                # Headings are useful as context for following tables but rarely
                # worth standalone chunks.
                continue
            elif block_type == "table":
                table_id = f"{node.get('node_id')}:table:{block_index:03d}"
                table_context = " ".join(
                    part
                    for part in [
                        normalize(block.get("caption")),
                        " ".join(normalize(cell) for cell in block.get("headers") or []),
                    ]
                    if part
                )
                table_budget = content_budget(
                    document,
                    ancestors,
                    node,
                    max_text_tokens,
                    extra_context=table_context,
                )
                for table_piece in chunk_table_rows(
                    block,
                    max_rows=table_rows_per_chunk,
                    max_tokens=table_budget,
                ):
                    text = f"{table_context}. {table_piece['text']}".strip(". ")
                    add_chunk(
                        node,
                        ancestors,
                        "table_chunk",
                        text,
                        block,
                        table_info={
                            "table_id": table_id,
                            "caption": normalize(block.get("caption")),
                            "headers": [normalize(cell) for cell in block.get("headers") or []],
                            "row_start": table_piece["row_start"],
                            "row_end": table_piece["row_end"],
                            "total_rows": block.get("row_count"),
                        },
                    )
        for child in node.get("children", []):
            visit(child, ancestors + [node])

    visit(document["tree"], [])
    chunk_count = len(chunks)
    for index, chunk in enumerate(chunks):
        chunk["chunk_index"] = index
        chunk["chunk_count"] = chunk_count
    return chunks


def content_budget(
    document: dict[str, Any],
    ancestors: list[dict[str, Any]],
    node: dict[str, Any],
    max_tokens: int,
    extra_context: str = "",
) -> int:
    context = ancestor_context(document, ancestors, node)
    context_tokens = count_tokens(join_natural([context, extra_context]))
    return max(128, max_tokens - context_tokens - 8)


def text_chunk_type(node: dict[str, Any]) -> str:
    node_type = node.get("type")
    if node_type == "article":
        return "article_text_chunk"
    if node_type == "clause":
        return "clause_text_chunk"
    if node_type == "point":
        return "point_text_chunk"
    if node_type == "appendix":
        return "appendix_text_chunk"
    if node_type == "footer":
        return "footer_text_chunk"
    return "text_chunk"


def is_indexable_text_node(node: dict[str, Any]) -> bool:
    """Return True for substantive legal content nodes only.

    Root preambles, pure document headings, signatures, and other footer-like
    material are useful as metadata/context at most, not standalone recall
    chunks.
    """
    return node.get("type") in {"article", "clause", "point", "subpoint"}


def make_qdrant_preview(chunk: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": chunk["id"],
        "payload": {key: value for key, value in chunk.items() if key != "id"},
        "vectors": {
            "dense": {"model": "BAAI/bge-m3", "source": "local", "status": "not_embedded"},
            "sparse": {"model": "Qdrant/bm25", "source": "qdrant", "field": "retrieval_text"},
        },
    }


def build_documents(vld_root: Path, ids: set[int]) -> list[dict[str, Any]]:
    metadata = load_metadata(vld_root, ids)
    contents = load_contents(vld_root, ids)
    documents = []
    for doc_id in sorted(ids):
        if doc_id in metadata and doc_id in contents:
            documents.append(build_tree(metadata[doc_id], contents[doc_id]))
    return documents


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vld-root", type=Path, default=Path("data/vietnamese-legal-documents"))
    parser.add_argument("--output-dir", type=Path, default=Path("build/vld_chunk_preview"))
    parser.add_argument("--document-id", action="append", default=[], help="Document id(s), repeat or comma-separate.")
    parser.add_argument("--ids-file", type=Path, help="Optional UTF-8 file containing document ids.")
    parser.add_argument("--max-text-tokens", type=int, default=2048)
    parser.add_argument("--table-rows-per-chunk", type=int, default=8)
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
    documents = build_documents(args.vld_root, ids)

    all_chunks: list[dict[str, Any]] = []
    per_doc = []
    for document in documents:
        chunks = build_chunks_for_tree(document, args.max_text_tokens, args.table_rows_per_chunk)
        all_chunks.extend(chunks)
        type_counts = Counter(chunk["chunk_type"] for chunk in chunks)
        per_doc.append(
            {
                "document_id": document["metadata"]["document_id"],
                "document_number": document["metadata"]["document_number"],
                "title": document["metadata"]["title"],
                "chunk_count": len(chunks),
                "chunk_type_counts": dict(type_counts),
                "table_chunks": sum(1 for chunk in chunks if chunk["contains_table"]),
            }
        )

    chunks_json = args.output_dir / "chunks.json"
    chunks_jsonl = args.output_dir / "chunks.jsonl"
    preview_jsonl = args.output_dir / "qdrant_payload_preview.jsonl"
    summary_json = args.output_dir / "summary.json"
    chunks_json.write_text(json.dumps(all_chunks, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with chunks_jsonl.open("w", encoding="utf-8", newline="\n") as handle:
        for chunk in all_chunks:
            handle.write(json.dumps(chunk, ensure_ascii=False) + "\n")
    with preview_jsonl.open("w", encoding="utf-8", newline="\n") as handle:
        for chunk in all_chunks:
            handle.write(json.dumps(make_qdrant_preview(chunk), ensure_ascii=False) + "\n")

    summary = {
        "requested_ids": sorted(ids),
        "built_documents": len(documents),
        "chunk_count": len(all_chunks),
        "chunk_type_counts": dict(Counter(chunk["chunk_type"] for chunk in all_chunks)),
        "table_chunk_count": sum(1 for chunk in all_chunks if chunk["contains_table"]),
        "outputs": {
            "chunks_json": str(chunks_json),
            "chunks_jsonl": str(chunks_jsonl),
            "qdrant_payload_preview": str(preview_jsonl),
            "summary_json": str(summary_json),
        },
        "documents": per_doc,
    }
    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
