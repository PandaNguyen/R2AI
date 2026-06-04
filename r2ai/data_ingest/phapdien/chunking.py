"""Tree-aware chunking for phapdien articles."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from r2ai.data_ingest.phapdien.constants import (
    DEFAULT_CHUNK_OVERLAP_TOKENS,
    DEFAULT_MAX_CHUNK_TOKENS,
    MIN_LEAF_SPLIT_TOKENS,
    SENTENCE_BOUNDARY_PATTERN,
    STRUCTURE_MARKER_PATTERN,
)
from r2ai.data_ingest.phapdien.text import normalize_text


@dataclass
class TextNode:
    label: str
    text: str
    level: int
    marker_kind: str
    children: list["TextNode"] = field(default_factory=list)


@dataclass(frozen=True)
class StructuredChunk:
    text: str
    path: list[str]
    method: str


def split_long_text(
    text: str,
    max_tokens: int = DEFAULT_MAX_CHUNK_TOKENS,
    overlap_tokens: int = DEFAULT_CHUNK_OVERLAP_TOKENS,
) -> list[str]:
    """Compatibility wrapper returning chunk texts only."""
    return [chunk.text for chunk in split_structured_text(text, max_tokens=max_tokens, overlap_tokens=overlap_tokens)]


def split_structured_text(
    text: str,
    max_tokens: int = DEFAULT_MAX_CHUNK_TOKENS,
    overlap_tokens: int = DEFAULT_CHUNK_OVERLAP_TOKENS,
) -> list[StructuredChunk]:
    text = normalize_text(text)
    if not text:
        return []
    if count_tokens(text) <= max_tokens:
        return [StructuredChunk(text=text, path=[], method="article")]

    root = parse_content_tree(text)
    leaves = _leaf_chunks(root, max_tokens=max_tokens, overlap_tokens=overlap_tokens)
    if leaves:
        return leaves
    return _split_leaf_with_context([], text, path=[], max_tokens=max_tokens, overlap_tokens=overlap_tokens)


def parse_content_tree(text: str) -> TextNode:
    text = normalize_text(text)
    root = TextNode(label="", text="", level=0, marker_kind="root")
    matches = list(STRUCTURE_MARKER_PATTERN.finditer(text))
    if not matches:
        root.text = text
        return root

    stack: list[TextNode] = [root]
    for index, match in enumerate(matches):
        label = match.group("label").strip()
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        body = text[start:end].strip()
        if not body:
            continue

        level, marker_kind = _marker_level(label, stack)
        while stack and stack[-1].level >= level:
            stack.pop()
        parent = stack[-1] if stack else root
        node = TextNode(label=label.rstrip("."), text=f"{label} {body}", level=level, marker_kind=marker_kind)
        parent.children.append(node)
        stack.append(node)

    if not root.children:
        root.text = text
    return root


def count_tokens(text: str) -> int:
    text = normalize_text(text)
    if not text:
        return 0
    try:
        encoding = _get_tiktoken_encoding()
    except Exception:
        return max(1, int(len(text.split()) * 1.35))
    return len(encoding.encode(text))


def _get_tiktoken_encoding() -> Any:
    import tiktoken

    return tiktoken.get_encoding("cl100k_base")


def _marker_level(label: str, stack: list[TextNode]) -> tuple[int, str]:
    clean = label.rstrip(".")
    if clean and clean[0].isdigit():
        return len(clean.split(".")), "number"
    numeric_parent_level = 0
    for node in reversed(stack):
        if node.marker_kind == "number":
            numeric_parent_level = node.level
            break
    return numeric_parent_level + 1, "bullet"


def _leaf_chunks(
    root: TextNode,
    max_tokens: int,
    overlap_tokens: int,
) -> list[StructuredChunk]:
    chunks: list[StructuredChunk] = []

    def visit(node: TextNode, ancestors: list[TextNode]) -> None:
        if node.children:
            for child in node.children:
                visit(child, ancestors + ([node] if node.text else []))
            return

        context_lines = [ancestor.text for ancestor in ancestors if ancestor.text]
        path = [ancestor.label for ancestor in ancestors if ancestor.label]
        if node.label:
            path.append(node.label)
        chunks.extend(
            _split_leaf_with_context(
                context_lines=context_lines,
                leaf_text=node.text,
                path=path,
                max_tokens=max_tokens,
                overlap_tokens=overlap_tokens,
            )
        )

    if root.children:
        for child in root.children:
            visit(child, [])
    elif root.text:
        chunks.extend(
            _split_leaf_with_context(
                context_lines=[],
                leaf_text=root.text,
                path=[],
                max_tokens=max_tokens,
                overlap_tokens=overlap_tokens,
            )
        )
    return chunks


def _split_leaf_with_context(
    context_lines: list[str],
    leaf_text: str,
    path: list[str],
    max_tokens: int,
    overlap_tokens: int,
) -> list[StructuredChunk]:
    full_text = _join_context(context_lines, leaf_text)
    if count_tokens(full_text) <= max_tokens:
        return [StructuredChunk(text=full_text, path=path, method="tree_leaf")]

    prefix = "\n".join(context_lines)
    prefix_tokens = count_tokens(prefix)
    leaf_budget = max(MIN_LEAF_SPLIT_TOKENS, max_tokens - prefix_tokens)
    if leaf_budget >= max_tokens:
        leaf_budget = max_tokens

    leaf_parts = _split_sentences_with_overlap(leaf_text, max_tokens=leaf_budget, overlap_tokens=overlap_tokens)
    return [
        StructuredChunk(text=_join_context(context_lines, part), path=path, method="tree_leaf_sentence")
        for part in leaf_parts
        if part
    ]


def _join_context(context_lines: list[str], leaf_text: str) -> str:
    lines = [line for line in context_lines if line]
    lines.append(leaf_text)
    return "\n".join(lines)


def _split_sentences_with_overlap(text: str, max_tokens: int, overlap_tokens: int) -> list[str]:
    sentences = _merge_marker_fragments(
        [part.strip() for part in SENTENCE_BOUNDARY_PATTERN.split(normalize_text(text)) if part.strip()]
    )
    if len(sentences) <= 1:
        return _hard_split_by_token_budget(text, max_tokens=max_tokens, overlap_tokens=overlap_tokens)

    chunks: list[str] = []
    current: list[str] = []
    for sentence in sentences:
        candidate = " ".join(current + [sentence]).strip()
        if count_tokens(candidate) <= max_tokens or not current:
            current.append(sentence)
            continue
        chunks.append(" ".join(current).strip())
        current = _overlap_tail(current, overlap_tokens)
        current.append(sentence)
    if current:
        chunks.append(" ".join(current).strip())
    return chunks


def _merge_marker_fragments(parts: list[str]) -> list[str]:
    merged: list[str] = []
    index = 0
    while index < len(parts):
        part = parts[index]
        if _is_marker_fragment(part) and index + 1 < len(parts):
            merged.append(f"{part} {parts[index + 1]}".strip())
            index += 2
            continue
        merged.append(part)
        index += 1
    return merged


def _is_marker_fragment(text: str) -> bool:
    clean = text.strip()
    if not clean:
        return False
    if clean.lower() in {"a.", "b.", "c.", "d.", "đ.", "e.", "g.", "h.", "i.", "k.", "l.", "m.", "n."}:
        return True
    return bool(clean.endswith(".") and clean[:-1].replace(".", "").isdigit())


def _overlap_tail(sentences: list[str], overlap_tokens: int) -> list[str]:
    if overlap_tokens <= 0:
        return []
    tail: list[str] = []
    for sentence in reversed(sentences):
        candidate = [sentence] + tail
        if count_tokens(" ".join(candidate)) > overlap_tokens and tail:
            break
        tail = candidate
    return tail


def _hard_split_by_token_budget(text: str, max_tokens: int, overlap_tokens: int) -> list[str]:
    text = normalize_text(text)
    if not text:
        return []
    chunks = _recursive_split_by_separators(
        text=text,
        max_tokens=max_tokens,
        separators=["\n", ". ", "; ", ", ", " "],
    )
    if overlap_tokens > 0 and len(chunks) > 1:
        chunks = _add_text_overlap(chunks, overlap_tokens=overlap_tokens, max_tokens=max_tokens)
    return chunks


def _recursive_split_by_separators(text: str, max_tokens: int, separators: list[str]) -> list[str]:
    text = normalize_text(text)
    if not text:
        return []
    if count_tokens(text) <= max_tokens:
        return [text]
    if not separators:
        return _split_by_words(text, max_tokens=max_tokens)

    separator = separators[0]
    pieces = _split_keep_separator(text, separator)
    if len(pieces) <= 1:
        return _recursive_split_by_separators(text, max_tokens=max_tokens, separators=separators[1:])

    chunks: list[str] = []
    current = ""
    for piece in pieces:
        candidate = f"{current} {piece}" if current else piece.strip()
        candidate = candidate.strip()
        if count_tokens(candidate) <= max_tokens or not current:
            current = candidate
            continue

        chunks.extend(_recursive_split_by_separators(current, max_tokens=max_tokens, separators=separators[1:]))
        current = piece.strip()

    if current:
        chunks.extend(_recursive_split_by_separators(current, max_tokens=max_tokens, separators=separators[1:]))
    return chunks


def _split_keep_separator(text: str, separator: str) -> list[str]:
    if separator == " ":
        return [part for part in text.split(" ") if part]

    raw_parts = text.split(separator)
    if len(raw_parts) <= 1:
        return [text]

    pieces: list[str] = []
    for index, part in enumerate(raw_parts):
        if not part:
            continue
        suffix = separator if index < len(raw_parts) - 1 else ""
        pieces.append(f"{part}{suffix}".strip())
    return pieces


def _split_by_words(text: str, max_tokens: int) -> list[str]:
    words = text.split()
    chunks: list[str] = []
    current: list[str] = []
    for word in words:
        candidate = " ".join(current + [word])
        if count_tokens(candidate) <= max_tokens or not current:
            current.append(word)
            continue
        chunks.append(" ".join(current))
        current = [word]
    if current:
        chunks.append(" ".join(current))
    return chunks


def _add_text_overlap(chunks: list[str], overlap_tokens: int, max_tokens: int) -> list[str]:
    overlapped = [chunks[0]]
    for chunk in chunks[1:]:
        overlap = _text_overlap_tail(overlapped[-1], overlap_tokens=overlap_tokens)
        candidate = f"{overlap} {chunk}".strip() if overlap else chunk
        if overlap and count_tokens(candidate) > max_tokens:
            overlap = _shrink_overlap(overlap, chunk, max_tokens=max_tokens)
            candidate = f"{overlap} {chunk}".strip() if overlap else chunk
        overlapped.append(candidate)
    return overlapped


def _text_overlap_tail(text: str, overlap_tokens: int) -> str:
    units = [part for part in _split_keep_separator(normalize_text(text), ", ") if part]
    if len(units) <= 1:
        units = normalize_text(text).split()
    tail: list[str] = []
    for unit in reversed(units):
        candidate = [unit] + tail
        if count_tokens(" ".join(candidate)) > overlap_tokens and tail:
            break
        tail = candidate
    return " ".join(tail).strip()


def _shrink_overlap(overlap: str, chunk: str, max_tokens: int) -> str:
    units = overlap.split()
    while units and count_tokens(f"{' '.join(units)} {chunk}") > max_tokens:
        units.pop(0)
    return " ".join(units)


def _word_overlap_tail(words: list[str], overlap_tokens: int) -> list[str]:
    if overlap_tokens <= 0:
        return []
    tail: list[str] = []
    for word in reversed(words):
        candidate = [word] + tail
        if count_tokens(" ".join(candidate)) > overlap_tokens and tail:
            break
        tail = candidate
    return tail
