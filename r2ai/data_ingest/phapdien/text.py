"""Text normalization helpers for phapdien records."""

from __future__ import annotations

import hashlib
import re
import uuid
from typing import Any

from r2ai.data_ingest.phapdien.constants import ARTICLE_PATTERN, LAW_ID_PATTERN


LEGAL_SOURCE_TYPE_PATTERN = re.compile(
    r"\b(Bộ luật|Luật|Pháp lệnh|Nghị quyết|Nghị định|Thông tư liên tịch|Thông tư|Quyết định|"
    r"Văn bản hợp nhất)\b",
    re.IGNORECASE,
)
SOURCE_TITLE_STOP_PATTERN = re.compile(
    r"(?:,\s*(?:có hiệu lực|hết hiệu lực|được sửa đổi|được bổ sung)|\s+ngày\s+\d{1,2}/\d{1,2}/\d{4})",
    re.IGNORECASE,
)
PHAPDIEN_ARTICLE_TITLE_PATTERN = re.compile(r"^Điều\s+\S+\s*(?P<title>.*)$", re.IGNORECASE)
PHAPDIEN_RELATED_ARTICLE_PATTERN = re.compile(
    r"\s+Điều\s+\d+(?:\.\d+)*\.[A-ZĐ]{1,8}\.\d+(?:\.\d+)*\.\s+",
    re.IGNORECASE,
)
RELATED_CONTENT_PATTERN = re.compile(r"\s*\(?Điều này có nội dung liên quan\b.*$", re.IGNORECASE)
COMPETITION_DOC_TITLE_CODE_PATTERN = re.compile(
    r"^(?P<kind>.+?)\s+số\s+(?P<code>\S+)\s+(?P<title>.+)$",
    re.IGNORECASE,
)


def normalize_text(value: Any) -> str:
    """Return a compact UTF-8-safe string for indexing and JSONL output."""
    if value is None:
        return ""
    text = str(value).replace("\u00a0", " ")
    return re.sub(r"\s+", " ", text).strip()


def labeled_line(label: str, value: Any) -> str:
    text = normalize_text(value)
    return f"{label} {text}" if text else ""


def strip_related_content_note(value: Any) -> str:
    text = normalize_text(value)
    related_article_match = PHAPDIEN_RELATED_ARTICLE_PATTERN.search(text)
    if related_article_match:
        text = text[: related_article_match.start()]
    return RELATED_CONTENT_PATTERN.sub("", text).strip()


def extract_source_article_title(article_title: str) -> str:
    text = normalize_text(article_title)
    match = PHAPDIEN_ARTICLE_TITLE_PATTERN.match(text)
    if not match:
        return ""
    return match.group("title").lstrip(". ").strip()


def extract_source_note_context(source_note_text: str) -> dict[str, str]:
    text = normalize_text(source_note_text).strip("() ")
    lookup_text = re.sub(r"\s*/\s*", "/", text)
    article_match = ARTICLE_PATTERN.search(lookup_text)
    law_code_match = LAW_ID_PATTERN.search(lookup_text)

    law_type = ""
    law_code = law_code_match.group(0).upper() if law_code_match else ""
    source_title = ""

    if law_code_match:
        type_matches = list(LEGAL_SOURCE_TYPE_PATTERN.finditer(lookup_text[: law_code_match.start()]))
        if type_matches:
            type_match = type_matches[-1]
            law_type = normalize_text(type_match.group(1))
            title_start = type_match.start()
            title_tail = lookup_text[law_code_match.end() :]
            stop_match = SOURCE_TITLE_STOP_PATTERN.search(title_tail)
            title_end = law_code_match.end() + (stop_match.start() if stop_match else len(title_tail))
            source_title = normalize_text(lookup_text[title_start:title_end].strip(" ,.;:-"))

    if not source_title and lookup_text:
        source_title = ARTICLE_PATTERN.sub("", lookup_text, count=1).strip(" ,.;:-")

    return {
        "source_law_type": law_type,
        "source_law_code": law_code,
        "source_doc_title": normalize_text(source_title),
        "source_article_no": f"Điều {article_match.group(1)}" if article_match else "",
    }


def source_title_has_name(source_doc_title: str, law_code: str) -> bool:
    title = normalize_text(source_doc_title)
    code = normalize_text(law_code).upper()
    if not title or not code:
        return False
    idx = title.upper().find(code)
    if idx < 0:
        return False
    suffix = title[idx + len(code) :].strip(" ,.;:-")
    return bool(suffix)


def prefer_source_doc_title(current_title: str, mapped_title: str, law_code: str) -> str:
    current = normalize_text(current_title)
    mapped = normalize_text(mapped_title)
    if not mapped:
        return current
    if not current:
        return mapped
    current_has_name = source_title_has_name(current, law_code)
    mapped_has_name = source_title_has_name(mapped, law_code)
    if mapped_has_name and (not current_has_name or len(mapped) > len(current)):
        return mapped
    return current


def stable_hash(value: str, prefix: str) -> str:
    digest = hashlib.sha1(value.encode("utf-8")).hexdigest()[:20]
    return f"{prefix}_{digest}"


def stable_point_id(value: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"r2ai-phapdien:{value}"))


def extract_article_no(article_title: str) -> tuple[str, str]:
    match = ARTICLE_PATTERN.search(article_title)
    if not match:
        return "", ""
    raw = f"Điều {match.group(1)}"
    return raw, raw


def normalize_competition_doc_title(doc_title: str, law_id: str, doc_title_format: str) -> str:
    doc_title = normalize_text(doc_title)
    match = COMPETITION_DOC_TITLE_CODE_PATTERN.match(doc_title)
    if match:
        kind = normalize_text(match.group("kind"))
        code = normalize_text(match.group("code"))
        title = uppercase_first(normalize_text(match.group("title")))
        if doc_title_format == "type2":
            return normalize_text(f"{kind} {code} {title}")
        return normalize_text(f"{kind} {title}")
    doc_title = re.sub(r"\bsố\s+", "", doc_title, flags=re.IGNORECASE)
    doc_title = normalize_text(doc_title)
    if law_id and law_id in doc_title:
        prefix, title = doc_title.split(law_id, maxsplit=1)
        kind = normalize_text(prefix.strip(" ,.;:-"))
        title = uppercase_first(normalize_text(title.strip(" ,.;:-")))
        if kind and title:
            if doc_title_format == "type2":
                return normalize_text(f"{kind} {law_id} {title}")
            return normalize_text(f"{kind} {title}")
    if doc_title_format == "type2" and law_id and law_id not in doc_title:
        first_word, rest = split_first_word(doc_title)
        if rest:
            return normalize_text(f"{first_word} {law_id} {uppercase_first(rest)}")
    return doc_title


def uppercase_first(text: str) -> str:
    if not text:
        return text
    return text[0].upper() + text[1:]


def split_first_word(text: str) -> tuple[str, str]:
    parts = text.split(" ", maxsplit=1)
    if len(parts) == 1:
        return parts[0], ""
    return parts[0], parts[1]
