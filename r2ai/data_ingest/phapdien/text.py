"""Text normalization helpers for phapdien records."""

from __future__ import annotations

import hashlib
import re
import uuid
from typing import Any

from r2ai.data_ingest.phapdien.constants import ARTICLE_PATTERN


def normalize_text(value: Any) -> str:
    """Return a compact UTF-8-safe string for indexing and JSONL output."""
    if value is None:
        return ""
    text = str(value).replace("\u00a0", " ")
    return re.sub(r"\s+", " ", text).strip()


def labeled_line(label: str, value: Any) -> str:
    text = normalize_text(value)
    return f"{label} {text}" if text else ""


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
