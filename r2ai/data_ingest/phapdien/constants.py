"""Constants for phapdien ingestion."""

from __future__ import annotations

import re

LONG_ARTICLE_CHARS = 3000
TARGET_CHUNK_CHARS = 1600
CHUNK_OVERLAP_CHARS = 200
CONTENT_PREVIEW_CHARS = 500
DEFAULT_MAX_CHUNK_TOKENS = 2048
DEFAULT_CHUNK_OVERLAP_TOKENS = 256
MIN_LEAF_SPLIT_TOKENS = 80

LAW_ID_PATTERN = re.compile(
    r"\b\d{1,4}/(?:\d{4}/)?(?:QH\d+|NĐ-CP|ND-CP|TT-[A-ZĐ]+|QĐ-[A-ZĐ]+|QD-[A-ZĐ]+|"
    r"VBHN-[A-ZĐ]+|NQ-[A-ZĐ]+|CP|TTLT-[A-ZĐ-]+|PL-UBTVQH\d+|UBTVQH\d+)\b",
    re.IGNORECASE,
)
ARTICLE_PATTERN = re.compile(r"\bĐiều\s+([0-9]+(?:\.[0-9A-Za-zĐđ]+)*)", re.IGNORECASE)
SECTION_BOUNDARY_PATTERN = re.compile(
    r"(?=(?:^|\s)(?:\d+\.\s+|[a-zđ]\)\s+|Điều\s+\d+|Khoản\s+\d+|Điểm\s+[a-zđ]))",
    re.IGNORECASE,
)
SENTENCE_BOUNDARY_PATTERN = re.compile(r"(?<=[.!?])\s+")
STRUCTURE_MARKER_PATTERN = re.compile(
    r"(?:(?<=^)|(?<=\s))(?P<label>\d+(?:\.\d+)*\.|[a-zđ]\)|[a-zđ]\.)\s+",
    re.IGNORECASE,
)
