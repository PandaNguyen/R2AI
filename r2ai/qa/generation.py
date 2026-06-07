"""Grounded legal QA prompt and validation helpers."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterable

from r2ai.retrieval.qdrant_search import (
    build_retrieval_only_answer,
    payload_to_competition_articles,
    payload_to_competition_docs,
)

ARTICLE_PATTERN = re.compile(r"\bĐiều\s+\d+[A-Za-z]?\b", re.IGNORECASE)
THINK_BLOCK_PATTERN = re.compile(r"<think>.*?</think>", re.IGNORECASE | re.DOTALL)

SYSTEM_PROMPT = (
    "Bạn là trợ lý pháp lý tiếng Việt cho SME. Chỉ trả lời dựa trên các trích đoạn được cung cấp. "
    "Không bịa văn bản, không nhắc điều luật ngoài danh sách cho phép."
)


def build_qa_context_row(
    question: dict[str, Any],
    result: dict[str, Any],
    *,
    context_limit: int | None = None,
) -> dict[str, Any]:
    """Build a Kaggle-friendly QA context row from a Qdrant retrieval result."""
    doc_title_format = result.get("doc_title_format", "type1")
    relevant_docs: list[str] = []
    relevant_articles: list[str] = []
    contexts: list[dict[str, Any]] = []
    seen_docs: set[str] = set()
    seen_articles: set[str] = set()

    for item in result.get("results", []):
        payload = item.get("payload") or {}
        docs = payload_to_competition_docs(payload, doc_title_format=doc_title_format)
        articles = payload_to_competition_articles(payload, doc_title_format=doc_title_format)
        for doc in docs:
            if doc not in seen_docs:
                seen_docs.add(doc)
                relevant_docs.append(doc)
        for article in articles:
            if article not in seen_articles:
                seen_articles.add(article)
                relevant_articles.append(article)
        if context_limit is None or len(contexts) < context_limit:
            retrieval_text = str(payload.get("retrieval_text") or "").strip()
            if retrieval_text:
                contexts.append(
                    {
                        "rank": item.get("rank"),
                        "score": item.get("score"),
                        "doc_ref": docs[0] if docs else "",
                        "article_ref": articles[0] if articles else "",
                        "retrieval_text": retrieval_text,
                    }
                )

    allowed_article_numbers = article_numbers_from_refs(relevant_articles)
    return {
        "id": question["id"],
        "question": question["question"],
        "relevant_docs": relevant_docs,
        "relevant_articles": relevant_articles,
        "allowed_article_numbers": allowed_article_numbers,
        "contexts": contexts,
    }


def article_numbers_from_refs(relevant_articles: Iterable[str]) -> list[str]:
    """Extract de-duplicated article labels from Challenge article refs."""
    article_numbers: list[str] = []
    seen: set[str] = set()
    for ref in relevant_articles:
        article = normalize_article_number(str(ref).rsplit("|", maxsplit=1)[-1])
        if article and article not in seen:
            seen.add(article)
            article_numbers.append(article)
    return article_numbers


def build_qa_messages(row: dict[str, Any]) -> list[dict[str, str]]:
    """Build Qwen chat messages for one grounded legal QA row."""
    allowed_articles = row.get("allowed_article_numbers") or []
    allowed_text = "; ".join(allowed_articles) if allowed_articles else "(không có điều luật cho phép)"
    context_blocks = []
    for index, context in enumerate(row.get("contexts") or [], start=1):
        article_ref = context.get("article_ref") or "Không rõ điều luật"
        doc_ref = context.get("doc_ref") or "Không rõ văn bản"
        retrieval_text = str(context.get("retrieval_text") or "").strip()
        context_blocks.append(
            f"[{index}] {article_ref}\nVăn bản: {doc_ref}\nTrích đoạn:\n{retrieval_text}"
        )
    contexts_text = "\n\n".join(context_blocks) if context_blocks else "Không có trích đoạn pháp luật."
    user_prompt = (
        f"Câu hỏi:\n{row['question']}\n\n"
        f"Danh sách điều luật được phép nhắc trong câu trả lời:\n{allowed_text}\n\n"
        f"Các trích đoạn pháp luật đã truy hồi:\n{contexts_text}\n\n"
        "Yêu cầu:\n"
        "- Chỉ trả lời bằng tiếng Việt, không xuất JSON, không giải thích quá trình suy luận.\n"
        "- Trả lời trực tiếp, rõ ràng, dễ hiểu cho chủ doanh nghiệp/kế toán/nhân sự không chuyên luật.\n"
        "- Chỉ dùng thông tin trong trích đoạn; nếu trích đoạn không đủ để kết luận chắc chắn, hãy nói rõ giới hạn đó.\n"
        "- Không nhắc bất kỳ điều luật nào ngoài danh sách được phép.\n"
        f"- Kết thúc bằng đúng dạng: Căn cứ pháp lý: {allowed_text}."
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]


def build_repair_messages(row: dict[str, Any], invalid_answer: str) -> list[dict[str, str]]:
    """Build a second-pass prompt for answers that mention disallowed articles."""
    allowed_articles = row.get("allowed_article_numbers") or []
    allowed_text = "; ".join(allowed_articles) if allowed_articles else "(không có điều luật cho phép)"
    disallowed = "; ".join(find_disallowed_articles(invalid_answer, allowed_articles))
    return build_qa_messages(row) + [
        {
            "role": "assistant",
            "content": invalid_answer,
        },
        {
            "role": "user",
            "content": (
                "Câu trả lời trên đã nhắc điều luật ngoài danh sách cho phép"
                f"{': ' + disallowed if disallowed else ''}. "
                f"Hãy viết lại, chỉ nhắc các điều sau: {allowed_text}. "
                "Chỉ trả lời phần answer cuối cùng, không JSON, không reasoning."
            ),
        },
    ]


def sanitize_generated_answer(text: str) -> str:
    """Remove Qwen thinking blocks and normalize whitespace for submission."""
    cleaned = THINK_BLOCK_PATTERN.sub("", text)
    start = cleaned.lower().find("</think>")
    if start >= 0:
        cleaned = cleaned[start + len("</think>") :]
    cleaned = cleaned.replace("<think>", "")
    lines = [line.strip() for line in cleaned.strip().splitlines()]
    return "\n".join(line for line in lines if line).strip()


def validate_answer_articles(answer: str, allowed_articles: Iterable[str]) -> list[str]:
    """Return disallowed article mentions found in an answer."""
    return find_disallowed_articles(answer, allowed_articles)


def find_disallowed_articles(answer: str, allowed_articles: Iterable[str]) -> list[str]:
    allowed = {normalize_article_number(article) for article in allowed_articles}
    disallowed: list[str] = []
    seen: set[str] = set()
    for article in extract_article_numbers(answer):
        if article not in allowed and article not in seen:
            seen.add(article)
            disallowed.append(article)
    return disallowed


def extract_article_numbers(text: str) -> list[str]:
    """Extract normalized ``Điều X`` mentions in first-seen order."""
    articles: list[str] = []
    seen: set[str] = set()
    for match in ARTICLE_PATTERN.finditer(text or ""):
        article = normalize_article_number(match.group(0))
        if article and article not in seen:
            seen.add(article)
            articles.append(article)
    return articles


def normalize_article_number(value: str) -> str:
    match = ARTICLE_PATTERN.search(value or "")
    if not match:
        return ""
    parts = match.group(0).split()
    return "Điều " + parts[-1]


def fallback_answer(allowed_articles: Iterable[str]) -> str:
    return build_retrieval_only_answer(list(allowed_articles))


def load_json_rows(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON list in {path}.")
    return data


def load_jsonl_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl_rows(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
            handle.write("\n")
