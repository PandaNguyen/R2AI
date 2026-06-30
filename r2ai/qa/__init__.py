"""Grounded answer generation modules."""

from r2ai.qa.generation import (
    QA_QUALITY_CRITERIA,
    SYSTEM_PROMPT,
    article_numbers_from_refs,
    build_qa_context_row,
    build_qa_messages,
    build_repair_messages,
    extract_article_numbers,
    fallback_answer,
    load_json_rows,
    load_jsonl_rows,
    sanitize_generated_answer,
    validate_answer_articles,
    write_jsonl_rows,
)
from r2ai.qa.llm import (
    CallableChatLLM,
    ChatLLM,
    LLMGenerationConfig,
    answer_needs_citation_repair,
    TransformersChatLLM,
    TransformersLLMConfig,
    generate_valid_answer,
)

__all__ = [
    "CallableChatLLM",
    "ChatLLM",
    "LLMGenerationConfig",
    "QA_QUALITY_CRITERIA",
    "SYSTEM_PROMPT",
    "TransformersChatLLM",
    "TransformersLLMConfig",
    "answer_needs_citation_repair",
    "article_numbers_from_refs",
    "build_qa_context_row",
    "build_qa_messages",
    "build_repair_messages",
    "extract_article_numbers",
    "fallback_answer",
    "generate_valid_answer",
    "load_json_rows",
    "load_jsonl_rows",
    "sanitize_generated_answer",
    "validate_answer_articles",
    "write_jsonl_rows",
]
