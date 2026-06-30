"""Search-to-answer orchestration that depends on interfaces, not scripts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from r2ai.qa.generation import article_numbers_from_refs, fallback_answer
from r2ai.qa.llm import ChatLLM, generate_valid_answer
from r2ai.search.contracts import (
    AnswerResult,
    ContextBlock,
    SearchBackend,
    SearchHit,
    SearchQuery,
    SearchResponse,
)

NoLlmAnswerMode = Literal["empty", "retrieval"]


@dataclass(frozen=True)
class SearchPipelineConfig:
    """Config for the thin search/answer orchestration layer."""

    context_limit: int | None = None
    max_context_chars: int = 3500
    no_llm_answer: NoLlmAnswerMode = "empty"


class SearchAnswerPipeline:
    """Run retrieval, build grounded context, and optionally call an LLM."""

    def __init__(
        self,
        backend: SearchBackend,
        *,
        llm: ChatLLM | None = None,
        config: SearchPipelineConfig | None = None,
    ) -> None:
        self.backend = backend
        self.llm = llm
        self.config = config or SearchPipelineConfig()

    def answer(self, query: SearchQuery | str) -> AnswerResult:
        normalized_query = query if isinstance(query, SearchQuery) else SearchQuery(text=query)
        response = self.backend.search(normalized_query)
        contexts = build_context_blocks(
            response.hits,
            max_chars=self.config.max_context_chars,
            limit=self.config.context_limit,
        )
        relevant_docs, relevant_articles = collect_references(response.hits)
        row = {
            "id": normalized_query.id,
            "question": normalized_query.text,
            "relevant_docs": relevant_docs,
            "relevant_articles": relevant_articles,
            "allowed_article_numbers": article_numbers_from_refs(relevant_articles),
            "contexts": [context.as_qa_context() for context in contexts],
        }
        if self.llm is None:
            answer = fallback_answer(row["allowed_article_numbers"]) if self.config.no_llm_answer == "retrieval" else ""
        else:
            answer = generate_valid_answer(self.llm, row)

        return AnswerResult(
            id=normalized_query.id,
            question=normalized_query.text,
            answer=answer,
            relevant_docs=relevant_docs,
            relevant_articles=relevant_articles,
            contexts=contexts,
            hits=response.hits,
            metadata={
                "backend": response.backend,
                "search": response.metadata,
                "allowed_article_numbers": row["allowed_article_numbers"],
            },
        )

    def answer_many(self, queries: list[SearchQuery] | list[str]) -> list[AnswerResult]:
        return [self.answer(query) for query in queries]


def collect_references(hits: list[SearchHit]) -> tuple[list[str], list[str]]:
    docs: list[str] = []
    articles: list[str] = []
    seen_docs: set[str] = set()
    seen_articles: set[str] = set()
    for hit in hits:
        for doc_ref in hit.doc_refs:
            if doc_ref and doc_ref not in seen_docs:
                seen_docs.add(doc_ref)
                docs.append(doc_ref)
        for article_ref in hit.article_refs:
            if article_ref and article_ref not in seen_articles:
                seen_articles.add(article_ref)
                articles.append(article_ref)
    return docs, articles


def build_context_blocks(
    hits: list[SearchHit],
    *,
    max_chars: int,
    limit: int | None = None,
) -> list[ContextBlock]:
    contexts: list[ContextBlock] = []
    total_chars = 0
    selected = hits if limit is None else hits[:limit]
    for hit in selected:
        text = hit.text.strip()
        if not text:
            continue
        remaining = max_chars - total_chars
        if remaining <= 0:
            break
        if len(text) > remaining:
            if remaining < 200:
                break
            text = text[:remaining].rstrip()
        contexts.append(
            ContextBlock(
                rank=hit.rank,
                score=hit.score,
                doc_ref=hit.doc_refs[0] if hit.doc_refs else "",
                article_ref=hit.article_refs[0] if hit.article_refs else "",
                text=text,
            )
        )
        total_chars += len(text) + 4
    return contexts
