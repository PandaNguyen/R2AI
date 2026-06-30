"""Small contracts for search backends and answer orchestration."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass(frozen=True)
class SearchQuery:
    """Input accepted by any search backend."""

    text: str
    id: Any | None = None
    query_vector: list[float] | None = None
    filters: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SearchHit:
    """Backend-neutral retrieval hit."""

    id: Any
    rank: int | None
    score: float | None
    text: str
    payload: dict[str, Any] = field(default_factory=dict)
    doc_refs: list[str] = field(default_factory=list)
    article_refs: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SearchResponse:
    """Search output normalized for QA and submission code."""

    query: SearchQuery
    hits: list[SearchHit]
    backend: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


class SearchBackend(Protocol):
    """Minimal interface for a retrieval implementation."""

    def search(self, query: SearchQuery) -> SearchResponse:
        """Return normalized retrieval hits for a query."""


@dataclass(frozen=True)
class ContextBlock:
    """One context block passed from search into answer generation."""

    rank: int | None
    score: float | None
    doc_ref: str
    article_ref: str
    text: str

    def as_qa_context(self) -> dict[str, Any]:
        return {
            "rank": self.rank,
            "score": self.score,
            "doc_ref": self.doc_ref,
            "article_ref": self.article_ref,
            "retrieval_text": self.text,
        }


@dataclass(frozen=True)
class AnswerResult:
    """Final answer shape used by higher-level scripts."""

    id: Any | None
    question: str
    answer: str
    relevant_docs: list[str]
    relevant_articles: list[str]
    contexts: list[ContextBlock]
    hits: list[SearchHit]
    metadata: dict[str, Any] = field(default_factory=dict)

    def as_submission_row(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "question": self.question,
            "answer": self.answer,
            "relevant_docs": self.relevant_docs,
            "relevant_articles": self.relevant_articles,
        }
