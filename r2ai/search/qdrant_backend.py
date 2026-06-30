"""Adapter from the existing Qdrant retrieval code to the new search contract."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from r2ai.indexing.config import QdrantSearchConfig
from r2ai.retrieval.qdrant_search import (
    payload_to_competition_articles,
    payload_to_competition_docs,
    result_text_for_reranking,
    search_qdrant,
)
from r2ai.search.contracts import SearchHit, SearchQuery, SearchResponse


class QdrantSearchBackend:
    """SearchBackend implementation backed by ``r2ai.retrieval.qdrant_search``."""

    def __init__(
        self,
        config: QdrantSearchConfig,
        *,
        dense_model: Any | None = None,
        sparse_model: Any | None = None,
        reranker: Any | None = None,
        llm_selector: Any | None = None,
        client: Any | None = None,
        models: Any | None = None,
    ) -> None:
        self.config = config
        self.dense_model = dense_model
        self.sparse_model = sparse_model
        self.reranker = reranker
        self.llm_selector = llm_selector
        self.client = client
        self.models = models

    def search(self, query: SearchQuery) -> SearchResponse:
        config = replace(
            self.config,
            query_text=query.text,
            query_vector=query.query_vector,
            topic_title=query.filters.get("topic_title", self.config.topic_title),
            subject_title=query.filters.get("subject_title", self.config.subject_title),
            source_law_id=query.filters.get("source_law_id", self.config.source_law_id),
            source_article_no=query.filters.get("source_article_no", self.config.source_article_no),
            citation_confidence=query.filters.get("citation_confidence", self.config.citation_confidence),
            topic_number=query.filters.get("topic_number", self.config.topic_number),
        )
        result = search_qdrant(
            config,
            dense_model=self.dense_model,
            sparse_model=self.sparse_model,
            reranker=self.reranker,
            llm_selector=self.llm_selector,
            client=self.client,
            models=self.models,
        )
        doc_title_format = result.get("doc_title_format", config.doc_title_format)
        hits = [qdrant_item_to_hit(item, doc_title_format=doc_title_format) for item in result.get("results", [])]
        metadata = {key: value for key, value in result.items() if key != "results"}
        return SearchResponse(query=query, hits=hits, backend="qdrant", metadata=metadata)


def qdrant_item_to_hit(item: dict[str, Any], *, doc_title_format: str) -> SearchHit:
    payload = item.get("payload") or {}
    metadata = {key: value for key, value in item.items() if key != "payload"}
    score = item.get("score")
    return SearchHit(
        id=item.get("id"),
        rank=item.get("rank"),
        score=float(score) if isinstance(score, int | float) else None,
        text=result_text_for_reranking(item),
        payload=dict(payload),
        doc_refs=payload_to_competition_docs(payload, doc_title_format=doc_title_format),
        article_refs=payload_to_competition_articles(payload, doc_title_format=doc_title_format),
        metadata=metadata,
    )
