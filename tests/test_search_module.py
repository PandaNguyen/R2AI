from __future__ import annotations

import unittest
from unittest.mock import patch

from r2ai.indexing.config import QdrantSearchConfig
from r2ai.qa.llm import CallableChatLLM
from r2ai.search.contracts import SearchHit, SearchQuery, SearchResponse
from r2ai.search.pipeline import SearchAnswerPipeline, SearchPipelineConfig, build_context_blocks, collect_references
from r2ai.search.qdrant_backend import QdrantSearchBackend


class FakeBackend:
    def __init__(self, hits: list[SearchHit]) -> None:
        self.hits = hits
        self.queries: list[SearchQuery] = []

    def search(self, query: SearchQuery) -> SearchResponse:
        self.queries.append(query)
        return SearchResponse(query=query, hits=self.hits, backend="fake", metadata={"top_k": len(self.hits)})


def demo_hits() -> list[SearchHit]:
    return [
        SearchHit(
            id="chunk-1",
            rank=1,
            score=0.9,
            text="Doanh nghiệp nhỏ và vừa được hỗ trợ theo Điều 4.",
            doc_refs=["04/2017/QH14|Luật Hỗ trợ doanh nghiệp nhỏ và vừa"],
            article_refs=["04/2017/QH14|Luật Hỗ trợ doanh nghiệp nhỏ và vừa|Điều 4"],
        ),
        SearchHit(
            id="chunk-2",
            rank=2,
            score=0.8,
            text="Việc hỗ trợ phải có trọng tâm theo Điều 5.",
            doc_refs=["04/2017/QH14|Luật Hỗ trợ doanh nghiệp nhỏ và vừa"],
            article_refs=["04/2017/QH14|Luật Hỗ trợ doanh nghiệp nhỏ và vừa|Điều 5"],
        ),
    ]


class SearchModuleTests(unittest.TestCase):
    def test_collect_references_deduplicates_docs_and_keeps_article_order(self) -> None:
        docs, articles = collect_references(demo_hits())

        self.assertEqual(docs, ["04/2017/QH14|Luật Hỗ trợ doanh nghiệp nhỏ và vừa"])
        self.assertEqual(
            articles,
            [
                "04/2017/QH14|Luật Hỗ trợ doanh nghiệp nhỏ và vừa|Điều 4",
                "04/2017/QH14|Luật Hỗ trợ doanh nghiệp nhỏ và vừa|Điều 5",
            ],
        )

    def test_build_context_blocks_respects_limit_and_char_budget(self) -> None:
        contexts = build_context_blocks(demo_hits(), max_chars=55, limit=2)

        self.assertEqual(len(contexts), 1)
        self.assertEqual(contexts[0].article_ref, "04/2017/QH14|Luật Hỗ trợ doanh nghiệp nhỏ và vừa|Điều 4")

    def test_pipeline_can_return_citations_without_llm(self) -> None:
        pipeline = SearchAnswerPipeline(FakeBackend(demo_hits()))

        result = pipeline.answer(SearchQuery(id=1, text="SME được hỗ trợ ra sao?"))

        self.assertEqual(result.answer, "")
        self.assertEqual(result.id, 1)
        self.assertEqual(result.metadata["backend"], "fake")
        self.assertEqual(result.as_submission_row()["relevant_articles"][0].rsplit("|", 1)[-1], "Điều 4")

    def test_pipeline_can_generate_answer_through_chat_llm_interface(self) -> None:
        llm = CallableChatLLM(lambda messages: "Được hỗ trợ theo Điều 4. Căn cứ pháp lý: Điều 4.")
        pipeline = SearchAnswerPipeline(
            FakeBackend(demo_hits()),
            llm=llm,
            config=SearchPipelineConfig(context_limit=1),
        )

        result = pipeline.answer("SME được hỗ trợ ra sao?")

        self.assertEqual(result.answer, "Được hỗ trợ theo Điều 4. Căn cứ pháp lý: Điều 4.")
        self.assertEqual(result.metadata["allowed_article_numbers"], ["Điều 4", "Điều 5"])
        self.assertEqual(len(result.contexts), 1)

    def test_qdrant_backend_adapts_existing_search_result(self) -> None:
        config = QdrantSearchConfig(
            collection_name="demo",
            qdrant_url="https://example.com",
            qdrant_api_key="secret",
            query_text="placeholder",
            search_mode="dense",
        )
        raw_result = {
            "collection_name": "demo",
            "query_text": "SME",
            "search_mode": "dense",
            "doc_title_format": "type1",
            "results": [
                {
                    "id": "chunk-1",
                    "rank": 1,
                    "score": 0.95,
                    "payload": {
                        "retrieval_text": "Luật số 04/2017/QH14 Hỗ trợ doanh nghiệp nhỏ và vừa Điều 4 Nội dung.",
                        "competition_law_id": "04/2017/QH14",
                        "competition_doc_title_type1": "Luật Hỗ trợ doanh nghiệp nhỏ và vừa",
                        "competition_article_no": "Điều 4",
                    },
                }
            ],
        }

        with patch("r2ai.search.qdrant_backend.search_qdrant", return_value=raw_result) as search_mock:
            response = QdrantSearchBackend(config).search(
                SearchQuery(
                    text="SME",
                    query_vector=[0.1, 0.2],
                    filters={"topic_title": "Doanh nghiệp"},
                )
            )

        called_config = search_mock.call_args.args[0]
        self.assertEqual(called_config.query_text, "SME")
        self.assertEqual(called_config.query_vector, [0.1, 0.2])
        self.assertEqual(called_config.topic_title, "Doanh nghiệp")
        self.assertEqual(response.backend, "qdrant")
        self.assertEqual(response.hits[0].doc_refs, ["04/2017/QH14|Luật Hỗ trợ doanh nghiệp nhỏ và vừa"])
        self.assertEqual(response.hits[0].article_refs, ["04/2017/QH14|Luật Hỗ trợ doanh nghiệp nhỏ và vừa|Điều 4"])


if __name__ == "__main__":
    unittest.main()
