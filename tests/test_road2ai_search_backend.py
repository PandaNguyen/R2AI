from __future__ import annotations

import unittest

from r2ai.search.contracts import SearchQuery
from r2ai.search.ir_results import response_to_ir_row
from r2ai.search.road2ai_search import Road2AISearchBackend, Road2AISearchConfig, normalize_chunk_payload


class FakeRoad2AIEngine:
    collection = "demo"

    def __init__(self) -> None:
        self.calls = []
        self.unloaded = False

    def retrieve_one(self, q_item, *, query_vectors=None):  # noqa: ANN001
        self.calls.append({"q_item": q_item, "query_vectors": query_vectors})
        return {
            "id": q_item["id"],
            "question": q_item["question"],
            "chunks": [
                {
                    "rank": 1,
                    "score": 0.95,
                    "rrf_score": 0.03,
                    "dense_score": 0.90,
                    "bm25_score": 0.95,
                    "rerank_score": 0.95,
                    "point_id": "point-2",
                    "doc_id": "7317",
                    "chunk_id": "chunk-2",
                    "law_type": "Luật",
                    "law_code": "04/2017/QH14",
                    "law_title": "Luật Hỗ trợ doanh nghiệp nhỏ và vừa",
                    "file_name": "chunk-2",
                    "article_number": "2",
                    "text": "Điều 2. Nội dung hỗ trợ doanh nghiệp nhỏ và vừa.",
                }
            ],
            "sub_queries": None,
        }

    def retrieve_batch(self, questions):  # noqa: ANN001
        return [self.retrieve_one(question) for question in questions]

    def unload(self) -> None:
        self.unloaded = True


class Road2AISearchBackendTests(unittest.TestCase):
    def test_normalize_chunk_payload_maps_vld_fields_to_legacy_aliases(self) -> None:
        normalized = normalize_chunk_payload(
            {
                "document_id": 7317,
                "document_number": "07/2002/NĐ-CP",
                "document_title": "Nghị định demo",
                "legal_type": "Nghị định",
                "article_no": "Điều 1",
                "retrieval_text": "Nội dung.",
            }
        )

        self.assertEqual(normalized["doc_id"], "7317")
        self.assertEqual(normalized["law_code"], "07/2002/NĐ-CP")
        self.assertEqual(normalized["law_title"], "Nghị định demo")
        self.assertEqual(normalized["article_number"], "1")
        self.assertEqual(normalized["text"], "Nội dung.")

    def test_backend_wraps_copied_retrieval_engine_into_search_response(self) -> None:
        engine = FakeRoad2AIEngine()
        backend = Road2AISearchBackend(
            Road2AISearchConfig(collection="demo", top_k=2, retrieve_pool=3, use_rerank=True),
            engine=engine,
        )

        response = backend.search(SearchQuery(id="q1", text="SME được hỗ trợ thế nào?", query_vector=[0.1, 0.2]))

        self.assertEqual(response.backend, "road2ai")
        self.assertEqual(engine.calls[0]["q_item"], {"id": "q1", "question": "SME được hỗ trợ thế nào?"})
        self.assertEqual(engine.calls[0]["query_vectors"], {"SME được hỗ trợ thế nào?": [0.1, 0.2]})
        self.assertEqual(response.hits[0].id, "point-2")
        self.assertEqual(response.hits[0].doc_refs, ["04/2017/QH14|Luật Hỗ trợ doanh nghiệp nhỏ và vừa"])
        self.assertEqual(response.hits[0].article_refs, ["04/2017/QH14|Luật Hỗ trợ doanh nghiệp nhỏ và vừa|Điều 2"])
        self.assertEqual(response.hits[0].metadata["dense_score"], 0.90)
        self.assertEqual(response.hits[0].metadata["bm25_score"], 0.95)

        row = response_to_ir_row(response)
        self.assertEqual(row["id"], "q1")
        self.assertEqual(row["result"]["backend"], "road2ai")
        self.assertEqual(row["result"]["results"][0]["payload"]["chunk_id"], "chunk-2")

    def test_unload_delegates_to_engine(self) -> None:
        engine = FakeRoad2AIEngine()
        backend = Road2AISearchBackend(Road2AISearchConfig(collection="demo"), engine=engine)

        backend.unload()

        self.assertTrue(engine.unloaded)


if __name__ == "__main__":
    unittest.main()
