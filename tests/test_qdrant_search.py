from __future__ import annotations

import unittest

from r2ai.indexing.config import QdrantSearchConfig
from r2ai.retrieval.qdrant_search import build_qdrant_filter, format_competition_row, fuse_ranked_points, search_qdrant


class FakeArray:
    def __init__(self, values):  # noqa: ANN001
        self._values = values

    def tolist(self):  # noqa: ANN201
        return list(self._values)


class FakeDenseVector:
    def __init__(self, values):  # noqa: ANN001
        self._values = values

    def tolist(self):  # noqa: ANN201
        return list(self._values)


class FakeDenseModel:
    def encode(self, texts, batch_size, normalize_embeddings, show_progress_bar):  # noqa: ANN001
        self.last_call = {
            "texts": texts,
            "batch_size": batch_size,
            "normalize_embeddings": normalize_embeddings,
            "show_progress_bar": show_progress_bar,
        }
        return [FakeDenseVector([0.4, 0.6])]


class FakeSparseEmbedding:
    def __init__(self, indices, values):  # noqa: ANN001
        self.indices = FakeArray(indices)
        self.values = FakeArray(values)


class FakeSparseModel:
    def embed(self, texts):  # noqa: ANN001
        self.last_call = {"texts": texts}
        return [FakeSparseEmbedding([1, 9], [0.3, 0.8])]


class FakePoint:
    def __init__(self, id, payload, score):  # noqa: A002, ANN001
        self.id = id
        self.payload = payload
        self.score = score


class FakeQueryResponse:
    def __init__(self, points):  # noqa: ANN001
        self.points = points


class FakeModels:
    class Prefetch:
        def __init__(self, query, using, limit):  # noqa: ANN001
            self.query = query
            self.using = using
            self.limit = limit

    class Rrf:
        pass

    class RrfQuery:
        def __init__(self, rrf):  # noqa: ANN001
            self.rrf = rrf

    class MatchValue:
        def __init__(self, value):  # noqa: ANN001
            self.value = value

    class Range:
        def __init__(self, gte, lte):  # noqa: ANN001
            self.gte = gte
            self.lte = lte

    class FieldCondition:
        def __init__(self, key, match=None, range=None):  # noqa: A002, ANN001
            self.key = key
            self.match = match
            self.range = range

    class Filter:
        def __init__(self, must):  # noqa: ANN001
            self.must = must

    class SparseVector:
        def __init__(self, indices, values):  # noqa: ANN001
            self.indices = indices
            self.values = values


class FakeClient:
    def __init__(self) -> None:
        self.calls = []

    def query_points(self, **kwargs):  # noqa: ANN003
        self.calls.append(kwargs)
        using = kwargs.get("using")
        if "prefetch" in kwargs:
            return FakeQueryResponse(
                [
                    FakePoint("chunk-2", {"article_title": "Điều 2"}, 0.019),
                    FakePoint("chunk-1", {"article_title": "Điều 1"}, 0.018),
                ]
            )
        if using == "dense":
            return FakeQueryResponse(
                [
                    FakePoint("chunk-1", {"article_title": "Điều 1"}, 0.91),
                    FakePoint("chunk-2", {"article_title": "Điều 2"}, 0.89),
                ]
            )
        return FakeQueryResponse(
            [
                FakePoint("chunk-2", {"article_title": "Điều 2"}, 0.95),
                FakePoint("chunk-3", {"article_title": "Điều 3"}, 0.88),
            ]
        )


class FakeRerankerTokenizer:
    def __call__(self, pairs, padding, truncation, return_tensors, max_length):  # noqa: ANN001
        self.last_call = {
            "pairs": pairs,
            "padding": padding,
            "truncation": truncation,
            "return_tensors": return_tensors,
            "max_length": max_length,
        }
        return {"pairs": pairs}


class FakeRerankerModel:
    def __call__(self, pairs, return_dict):  # noqa: ANN001
        scores = [-1.0 if "Điều 1" in pair[1] else 5.0 for pair in pairs]
        return {"logits": scores}


class QdrantSearchTests(unittest.TestCase):
    def test_build_qdrant_filter_includes_keyword_and_range_conditions(self) -> None:
        config = QdrantSearchConfig(
            collection_name="demo",
            qdrant_url="https://example.com",
            qdrant_api_key="secret",
            query_text="doanh nghiệp",
            topic_title="Doanh nghiệp",
            source_law_id="59/2020/QH14",
            topic_number=7,
        )

        result = build_qdrant_filter(config, FakeModels)

        self.assertIsNotNone(result)
        self.assertEqual(len(result.must), 3)
        self.assertEqual(result.must[0].key, "topic_title")
        self.assertEqual(result.must[0].match.value, "Doanh nghiệp")
        self.assertEqual(result.must[1].match.value, "59/2020/QH14")
        self.assertEqual(result.must[2].range.gte, 7)
        self.assertEqual(result.must[2].range.lte, 7)

    def test_fuse_ranked_points_uses_rrf_ordering(self) -> None:
        dense_hits = [
            FakePoint("a", {"label": "A"}, 0.91),
            FakePoint("b", {"label": "B"}, 0.90),
        ]
        sparse_hits = [
            FakePoint("b", {"label": "B"}, 0.95),
            FakePoint("c", {"label": "C"}, 0.88),
        ]

        result = fuse_ranked_points(dense_hits=dense_hits, sparse_hits=sparse_hits, limit=3)

        self.assertEqual([item["id"] for item in result], ["b", "a", "c"])
        self.assertEqual(result[0]["dense_rank"], 2)
        self.assertEqual(result[0]["sparse_rank"], 1)

    def test_search_qdrant_runs_server_side_hybrid_query_and_returns_results(self) -> None:
        config = QdrantSearchConfig(
            collection_name="demo",
            qdrant_url="https://example.com",
            qdrant_api_key="secret",
            query_text="quy định doanh nghiệp",
            top_k=2,
            prefetch_limit=4,
            topic_title="Doanh nghiệp",
        )
        dense_model = FakeDenseModel()
        sparse_model = FakeSparseModel()
        client = FakeClient()

        result = search_qdrant(
            config,
            dense_model=dense_model,
            sparse_model=sparse_model,
            client=client,
            models=FakeModels,
        )

        self.assertEqual(result["collection_name"], "demo")
        self.assertEqual([item["id"] for item in result["results"]], ["chunk-2", "chunk-1"])
        self.assertEqual(len(client.calls), 1)
        self.assertEqual(client.calls[0]["prefetch"][0].using, "bm25")
        self.assertEqual(client.calls[0]["prefetch"][1].using, "dense")
        self.assertEqual(client.calls[0]["prefetch"][0].limit, 4)
        self.assertIsInstance(client.calls[0]["query"], FakeModels.RrfQuery)
        self.assertEqual(dense_model.last_call["texts"], ["quy định doanh nghiệp"])
        self.assertEqual(sparse_model.last_call["texts"], ["quy định doanh nghiệp"])

    def test_search_qdrant_uses_precomputed_dense_vector_without_encoding(self) -> None:
        config = QdrantSearchConfig(
            collection_name="demo",
            qdrant_url="https://example.com",
            qdrant_api_key="secret",
            query_text="quy định doanh nghiệp",
            search_mode="dense",
            query_vector=[0.1, 0.2],
            top_k=2,
        )
        client = FakeClient()

        result = search_qdrant(config, client=client, models=FakeModels)

        self.assertTrue(result["used_precomputed_dense_vector"])
        self.assertEqual(client.calls[0]["query"], [0.1, 0.2])
        self.assertEqual(client.calls[0]["using"], "dense")

    def test_search_qdrant_reranks_candidate_results(self) -> None:
        config = QdrantSearchConfig(
            collection_name="demo",
            qdrant_url="https://example.com",
            qdrant_api_key="secret",
            query_text="quy định doanh nghiệp",
            search_mode="dense",
            query_vector=[0.1, 0.2],
            top_k=1,
            prefetch_limit=2,
            rerank=True,
        )
        client = FakeClient()
        tokenizer = FakeRerankerTokenizer()
        model = FakeRerankerModel()

        result = search_qdrant(config, client=client, models=FakeModels, reranker=(tokenizer, model))

        self.assertEqual([item["id"] for item in result["results"]], ["chunk-2"])
        self.assertEqual(client.calls[0]["limit"], 2)
        self.assertEqual(tokenizer.last_call["max_length"], 2304)
        self.assertEqual(result["results"][0]["rank"], 1)
        self.assertEqual(result["results"][0]["retrieval_rank"], 2)
        self.assertEqual(result["results"][0]["rerank_score"], 5.0)

    def test_format_competition_row_matches_challenge_schema(self) -> None:
        result = {
            "search_mode": "dense",
            "results": [
                {
                    "id": "point-1",
                    "score": 0.9,
                    "payload": {
                        "source_law_id_candidates": ["04/2017/QH14"],
                        "source_doc_title_candidates": ["Luật Hỗ trợ doanh nghiệp nhỏ và vừa"],
                        "source_article_no_candidates": ["Điều 4", "Điều 5"],
                    },
                }
            ],
        }

        row = format_competition_row({"id": 1, "question": "Điều kiện hỗ trợ SME?"}, result)

        self.assertEqual(set(row), {"id", "question", "answer", "relevant_docs", "relevant_articles"})
        self.assertEqual(row["id"], 1)
        self.assertEqual(row["relevant_docs"], ["04/2017/QH14|Luật Hỗ trợ doanh nghiệp nhỏ và vừa"])
        self.assertEqual(
            row["relevant_articles"],
            [
                "04/2017/QH14|Luật Hỗ trợ doanh nghiệp nhỏ và vừa|Điều 4",
            ],
        )
        self.assertIn("Điều 4", row["answer"])

    def test_doc_title_format_type1_and_type2_drop_so(self) -> None:
        base_result = {
            "search_mode": "dense",
            "results": [
                {
                    "id": "point-1",
                    "score": 0.9,
                    "payload": {
                        "retrieval_text": "Luật số 04/2017/QH14 hỗ trợ doanh nghiệp nhỏ và vừa Điều 12 Hỗ trợ công nghệ:\nNội dung",
                        "source_law_id_candidates": ["04/2017/QH14"],
                    },
                }
            ],
        }

        type1_row = format_competition_row(
            {"id": 1, "question": "Điều kiện hỗ trợ SME?"},
            {**base_result, "doc_title_format": "type1"},
        )
        type2_row = format_competition_row(
            {"id": 1, "question": "Điều kiện hỗ trợ SME?"},
            {**base_result, "doc_title_format": "type2"},
        )

        self.assertEqual(type1_row["relevant_docs"], ["04/2017/QH14|Luật Hỗ trợ doanh nghiệp nhỏ và vừa"])
        self.assertEqual(
            type2_row["relevant_docs"],
            ["04/2017/QH14|Luật 04/2017/QH14 Hỗ trợ doanh nghiệp nhỏ và vừa"],
        )

    def test_answer_article_limit_only_limits_answer_patterns(self) -> None:
        result = {
            "search_mode": "dense",
            "doc_title_format": "type2",
            "answer_article_limit": 1,
            "results": [
                {
                    "id": "point-1",
                    "score": 0.9,
                    "payload": {
                        "retrieval_text": "Luật số 04/2017/QH14 hỗ trợ doanh nghiệp nhỏ và vừa Điều 4 Điều kiện hỗ trợ:\nNội dung",
                        "source_law_id_candidates": ["04/2017/QH14"],
                    },
                },
                {
                    "id": "point-2",
                    "score": 0.8,
                    "payload": {
                        "retrieval_text": "Luật số 04/2017/QH14 hỗ trợ doanh nghiệp nhỏ và vừa Điều 5 Nguyên tắc hỗ trợ:\nNội dung",
                        "source_law_id_candidates": ["04/2017/QH14"],
                    },
                },
            ],
        }

        row = format_competition_row({"id": 1, "question": "Điều kiện hỗ trợ SME?"}, result)

        self.assertEqual(len(row["relevant_articles"]), 2)
        self.assertIn("Điều 4", row["answer"])
        self.assertNotIn("Điều 5", row["answer"])


if __name__ == "__main__":
    unittest.main()
