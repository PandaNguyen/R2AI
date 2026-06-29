from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from r2ai.indexing.config import QdrantSearchConfig
from r2ai.retrieval.qdrant_search import (
    build_llm_candidate_selector_prompt,
    build_qdrant_filter,
    count_llm_selector_context_tokens,
    count_llm_selector_prompt_tokens,
    extract_llm_selector_context,
    format_competition_row,
    fuse_ranked_points,
    load_search_result_rows,
    maybe_rerank_results,
    maybe_select_llm_submission_candidates,
    parse_llm_candidate_selection,
    search_qdrant,
    search_qdrant_batch,
    select_submission_rows_from_search_results,
    write_submission,
)


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
    def encode(self, texts, batch_size, normalize_embeddings, show_progress_bar, **kwargs):  # noqa: ANN001
        self.last_call = {
            "texts": texts,
            "batch_size": batch_size,
            "normalize_embeddings": normalize_embeddings,
            "show_progress_bar": show_progress_bar,
            "kwargs": kwargs,
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


def fake_article_payload(article_no: str) -> dict[str, str]:
    return {
        "article_title": article_no,
        "competition_law_id": "01/2020/QH14",
        "competition_doc_title_type1": "Luật Demo",
        "competition_article_no": article_no,
    }


class FakeModels:
    class Prefetch:
        def __init__(self, query, using, limit):  # noqa: ANN001
            self.query = query
            self.using = using
            self.limit = limit

    class Rrf:
        def __init__(self, weights=None):  # noqa: ANN001
            self.weights = weights

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
                    FakePoint("chunk-2", fake_article_payload("Điều 2"), 0.019),
                    FakePoint("chunk-1", fake_article_payload("Điều 1"), 0.018),
                ]
            )
        if using == "dense":
            return FakeQueryResponse(
                [
                    FakePoint("chunk-1", fake_article_payload("Điều 1"), 0.91),
                    FakePoint("chunk-2", fake_article_payload("Điều 2"), 0.89),
                ]
            )
        return FakeQueryResponse(
            [
                FakePoint("chunk-2", fake_article_payload("Điều 2"), 0.95),
                FakePoint("chunk-3", fake_article_payload("Điều 3"), 0.88),
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


class FakeLLMTokenizer:
    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):  # noqa: ANN001
        return "\n".join(message["content"] for message in messages)

    def __call__(self, text, add_special_tokens=False):  # noqa: ANN001
        return {"input_ids": list(text)}


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

    def test_fuse_ranked_points_applies_sparse_dense_weights(self) -> None:
        dense_hits = [FakePoint("dense-only", {"label": "Dense"}, 0.91)]
        sparse_hits = [FakePoint("sparse-only", {"label": "Sparse"}, 0.95)]

        result = fuse_ranked_points(dense_hits=dense_hits, sparse_hits=sparse_hits, limit=2, rrf_weights=(3.0, 1.0))

        self.assertEqual([item["id"] for item in result], ["sparse-only", "dense-only"])

    def test_search_qdrant_runs_server_side_hybrid_query_and_returns_results(self) -> None:
        config = QdrantSearchConfig(
            collection_name="demo",
            qdrant_url="https://example.com",
            qdrant_api_key="secret",
            query_text="quy định doanh nghiệp",
            top_k=2,
            prefetch_limit=4,
            rrf_weights=(3.0, 1.0),
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
        self.assertEqual(client.calls[0]["query"].rrf.weights, [3.0, 1.0])
        self.assertEqual(result["rrf_weights"], [3.0, 1.0])
        self.assertEqual(dense_model.last_call["texts"], ["quy định doanh nghiệp"])
        self.assertEqual(dense_model.last_call["kwargs"], {})
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

        result = search_qdrant(config, client=client, models=FakeModels, reranker={"backend": "cross_encoder", "tokenizer": tokenizer, "model": model})

        self.assertEqual([item["id"] for item in result["results"]], ["chunk-2"])
        self.assertEqual(client.calls[0]["limit"], 2)
        self.assertEqual(tokenizer.last_call["max_length"], 2304)
        self.assertEqual(result["results"][0]["rank"], 1)
        self.assertEqual(result["results"][0]["retrieval_rank"], 2)
        self.assertEqual(result["results"][0]["rerank_score"], 5.0)

    def test_search_qdrant_filters_reranked_results_by_threshold(self) -> None:
        config = QdrantSearchConfig(
            collection_name="demo",
            qdrant_url="https://example.com",
            qdrant_api_key="secret",
            query_text="quy định doanh nghiệp",
            search_mode="dense",
            query_vector=[0.1, 0.2],
            top_k=2,
            prefetch_limit=2,
            rerank=True,
            rerank_threshold=0.0,
        )
        client = FakeClient()
        tokenizer = FakeRerankerTokenizer()
        model = FakeRerankerModel()

        result = search_qdrant(config, client=client, models=FakeModels, reranker={"backend": "cross_encoder", "tokenizer": tokenizer, "model": model})

        self.assertEqual(result["rerank_threshold"], 0.0)
        self.assertEqual([item["id"] for item in result["results"]], ["chunk-2"])
        self.assertTrue(all(item["rerank_score"] >= 0.0 for item in result["results"]))

    def test_parse_llm_candidate_selection_accepts_json_object_and_code_fence(self) -> None:
        self.assertEqual(parse_llm_candidate_selection('{"selected": [1, 3], "answerable": true}', 3), [1, 3])
        self.assertEqual(parse_llm_candidate_selection('```json\n{"selected": [2, 2, 9]}\n```', 3), [2])

    def test_llm_candidate_prompt_respects_token_budget(self) -> None:
        tokenizer = FakeLLMTokenizer()
        candidates = [
            {
                "id": f"chunk-{index}",
                "rank": index,
                "score": 1.0 / index,
                "payload": {
                    **fake_article_payload(f"Điều {index}"),
                    "retrieval_text": "Nội dung rất dài. " * 200,
                },
            }
            for index in range(1, 4)
        ]

        prompt = build_llm_candidate_selector_prompt(
            "Câu hỏi demo?",
            candidates,
            tokenizer=tokenizer,
            candidate_token_budget=900,
        )

        self.assertIn("Candidate 1", prompt)
        self.assertLessEqual(count_llm_selector_context_tokens(tokenizer, extract_llm_selector_context(prompt)), 900)
        self.assertGreater(count_llm_selector_prompt_tokens(tokenizer, "", prompt), 900)

    def test_llm_candidate_selector_can_return_variable_count_beyond_top_k(self) -> None:
        config = QdrantSearchConfig(
            collection_name="demo",
            qdrant_url="https://example.com",
            qdrant_api_key="secret",
            query_text="quy định doanh nghiệp",
            top_k=1,
            llm_select_candidates=True,
            llm_selector_max_candidates=3,
        )
        results = [
            {"id": "chunk-1", "rank": 1, "score": 0.9, "payload": fake_article_payload("Điều 1")},
            {"id": "chunk-2", "rank": 2, "score": 0.8, "payload": fake_article_payload("Điều 2")},
            {"id": "chunk-3", "rank": 3, "score": 0.7, "payload": fake_article_payload("Điều 3")},
        ]

        selected = maybe_select_llm_submission_candidates(
            config,
            config.query_text,
            results,
            llm_selector=lambda prompt: '{"selected": [1, 3], "answerable": true}',
        )

        self.assertEqual([item["id"] for item in selected], ["chunk-1", "chunk-3"])
        self.assertEqual([item["rank"] for item in selected], [1, 2])
        self.assertEqual([item["pre_selector_rank"] for item in selected], [1, 3])
        self.assertEqual([item["llm_candidate_index"] for item in selected], [1, 3])

    def test_llm_candidate_selector_falls_back_to_top_k_on_invalid_json(self) -> None:
        config = QdrantSearchConfig(
            collection_name="demo",
            qdrant_url="https://example.com",
            qdrant_api_key="secret",
            query_text="quy định doanh nghiệp",
            top_k=1,
            llm_select_candidates=True,
        )
        results = [
            {"id": "chunk-1", "rank": 1, "score": 0.9, "payload": fake_article_payload("Điều 1")},
            {"id": "chunk-2", "rank": 2, "score": 0.8, "payload": fake_article_payload("Điều 2")},
        ]

        selected = maybe_select_llm_submission_candidates(
            config,
            config.query_text,
            results,
            llm_selector=lambda prompt: "Điều 1 có vẻ liên quan",
        )

        self.assertEqual([item["id"] for item in selected], ["chunk-1"])

    def test_llm_candidate_selector_falls_back_to_top_k_on_empty_selection(self) -> None:
        config = QdrantSearchConfig(
            collection_name="demo",
            qdrant_url="https://example.com",
            qdrant_api_key="secret",
            query_text="quy định doanh nghiệp",
            top_k=1,
            llm_select_candidates=True,
        )
        results = [
            {"id": "chunk-1", "rank": 1, "score": 0.9, "payload": fake_article_payload("Điều 1")},
            {"id": "chunk-2", "rank": 2, "score": 0.8, "payload": fake_article_payload("Điều 2")},
        ]

        selected = maybe_select_llm_submission_candidates(
            config,
            config.query_text,
            results,
            llm_selector=lambda prompt: '{"selected": [], "answerable": false}',
        )

        self.assertEqual([item["id"] for item in selected], ["chunk-1"])

    def test_batch_can_write_reranked_candidate_results_before_llm_selection(self) -> None:
        config = QdrantSearchConfig(
            collection_name="demo",
            qdrant_url="https://example.com",
            qdrant_api_key="secret",
            query_text="placeholder",
            search_mode="dense",
            top_k=1,
            prefetch_limit=2,
        )

        rows = search_qdrant_batch(
            config,
            questions=[{"id": "q1", "question": "quy định doanh nghiệp"}],
            query_vectors=[[0.1, 0.2]],
            client=FakeClient(),
            models=FakeModels,
            output_search_results=True,
        )

        self.assertEqual(rows[0]["id"], "q1")
        self.assertIn("result", rows[0])
        self.assertTrue(rows[0]["result"]["keep_candidate_pool"])
        self.assertFalse(rows[0]["result"]["llm_select_candidates"])
        self.assertEqual([item["id"] for item in rows[0]["result"]["results"]], ["chunk-1", "chunk-2"])

    def test_select_submission_rows_from_saved_candidate_results_loads_only_selector_phase(self) -> None:
        candidate_rows = [
            {
                "id": "q1",
                "question": "quy định doanh nghiệp",
                "result": {
                    "search_mode": "dense",
                    "top_k": 1,
                    "doc_title_format": "type1",
                    "exclude_local_documents": True,
                    "require_article": True,
                    "results": [
                        {"id": "chunk-1", "rank": 1, "score": 0.9, "payload": fake_article_payload("Điều 1")},
                        {"id": "chunk-2", "rank": 2, "score": 0.8, "payload": fake_article_payload("Điều 2")},
                    ],
                },
            }
        ]
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "candidates.jsonl"
            write_submission(path, candidate_rows, output_format="jsonl")
            loaded_rows = load_search_result_rows(path)

        rows = select_submission_rows_from_search_results(
            loaded_rows,
            QdrantSearchConfig(
                collection_name="",
                qdrant_url="",
                qdrant_api_key="",
                query_text="placeholder",
                top_k=1,
                llm_select_candidates=True,
            ),
            llm_selector=lambda prompt: '{"selected": [2], "answerable": true}',
        )

        self.assertEqual(rows[0]["id"], "q1")
        self.assertEqual(rows[0]["relevant_articles"], ["01/2020/QH14|Luật Demo|Điều 2"])
        self.assertIn("Điều 2", rows[0]["answer"])

    def test_maybe_rerank_results_deduplicates_vbhn_before_scoring(self) -> None:
        config = QdrantSearchConfig(
            collection_name="demo",
            qdrant_url="https://example.com",
            qdrant_api_key="secret",
            query_text="Điều kiện giáo dục nghề nghiệp",
            top_k=2,
            rerank=True,
        )
        tokenizer = FakeRerankerTokenizer()
        model = FakeRerankerModel()
        results = [
            {
                "id": "old-high-score",
                "rank": 1,
                "score": 0.99,
                "payload": {
                    "competition_law_id": "09/VBHN-VPQH",
                    "competition_doc_title_type1": "Văn bản hợp nhất Năm 2015 hợp nhất Luật giáo dục nghề nghiệp do Văn phòng Quốc hội ban hành",
                    "competition_article_no": "Điều 26",
                },
            },
            {
                "id": "new-low-score",
                "rank": 2,
                "score": 0.1,
                "payload": {
                    "competition_law_id": "18/VBHN-VPQH",
                    "competition_doc_title_type1": "Văn bản hợp nhất Năm 2019 hợp nhất Luật Giáo dục nghề nghiệp do Văn phòng Quốc hội ban hành",
                    "competition_article_no": "Điều 26",
                },
            },
        ]

        output = maybe_rerank_results(
            config,
            config.query_text,
            results,
            reranker={"backend": "cross_encoder", "tokenizer": tokenizer, "model": model},
        )

        self.assertEqual([item["id"] for item in output], ["new-low-score"])
        self.assertEqual(len(tokenizer.last_call["pairs"]), 1)

    def test_format_competition_row_deduplicates_vbhn_versions_by_newest_subject_and_article(self) -> None:
        result = {
            "search_mode": "hybrid",
            "doc_title_format": "type1",
            "results": [
                {
                    "id": "vbhn-2015",
                    "score": 0.9,
                    "payload": {
                        "competition_law_id": "09/VBHN-VPQH",
                        "competition_doc_title_type1": "Văn bản hợp nhất Năm 2015 hợp nhất Luật giáo dục nghề nghiệp do Văn phòng Quốc hội ban hành",
                        "competition_article_no": "Điều 26",
                    },
                },
                {
                    "id": "vbhn-2019",
                    "score": 0.8,
                    "payload": {
                        "competition_law_id": "18/VBHN-VPQH",
                        "competition_doc_title_type1": "Văn bản hợp nhất Năm 2019 hợp nhất Luật Giáo dục nghề nghiệp do Văn phòng Quốc hội ban hành",
                        "competition_article_no": "Điều 26",
                    },
                },
            ],
        }

        row = format_competition_row({"id": 1, "question": "Điều kiện giáo dục nghề nghiệp?"}, result)

        self.assertEqual(
            row["relevant_articles"],
            [
                "18/VBHN-VPQH|Văn bản hợp nhất Năm 2019 hợp nhất Luật Giáo dục nghề nghiệp do Văn phòng Quốc hội ban hành|Điều 26"
            ],
        )
        self.assertEqual(len(row["relevant_docs"]), 1)

    def test_format_competition_row_prefers_full_issue_date_for_vbhn_duplicates(self) -> None:
        result = {
            "search_mode": "hybrid",
            "doc_title_format": "type1",
            "results": [
                {
                    "id": "high-score-old",
                    "score": 0.99,
                    "payload": {
                        "competition_law_id": "09/VBHN-VPQH",
                        "competition_doc_title_type1": "Văn bản hợp nhất ngày 01/01/2015 hợp nhất Luật giáo dục nghề nghiệp do Văn phòng Quốc hội ban hành",
                        "competition_article_no": "Điều 26",
                    },
                },
                {
                    "id": "low-score-new",
                    "score": 0.1,
                    "payload": {
                        "competition_law_id": "18/VBHN-VPQH",
                        "competition_doc_title_type1": "Văn bản hợp nhất ngày 20/12/2019 hợp nhất Luật Giáo dục nghề nghiệp do Văn phòng Quốc hội ban hành",
                        "competition_article_no": "Điều 26",
                    },
                },
            ],
        }

        row = format_competition_row({"id": 1, "question": "Điều kiện giáo dục nghề nghiệp?"}, result)

        self.assertEqual(
            row["relevant_articles"],
            [
                "18/VBHN-VPQH|Văn bản hợp nhất ngày 20/12/2019 hợp nhất Luật Giáo dục nghề nghiệp do Văn phòng Quốc hội ban hành|Điều 26"
            ],
        )

    def test_format_competition_row_keeps_vbhn_same_number_with_different_subjects(self) -> None:
        result = {
            "search_mode": "hybrid",
            "doc_title_format": "type1",
            "results": [
                {
                    "id": "education",
                    "score": 0.9,
                    "payload": {
                        "competition_law_id": "18/VBHN-VPQH",
                        "competition_doc_title_type1": "Văn bản hợp nhất Năm 2019 hợp nhất Luật Giáo dục nghề nghiệp do Văn phòng Quốc hội ban hành",
                        "competition_article_no": "Điều 26",
                    },
                },
                {
                    "id": "labor",
                    "score": 0.8,
                    "payload": {
                        "competition_law_id": "18/VBHN-VPQH",
                        "competition_doc_title_type1": "Văn bản hợp nhất năm 2026 hợp nhất Bộ luật Lao động do Văn phòng Quốc hội ban hành",
                        "competition_article_no": "Điều 26",
                    },
                },
            ],
        }

        row = format_competition_row({"id": 1, "question": "Điều 26?"}, result)

        self.assertEqual(len(row["relevant_docs"]), 2)
        self.assertEqual(len(row["relevant_articles"]), 2)

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

    def test_doc_title_format_type1_handles_header_without_so(self) -> None:
        result = {
            "search_mode": "dense",
            "doc_title_format": "type1",
            "results": [
                {
                    "id": "point-1",
                    "score": 0.9,
                    "payload": {
                        "retrieval_text": "Nghị định 12/2022/NĐ-CP quy định xử phạt vi phạm hành chính về lao động Điều 6:\nNội dung",
                        "source_law_id_candidates": ["12/2022/NĐ-CP"],
                    },
                }
            ],
        }

        row = format_competition_row({"id": 1, "question": "Thử việc sai bị phạt thế nào?"}, result)

        self.assertEqual(
            row["relevant_docs"],
            ["12/2022/NĐ-CP|Nghị định Quy định xử phạt vi phạm hành chính về lao động"],
        )

    def test_format_competition_row_prefers_explicit_type1_metadata(self) -> None:
        result = {
            "search_mode": "dense",
            "doc_title_format": "type1",
            "results": [
                {
                    "id": "point-1",
                    "score": 0.9,
                    "payload": {
                        "retrieval_text": "Header không chuẩn:\nNội dung",
                        "source_law_id_candidates": ["WRONG"],
                        "competition_law_id": "04/2017/QH14",
                        "competition_doc_title_type1": "Luật Hỗ trợ doanh nghiệp nhỏ và vừa",
                        "competition_article_no": "Điều 12",
                    },
                }
            ],
        }

        row = format_competition_row({"id": 1, "question": "SME được hỗ trợ gì?"}, result)

        self.assertEqual(row["relevant_docs"], ["04/2017/QH14|Luật Hỗ trợ doanh nghiệp nhỏ và vừa"])
        self.assertEqual(
            row["relevant_articles"],
            ["04/2017/QH14|Luật Hỗ trợ doanh nghiệp nhỏ và vừa|Điều 12"],
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
