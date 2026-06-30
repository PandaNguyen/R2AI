from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from r2ai.search.contracts import SearchHit, SearchQuery, SearchResponse
from r2ai.search.ir_main_flow import (
    IRMainFlowConfig,
    build_road2ai_search_config,
    load_sub_query_map,
    response_to_submission_row,
    run_ir_main_flow,
)


class FakeIRBackend:
    def __init__(self) -> None:
        self.queries: list[SearchQuery] = []
        self.unloaded = False

    def search_many(self, queries: list[SearchQuery]) -> list[SearchResponse]:
        self.queries.extend(queries)
        return [
            SearchResponse(
                query=query,
                backend="fake",
                metadata={"top_k": 1},
                hits=[
                    SearchHit(
                        id=f"hit-{query.id}",
                        rank=1,
                        score=0.9,
                        text="Điều 1. Nội dung demo.",
                        payload={"chunk_id": f"chunk-{query.id}"},
                        doc_refs=["01/2024/QH15|Luật Demo"],
                        article_refs=["01/2024/QH15|Luật Demo|Điều 1"],
                    )
                ],
            )
            for query in queries
        ]

    def unload(self) -> None:
        self.unloaded = True


class IRMainFlowTests(unittest.TestCase):
    def test_config_mapping_normalizes_flow_settings(self) -> None:
        config = IRMainFlowConfig.from_mapping(
            {
                "flow": {"limit": "1", "progress_every": 0},
                "road2ai": {"mode": "dense", "top_k": 2, "use_rerank": False},
            }
        )

        self.assertEqual(config.limit, 1)
        self.assertEqual(config.progress_every, 0)
        self.assertEqual(config.road2ai["mode"], "dense")

    def test_road2ai_config_preserves_huggingface_model_ids(self) -> None:
        config = build_road2ai_search_config(
            {
                "embed_model_path": "AITeamVN/Vietnamese_Embedding_v2",
                "rerank_model_path": "AITeamVN/Vietnamese_Reranker",
                "qdrant_url": "http://localhost:6333",
                "qdrant_api_key": "",
                "collection": "demo",
            }
        )

        self.assertEqual(config.embed_model_path, "AITeamVN/Vietnamese_Embedding_v2")
        self.assertEqual(config.rerank_model_path, "AITeamVN/Vietnamese_Reranker")


    def test_load_sub_query_map_accepts_object_and_aliases_numeric_ids(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "sub_queries.json"
            path.write_text(json.dumps({"1": ["điều kiện", "thủ tục"]}), encoding="utf-8")

            sub_query_map = load_sub_query_map(path)

        self.assertEqual(sub_query_map["1"], ["điều kiện", "thủ tục"])
        self.assertEqual(sub_query_map[1], ["điều kiện", "thủ tục"])


    def test_response_to_submission_row_can_generate_llm_answer(self) -> None:
        response = FakeIRBackend().search_many([SearchQuery(id="q1", text="Câu hỏi 1?")])[0]

        class FakeLLM:
            def generate(self, messages, *, max_new_tokens=None):  # noqa: ANN001
                self.messages = messages
                return "Theo Điều 1. Căn cứ pháp lý: Điều 1."

        llm = FakeLLM()
        row = response_to_submission_row(response, {"answer_mode": "llm", "context_limit": 1}, llm=llm)

        self.assertEqual(row["answer"], "Theo Điều 1. Căn cứ pháp lý: Điều 1.")
        self.assertEqual(row["relevant_articles"], ["01/2024/QH15|Luật Demo|Điều 1"])
        self.assertIn("Câu hỏi 1?", llm.messages[1]["content"])

    def test_run_ir_main_flow_reads_questions_and_writes_submission_json(self) -> None:
        backend = FakeIRBackend()
        with tempfile.TemporaryDirectory() as tmpdir:
            questions_path = Path(tmpdir) / "questions.json"
            output_path = Path(tmpdir) / "results.json"
            ir_output_path = Path(tmpdir) / "ir.jsonl"
            questions_path.write_text(
                json.dumps(
                    {
                        "questions": [
                            {"id": "q1", "question": "Câu hỏi 1?"},
                            {"id": "q2", "question": "Câu hỏi 2?"},
                        ]
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            count = run_ir_main_flow(
                questions_path,
                output_path,
                config=IRMainFlowConfig(limit=1, progress_every=0, ir_output=ir_output_path),
                backend=backend,
            )

            rows = json.loads(output_path.read_text(encoding="utf-8"))
            ir_rows = [json.loads(line) for line in ir_output_path.read_text(encoding="utf-8").splitlines()]

        self.assertEqual(count, 1)
        self.assertTrue(backend.unloaded)
        self.assertEqual([query.id for query in backend.queries], ["q1"])
        self.assertEqual(rows[0]["id"], "q1")
        self.assertEqual(rows[0]["answer"], "")
        self.assertEqual(rows[0]["relevant_articles"], ["01/2024/QH15|Luật Demo|Điều 1"])
        self.assertEqual(ir_rows[0]["result"]["backend"], "fake")
        self.assertEqual(ir_rows[0]["result"]["results"][0]["payload"]["chunk_id"], "chunk-q1")


if __name__ == "__main__":
    unittest.main()
