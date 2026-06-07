from __future__ import annotations

import unittest

from r2ai.qa.generation import (
    build_qa_context_row,
    build_qa_messages,
    extract_article_numbers,
    fallback_answer,
    sanitize_generated_answer,
    validate_answer_articles,
)


class QaGenerationTests(unittest.TestCase):
    def test_build_qa_context_row_dedupes_refs_and_keeps_retrieval_text(self) -> None:
        result = {
            "doc_title_format": "type1",
            "results": [
                {
                    "rank": 1,
                    "score": 0.9,
                    "payload": {
                        "retrieval_text": (
                            "Luật số 04/2017/QH14 Hỗ trợ doanh nghiệp nhỏ và vừa Điều 4 Điều kiện hỗ trợ:\n"
                            "Doanh nghiệp nhỏ và vừa được hỗ trợ theo quy định."
                        ),
                        "source_law_id_candidates": ["04/2017/QH14"],
                    },
                },
                {
                    "rank": 2,
                    "score": 0.8,
                    "payload": {
                        "retrieval_text": (
                            "Luật số 04/2017/QH14 Hỗ trợ doanh nghiệp nhỏ và vừa Điều 5 Nguyên tắc hỗ trợ:\n"
                            "Việc hỗ trợ phải có trọng tâm và phù hợp nguồn lực."
                        ),
                        "source_law_id_candidates": ["04/2017/QH14"],
                    },
                },
            ],
        }

        row = build_qa_context_row({"id": 1, "question": "SME được hỗ trợ ra sao?"}, result)

        self.assertEqual(row["id"], 1)
        self.assertEqual(row["relevant_docs"], ["04/2017/QH14|Luật Hỗ trợ doanh nghiệp nhỏ và vừa"])
        self.assertEqual(
            row["relevant_articles"],
            [
                "04/2017/QH14|Luật Hỗ trợ doanh nghiệp nhỏ và vừa|Điều 4",
                "04/2017/QH14|Luật Hỗ trợ doanh nghiệp nhỏ và vừa|Điều 5",
            ],
        )
        self.assertEqual(row["allowed_article_numbers"], ["Điều 4", "Điều 5"])
        self.assertEqual(len(row["contexts"]), 2)
        self.assertIn("Doanh nghiệp nhỏ và vừa", row["contexts"][0]["retrieval_text"])

    def test_build_qa_messages_preserves_vietnamese_and_allowed_citations(self) -> None:
        row = {
            "id": 1,
            "question": "Nếu công ty giữ bản chính bằng cấp thì sao?",
            "allowed_article_numbers": ["Điều 17", "Điều 9"],
            "contexts": [
                {
                    "article_ref": "45/2019/QH14|Bộ luật Lao động|Điều 17",
                    "doc_ref": "45/2019/QH14|Bộ luật Lao động",
                    "retrieval_text": "Bộ luật số 45/2019/QH14 Bộ Luật lao động Điều 17:\nKhông được giữ bản chính giấy tờ.",
                }
            ],
        }

        messages = build_qa_messages(row)
        prompt = "\n".join(message["content"] for message in messages)

        self.assertIn("trợ lý pháp lý tiếng Việt", prompt)
        self.assertIn("bản chính bằng cấp", prompt)
        self.assertIn("Căn cứ pháp lý: Điều 17; Điều 9.", prompt)
        self.assertIn("Không được giữ bản chính giấy tờ.", prompt)

    def test_validator_detects_disallowed_articles_and_fallback_is_safe(self) -> None:
        answer = "Theo Điều 17, công ty không được giữ bản chính. Điều 8 quy định thêm về xử phạt."

        self.assertEqual(validate_answer_articles(answer, ["Điều 17"]), ["Điều 8"])
        self.assertEqual(extract_article_numbers(answer), ["Điều 17", "Điều 8"])
        self.assertEqual(
            fallback_answer(["Điều 17"]),
            "Các căn cứ pháp luật liên quan được hệ thống truy hồi gồm: Điều 17.",
        )

    def test_sanitize_generated_answer_strips_thinking_content(self) -> None:
        raw = "<think>nháp nội bộ</think>\nCâu trả lời cuối.\n\nCăn cứ pháp lý: Điều 4."

        self.assertEqual(
            sanitize_generated_answer(raw),
            "Câu trả lời cuối.\nCăn cứ pháp lý: Điều 4.",
        )


if __name__ == "__main__":
    unittest.main()
