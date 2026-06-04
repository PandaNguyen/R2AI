from __future__ import annotations

import unittest

from r2ai.phapdien import (
    build_retrieval_text,
    canonicalize_article,
    canonical_hash_basis,
    extract_citation_candidates,
    make_retrieval_units,
    split_long_text,
)


class CitationExtractionTests(unittest.TestCase):
    def test_extracts_common_vietnamese_law_ids(self) -> None:
        result = extract_citation_candidates(
            "Căn cứ Bộ luật Lao động số 45/2019/QH14 và Nghị định số 12/2022/NĐ-CP.",
            [{"text": "Bộ luật Lao động", "href": "https://example.test"}],
        )

        self.assertEqual(result["citation_confidence"], "high")
        self.assertIn("45/2019/QH14", result["source_law_id_candidates"])
        self.assertIn("12/2022/NĐ-CP", result["source_law_id_candidates"])
        self.assertIn("Bộ luật Lao động", result["source_doc_title_candidates"])

    def test_extracts_source_article_numbers(self) -> None:
        result = extract_citation_candidates(
            "(Điều 17 Bộ luật số 45/2019/QH14 Lao động ngày 20/11/2019)",
            None,
        )

        self.assertEqual(result["source_article_no_candidates"], ["Điều 17"])

    def test_medium_confidence_when_source_text_exists_without_id(self) -> None:
        result = extract_citation_candidates("Theo quy định của Luật Doanh nghiệp.", None)

        self.assertEqual(result["citation_confidence"], "medium")
        self.assertEqual(result["source_law_id_candidates"], [])


class ChunkingTests(unittest.TestCase):
    def test_short_article_stays_single_chunk(self) -> None:
        chunks = split_long_text("Nội dung ngắn của một điều luật.")

        self.assertEqual(chunks, ["Nội dung ngắn của một điều luật."])

    def test_long_article_splits_without_losing_parent_link(self) -> None:
        row = {
            "article_anchor": "#abc",
            "article_title": "Điều 20. Trách nhiệm của doanh nghiệp",
            "topic_number": 12,
            "subject_number": 1,
            "topic_title": "Doanh nghiệp, hợp tác xã",
            "subject_title": "Doanh nghiệp",
            "content_text": " ".join(f"{i}. Nội dung khoản {i}." for i in range(1, 260)),
            "source_note_text": "Luật số 59/2020/QH14",
        }
        article = canonicalize_article(row)
        units = make_retrieval_units(article)

        self.assertGreater(len(units), 1)
        self.assertEqual({unit["canonical_article_id"] for unit in units}, {article["canonical_article_id"]})
        self.assertEqual({unit["chunk_count"] for unit in units}, {len(units)})


class CanonicalIdTests(unittest.TestCase):
    def test_duplicate_disambiguator_changes_id(self) -> None:
        row = {
            "article_anchor": "#same",
            "article_title": "Điều 1. Nội dung",
            "topic_id": "topic",
            "subject_id": "subject",
            "topic_number": 1,
            "subject_number": 2,
            "content_text": "Nội dung",
        }

        self.assertEqual(canonical_hash_basis(row), canonical_hash_basis(row))
        first = canonicalize_article(row, disambiguator=0)
        second = canonicalize_article(row, disambiguator=1)

        self.assertNotEqual(first["canonical_article_id"], second["canonical_article_id"])


class RetrievalTextTests(unittest.TestCase):
    def test_retrieval_text_uses_tree_order(self) -> None:
        text = build_retrieval_text(
            {
                "topic_title": "Doanh nghiệp, hợp tác xã",
                "subject_title": "Hỗ trợ doanh nghiệp nhỏ và vừa",
                "chapter_title": "Chương II - NỘI DUNG HỖ TRỢ",
                "article_title": "Điều 12. Hỗ trợ thuế",
                "source_note_text": "Luật số 04/2017/QH14",
                "content_text": "Nội dung điều luật.",
                "related_note_text": "Liên quan Điều 13.",
            }
        )

        self.assertEqual(
            text.splitlines(),
            [
                "Chủ đề Doanh nghiệp, hợp tác xã",
                "Đề mục Hỗ trợ doanh nghiệp nhỏ và vừa",
                "Chương II - NỘI DUNG HỖ TRỢ",
                "Điều pháp điển Điều 12. Hỗ trợ thuế",
                "Nguồn Luật số 04/2017/QH14",
                "Nội dung Nội dung điều luật.",
                "Liên quan Liên quan Điều 13.",
            ],
        )


if __name__ == "__main__":
    unittest.main()
