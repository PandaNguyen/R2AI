from __future__ import annotations

import unittest

from r2ai.phapdien import (
    build_retrieval_text,
    build_source_title_map,
    canonicalize_article,
    canonical_hash_basis,
    extract_citation_candidates,
    make_retrieval_units,
    split_long_text,
    split_structured_text,
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

    def test_structured_chunks_include_parent_context_for_numeric_children(self) -> None:
        chunks = split_structured_text(
            "1. Nắm tình hình an ninh trật tự. Cụ thể là: "
            "1.1. Tình hình hoạt động của bị can, bị cáo. "
            "1.2. Các dấu hiệu tham nhũng, tiêu cực.",
            max_tokens=25,
            overlap_tokens=5,
        )

        self.assertEqual([chunk.path for chunk in chunks], [["1", "1.1"], ["1", "1.2"]])
        self.assertTrue(all("1. Nắm tình hình an ninh trật tự. Cụ thể là:" in chunk.text for chunk in chunks))
        self.assertIn("1.1. Tình hình hoạt động", chunks[0].text)
        self.assertIn("1.2. Các dấu hiệu", chunks[1].text)

    def test_nested_alpha_children_keep_numeric_parent_context(self) -> None:
        chunks = split_structured_text(
            "3. Đôn đốc nhân dân. 3.1. Thực hiện đăng ký tạm trú: "
            "a. Tiếp nhận khai báo tạm trú. b. Báo cáo Công an phường.",
            max_tokens=25,
            overlap_tokens=5,
        )

        self.assertEqual([chunk.path for chunk in chunks], [["3", "3.1", "a"], ["3", "3.1", "b"]])
        self.assertTrue(all("3. Đôn đốc nhân dân." in chunk.text for chunk in chunks))
        self.assertTrue(all("3.1. Thực hiện đăng ký tạm trú:" in chunk.text for chunk in chunks))

    def test_hyphen_inside_phrase_is_not_a_structural_bullet(self) -> None:
        chunks = split_structured_text(
            "1.3. Tổ chức chính trị - xã hội, tổ chức xã hội nghề nghiệp.",
            max_tokens=8,
            overlap_tokens=2,
        )

        self.assertEqual(chunks[0].path, ["1.3"])
        self.assertNotIn("-", chunks[0].path)

    def test_sentence_fallback_keeps_numeric_marker_with_text(self) -> None:
        chunks = split_structured_text(
            "1.1. " + " ".join(f"nội dung {i}." for i in range(30)),
            max_tokens=20,
            overlap_tokens=3,
        )

        self.assertTrue(chunks[0].text.startswith("1.1. nội dung"))

    def test_leaf_fallback_prefers_punctuation_boundaries(self) -> None:
        chunks = split_structured_text(
            "1.1. " + "đoạn một, đoạn hai; đoạn ba, đoạn bốn; đoạn năm, đoạn sáu",
            max_tokens=12,
            overlap_tokens=0,
        )

        self.assertGreater(len(chunks), 1)
        for chunk in chunks[:-1]:
            self.assertTrue(chunk.text.rstrip().endswith((",", ";", ".")), chunk.text)


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
    def test_retrieval_text_rebuilds_source_context_without_labels(self) -> None:
        text = build_retrieval_text(
            {
                "topic_title": "Doanh nghiệp, hợp tác xã",
                "subject_title": "Hỗ trợ doanh nghiệp nhỏ và vừa",
                "chapter_title": "Chương II - NỘI DUNG HỖ TRỢ",
                "article_title": "Điều 1. Phạm vi điều chỉnh",
                "source_note_text": (
                    "(Điều 1 Luật số 32/2004/QH11 An ninh Quốc gia, có hiệu lực thi hành kể từ ngày 01/07/2005)"
                ),
                "content_text": (
                    "Luật này quy định về chính sách an ninh quốc gia; nguyên tắc, nhiệm vụ, "
                    "biện pháp bảo vệ an ninh quốc gia. "
                    "Điều 4.3.NĐ.1.3. Viện pháp y quốc gia Nội dung điều liên quan. "
                    "(Điều này có nội dung liên quan đến Điều 4.3.NĐ.1.3. Viện pháp y quốc gia)"
                ),
                "related_note_text": "Liên quan Điều 13.",
            }
        )

        self.assertEqual(
            text,
            "Luật số 32/2004/QH11 An ninh Quốc gia Điều 1 Phạm vi điều chỉnh:\n"
            "Luật này quy định về chính sách an ninh quốc gia; nguyên tắc, nhiệm vụ, "
            "biện pháp bảo vệ an ninh quốc gia.",
        )

    def test_retrieval_text_uses_lawcode_title_mapping(self) -> None:
        titled_article = canonicalize_article(
            {
                "article_anchor": "#law-1",
                "article_title": "Điều 1. Phạm vi điều chỉnh",
                "source_note_text": "(Điều 1 Luật số 32/2004/QH11 An ninh Quốc gia, có hiệu lực thi hành kể từ ngày 01/07/2005)",
                "content_text": "Nội dung điều 1.",
            }
        )
        untitled_article = canonicalize_article(
            {
                "article_anchor": "#law-2",
                "article_title": "Điều 2. Đối tượng áp dụng",
                "source_note_text": "(Điều 2 Luật số 32/2004/QH11, có hiệu lực thi hành kể từ ngày 01/07/2005)",
                "content_text": "Nội dung điều 2.",
            }
        )
        source_title_by_law_code = build_source_title_map([titled_article, untitled_article])

        text = build_retrieval_text(
            untitled_article,
            source_title_by_law_code=source_title_by_law_code,
        )

        self.assertEqual(
            text,
            "Luật số 32/2004/QH11 An ninh Quốc gia Điều 2 Đối tượng áp dụng:\n"
            "Nội dung điều 2.",
        )

    def test_retrieval_units_drop_related_content_note(self) -> None:
        article = canonicalize_article(
            {
                "article_anchor": "#abc",
                "article_title": "Điều 8.3.PL.14. Bảo đảm cơ cấu dân số hợp lý",
                "topic_number": 8,
                "subject_number": 3,
                "topic_title": "Dân số",
                "subject_title": "Dân số",
                "source_note_text": "(Điều 14 Pháp lệnh số 06 /2003/PL-UBTVQH11, có hiệu lực thi hành kể từ ngày 01/05/2003)",
                "content_text": (
                    "1. Nhà nước có chính sách bảo đảm cơ cấu dân số hợp lý. "
                    "(Điều này có nội dung liên quan đến Điều 8.3.NĐ.1.2. Chính sách dân số)"
                ),
                "related_note_text": "Điều 8.3.NĐ.1.2. Chính sách dân số",
            }
        )

        units = make_retrieval_units(article)

        self.assertEqual(len(units), 1)
        self.assertIn("Pháp lệnh số 06/2003/PL-UBTVQH11 Điều 14 Bảo đảm cơ cấu dân số hợp lý", units[0]["retrieval_text"])
        self.assertNotIn("Chủ đề", units[0]["retrieval_text"])
        self.assertNotIn("Tên văn bản nguồn", units[0]["retrieval_text"])
        self.assertNotIn("Liên quan", units[0]["retrieval_text"])
        self.assertNotIn("Điều 8.3.NĐ.1.2", units[0]["retrieval_text"])
        self.assertNotIn("Điều này có nội dung liên quan", units[0]["retrieval_text"])


if __name__ == "__main__":
    unittest.main()
