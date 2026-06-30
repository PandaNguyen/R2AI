from __future__ import annotations

import runpy
import unittest
from pathlib import Path

from r2ai.data_ingest.vld import business_collection, business_qdrant


class VldModuleTests(unittest.TestCase):
    def test_business_qdrant_parser_keeps_ingest_defaults(self) -> None:
        args = business_qdrant.build_parser().parse_args([])

        self.assertEqual(args.vld_root, Path("data/vietnamese-legal-documents"))
        self.assertEqual(args.output_dir, Path("build/vld_business_scope"))
        self.assertTrue(args.article_only)
        self.assertEqual(args.extra_support_keyword, [])


    def test_classify_metadata_accepts_extra_support_keywords(self) -> None:
        row = {
            "id": 1,
            "document_number": "99/2024/QH15",
            "title": "Luật thử nghiệm về quyền dữ liệu",
            "legal_type": "Luật",
            "legal_sectors": "Lĩnh vực khác",
            "issuance_date": "01/01/2024",
            "issuing_authority": "Quốc hội",
            "effect_status": "In effect",
        }

        self.assertEqual(business_collection.classify_metadata(row, min_year=2000), (False, "excluded"))
        self.assertEqual(
            business_collection.classify_metadata(row, min_year=2000, extra_support_keywords=["quyền"]),
            (True, "support"),
        )

    def test_legacy_qdrant_script_wrapper_imports_package(self) -> None:
        globals_after_run = runpy.run_path("scripts/build_vld_business_qdrant.py")

        self.assertIn("main", globals_after_run)


if __name__ == "__main__":
    unittest.main()
