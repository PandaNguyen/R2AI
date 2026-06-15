from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from r2ai.cli import build_parser
from r2ai.indexing.config import QdrantSearchConfig


class CliTests(unittest.TestCase):
    def test_search_collection_defaults_to_env_when_flag_is_omitted(self) -> None:
        args = build_parser().parse_args(["search-qdrant", "doanh nghiệp"])

        self.assertIsNone(args.collection)

        with patch.dict(
            os.environ,
            {
                "QDRANT_URL": "https://example.com",
                "QDRANT_API_KEY": "secret",
                "QDRANT_COLLECTION": "vld_business_law",
            },
            clear=True,
        ):
            config = QdrantSearchConfig.from_env(
                query_text=args.query,
                collection_name=args.collection,
            )

        self.assertEqual(config.collection_name, "vld_business_law")

    def test_search_collection_flag_overrides_env(self) -> None:
        args = build_parser().parse_args(["search-qdrant", "doanh nghiệp", "--collection", "manual_collection"])

        with patch.dict(
            os.environ,
            {
                "QDRANT_URL": "https://example.com",
                "QDRANT_API_KEY": "secret",
                "QDRANT_COLLECTION": "vld_business_law",
            },
            clear=True,
        ):
            config = QdrantSearchConfig.from_env(
                query_text=args.query,
                collection_name=args.collection,
            )

        self.assertEqual(config.collection_name, "manual_collection")


if __name__ == "__main__":
    unittest.main()
