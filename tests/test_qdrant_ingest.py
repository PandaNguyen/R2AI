from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from r2ai.data_ingest.phapdien.download import has_phapdien_articles
from r2ai.indexing.qdrant_ingest import (
    _dense_encode_kwargs,
    _ingest_checkpoint_path,
    batched,
    _hnsw_config_diff,
    make_points,
    remote_point_matches_preview,
)


class FakeDenseVector:
    def __init__(self, values: list[float]) -> None:
        self._values = values

    def tolist(self) -> list[float]:
        return self._values


class FakeDenseModel:
    def encode(self, texts, batch_size, normalize_embeddings, show_progress_bar, **kwargs):  # noqa: ANN001
        self.last_call = {
            "texts": texts,
            "batch_size": batch_size,
            "normalize_embeddings": normalize_embeddings,
            "show_progress_bar": show_progress_bar,
            "kwargs": kwargs,
        }
        return [FakeDenseVector([float(index), 1.0]) for index, _ in enumerate(texts)]


class FakeArray:
    def __init__(self, values: list[float | int]) -> None:
        self._values = values

    def tolist(self) -> list[float | int]:
        return self._values


class FakeSparseEmbedding:
    def __init__(self) -> None:
        self.indices = FakeArray([10, 20])
        self.values = FakeArray([1.5, 2.5])


class FakeSparseModel:
    def embed(self, texts):  # noqa: ANN001
        return [FakeSparseEmbedding() for _ in texts]


class FakeModels:
    class HnswConfigDiff:
        def __init__(self, **kwargs):  # noqa: ANN003
            self.kwargs = kwargs

    class SparseVector:
        def __init__(self, indices, values):  # noqa: ANN001
            self.indices = indices
            self.values = values

    class PointStruct:
        def __init__(self, id, payload, vector):  # noqa: A002, ANN001
            self.id = id
            self.payload = payload
            self.vector = vector


class QdrantIngestTests(unittest.TestCase):
    def test_batched_groups_rows(self) -> None:
        rows = [{"i": i} for i in range(5)]

        self.assertEqual([len(batch) for batch in batched(rows, 2)], [2, 2, 1])

    def test_make_points_uses_named_dense_and_sparse_vectors(self) -> None:
        rows = [
            {"id": "point-1", "payload": {"retrieval_text": "Chủ đề Doanh nghiệp"}},
            {"id": "point-2", "payload": {"retrieval_text": "Đề mục Lao động"}},
        ]

        points = make_points(
            rows=rows,
            dense_model=FakeDenseModel(),
            sparse_model=FakeSparseModel(),
            models=FakeModels,
            dense_vector_name="dense",
            sparse_vector_name="bm25",
        )

        self.assertEqual(points[0].id, "point-1")
        self.assertEqual(points[0].vector["dense"], [0.0, 1.0])
        self.assertEqual(points[0].vector["bm25"].indices, [10, 20])
        self.assertEqual(points[0].vector["bm25"].values, [1.5, 2.5])

    def test_make_points_uses_jina_document_prompt_and_payload_metadata(self) -> None:
        rows = [{"id": "point-1", "payload": {"retrieval_text": "Chủ đề Doanh nghiệp"}}]
        dense_model = FakeDenseModel()

        points = make_points(
            rows=rows,
            dense_model=dense_model,
            sparse_model=FakeSparseModel(),
            models=FakeModels,
            dense_vector_name="dense",
            sparse_vector_name="bm25",
            dense_model_name="jinaai/jina-embeddings-v5-text-small",
            payload_metadata={"embedding_dense_model": "jinaai/jina-embeddings-v5-text-small"},
        )

        self.assertEqual(dense_model.last_call["kwargs"], {"task": "retrieval", "prompt_name": "document"})
        self.assertEqual(points[0].payload["embedding_dense_model"], "jinaai/jina-embeddings-v5-text-small")
        self.assertNotIn("embedding_dense_model", rows[0]["payload"])

    def test_jina_query_and_document_encode_kwargs(self) -> None:
        self.assertEqual(
            _dense_encode_kwargs("jinaai/jina-embeddings-v5-text-small", prompt_name="query"),
            {"task": "retrieval", "prompt_name": "query"},
        )
        self.assertEqual(_dense_encode_kwargs("AITeamVN/Vietnamese_Embedding_v2", prompt_name="query"), {})

    def test_remote_point_match_requires_embedding_metadata_when_present(self) -> None:
        preview_row = {
            "id": "point-1",
            "payload": {"retrieval_text_sha1": "abc", "document_id": "doc-1"},
        }
        metadata = {"embedding_dense_model": "jinaai/jina-embeddings-v5-text-small", "embedding_dense_size": 1024}

        self.assertFalse(
            remote_point_matches_preview(
                preview_row,
                {"id": "point-1", "payload": {"retrieval_text_sha1": "abc", "document_id": "doc-1"}},
                payload_metadata=metadata,
            )
        )
        self.assertTrue(
            remote_point_matches_preview(
                preview_row,
                {
                    "id": "point-1",
                    "payload": {
                        "retrieval_text_sha1": "abc",
                        "document_id": "doc-1",
                        "embedding_dense_model": "jinaai/jina-embeddings-v5-text-small",
                        "embedding_dense_size": 1024,
                    },
                },
                payload_metadata=metadata,
            )
        )

    def test_ingest_checkpoint_path_includes_embedding_signature(self) -> None:
        checkpoint_path = _ingest_checkpoint_path(
            Path("build"),
            "vld_business_law",
            "jinaai/jina-embeddings-v5-text-small",
            "Qdrant/bm25",
            1024,
        )

        self.assertIn("jinaai_jina-embeddings-v5-text-small", checkpoint_path.name)
        self.assertIn("1024", checkpoint_path.name)

    def test_hnsw_config_diff_uses_optional_m_and_ef_construct(self) -> None:
        self.assertIsNone(_hnsw_config_diff(FakeModels, hnsw_m=None, hnsw_ef_construct=None))

        config = _hnsw_config_diff(FakeModels, hnsw_m=48, hnsw_ef_construct=256)

        self.assertEqual(config.kwargs, {"m": 48, "ef_construct": 256})


class PhapdienDownloadTests(unittest.TestCase):
    def test_lfs_pointer_sized_parquet_is_not_ready_data(self) -> None:
        with TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "articles-00000-of-00001.parquet"
            path.write_text("version https://git-lfs.github.com/spec/v1", encoding="utf-8")

            self.assertFalse(has_phapdien_articles(Path(tmpdir)))


if __name__ == "__main__":
    unittest.main()
