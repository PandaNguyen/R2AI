from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from r2ai.data_ingest.phapdien.download import has_phapdien_articles
from r2ai.indexing.qdrant_ingest import batched, make_points


class FakeDenseVector:
    def __init__(self, values: list[float]) -> None:
        self._values = values

    def tolist(self) -> list[float]:
        return self._values


class FakeDenseModel:
    def encode(self, texts, batch_size, normalize_embeddings, show_progress_bar):  # noqa: ANN001
        self.last_call = {
            "texts": texts,
            "batch_size": batch_size,
            "normalize_embeddings": normalize_embeddings,
            "show_progress_bar": show_progress_bar,
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


class PhapdienDownloadTests(unittest.TestCase):
    def test_lfs_pointer_sized_parquet_is_not_ready_data(self) -> None:
        with TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "articles-00000-of-00001.parquet"
            path.write_text("version https://git-lfs.github.com/spec/v1", encoding="utf-8")

            self.assertFalse(has_phapdien_articles(Path(tmpdir)))


if __name__ == "__main__":
    unittest.main()
