from __future__ import annotations

import unittest

from annual_report_agent.schemas import Chunk

try:
    import numpy as np

    from annual_report_agent.retrieval.dense import DenseRetriever
except ImportError:
    np = None
    DenseRetriever = None


@unittest.skipIf(np is None, "numpy is not installed in the dependency-free smoke environment")
class FakeEncoder:
    def encode_documents(self, texts):
        mapping = {
            "研发投入": [1.0, 0.0],
            "经营现金流": [0.0, 1.0],
        }
        return np.asarray([mapping[text] for text in texts], dtype=np.float32)

    def encode_queries(self, texts):
        if texts == ["研发情况"]:
            return np.asarray([[1.0, 0.0]], dtype=np.float32)
        return np.asarray([[0.0, 1.0]], dtype=np.float32)


@unittest.skipIf(np is None, "numpy is not installed in the dependency-free smoke environment")
class DenseRetrieverTests(unittest.TestCase):
    def setUp(self) -> None:
        self.chunks = [
            Chunk("a", "d1", "甲公司", 2024, "研发", 1, "研发投入"),
            Chunk("b", "d1", "甲公司", 2024, "现金流", 2, "经营现金流"),
        ]

    def test_dense_ranking(self) -> None:
        retriever = DenseRetriever(self.chunks, FakeEncoder())
        results = retriever.search("研发情况", top_k=2)
        self.assertEqual([result.chunk.chunk_id for result in results], ["a", "b"])

    def test_empty_top_k(self) -> None:
        retriever = DenseRetriever(self.chunks, FakeEncoder())
        self.assertEqual(retriever.search("研发情况", top_k=0), [])


if __name__ == "__main__":
    unittest.main()
