from __future__ import annotations

import unittest

from annual_report_agent.retrieval import BM25Retriever
from annual_report_agent.schemas import Chunk


class BM25RetrieverTests(unittest.TestCase):
    def setUp(self) -> None:
        self.chunks = [
            Chunk("a", "d1", "甲公司", 2024, "研发", 1, "研发投入为十亿元"),
            Chunk("b", "d2", "乙公司", 2024, "现金流", 2, "现金流净额为八亿元"),
        ]

    def test_relevant_chunk_ranks_first(self) -> None:
        results = BM25Retriever(self.chunks).search("甲公司研发投入", top_k=2)
        self.assertEqual(results[0].chunk.chunk_id, "a")

    def test_non_positive_top_k_returns_empty(self) -> None:
        self.assertEqual(BM25Retriever(self.chunks).search("研发", top_k=0), [])


if __name__ == "__main__":
    unittest.main()
