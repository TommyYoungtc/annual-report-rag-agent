from __future__ import annotations

import unittest

import numpy as np

from annual_report_agent.retrieval import rerank_results
from annual_report_agent.schemas import Chunk, SearchResult


class KeywordScorer:
    def score(self, query, documents):
        return np.asarray(
            [1.0 if "研发投入" in document else -1.0 for document in documents],
            dtype=np.float32,
        )


class BrokenScorer:
    def score(self, query, documents):
        return np.asarray([1.0], dtype=np.float32)


class RerankerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.results = [
            SearchResult(
                Chunk("a", "d", "公司", 2024, "业务", 1, "主营业务情况"),
                0.9,
                1,
                "hybrid",
            ),
            SearchResult(
                Chunk("b", "d", "公司", 2024, "研发", 2, "研发投入为十亿元"),
                0.7,
                2,
                "hybrid",
            ),
        ]

    def test_reranker_reorders_candidates(self) -> None:
        reranked = rerank_results("研发投入是多少？", self.results, KeywordScorer(), top_k=2)
        self.assertEqual([result.chunk.chunk_id for result in reranked], ["b", "a"])
        self.assertEqual(reranked[0].source, "reranker")

    def test_non_positive_top_k_returns_empty(self) -> None:
        self.assertEqual(rerank_results("问题", self.results, KeywordScorer(), top_k=0), [])

    def test_score_count_must_match_candidates(self) -> None:
        with self.assertRaises(ValueError):
            rerank_results("问题", self.results, BrokenScorer(), top_k=2)


if __name__ == "__main__":
    unittest.main()
