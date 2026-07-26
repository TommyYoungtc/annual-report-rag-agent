from __future__ import annotations

import unittest

from annual_report_agent.retrieval import reciprocal_rank_fusion
from annual_report_agent.schemas import Chunk, SearchResult


class FusionTests(unittest.TestCase):
    def test_chunk_in_both_lists_wins(self) -> None:
        a = Chunk("a", "d", "公司", 2024, "章节", 1, "A")
        b = Chunk("b", "d", "公司", 2024, "章节", 2, "B")
        dense = [
            SearchResult(a, 0.9, 1, "dense"),
            SearchResult(b, 0.8, 2, "dense"),
        ]
        bm25 = [SearchResult(b, 4.0, 1, "bm25")]
        fused = reciprocal_rank_fusion([dense, bm25], rank_constant=60, top_k=2)
        self.assertEqual(fused[0].chunk.chunk_id, "b")
        self.assertEqual(fused[0].source, "bm25+dense")

    def test_weights_can_prioritize_sparse_ranking(self) -> None:
        a = Chunk("a", "d", "公司", 2024, "章节", 1, "A")
        b = Chunk("b", "d", "公司", 2024, "章节", 2, "B")
        dense = [SearchResult(a, 0.9, 1, "dense")]
        bm25 = [SearchResult(b, 4.0, 1, "bm25")]
        fused = reciprocal_rank_fusion(
            [dense, bm25],
            rank_constant=60,
            top_k=2,
            weights=[1.0, 2.0],
        )
        self.assertEqual(fused[0].chunk.chunk_id, "b")

    def test_weights_length_must_match_rankings(self) -> None:
        with self.assertRaises(ValueError):
            reciprocal_rank_fusion([[]], weights=[1.0, 2.0])


if __name__ == "__main__":
    unittest.main()
