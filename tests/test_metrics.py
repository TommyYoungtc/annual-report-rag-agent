from __future__ import annotations

import unittest

from annual_report_agent.evaluation import evaluate_retrieval, ndcg_at_k, recall_at_k


class RetrievalMetricsTests(unittest.TestCase):
    def test_recall(self) -> None:
        self.assertEqual(recall_at_k(["a", "b"], ["a", "c"], 2), 0.5)

    def test_ndcg_perfect_ranking(self) -> None:
        self.assertAlmostEqual(ndcg_at_k(["a", "b"], ["a", "b"], 2), 1.0)

    def test_aggregate_metrics(self) -> None:
        metrics = evaluate_retrieval(
            [(["a", "x"], ["a"]), (["x", "b"], ["b"])],
            recall_ks=(1, 2),
            ndcg_k=2,
        )
        self.assertEqual(metrics["recall@1"], 0.5)
        self.assertEqual(metrics["recall@2"], 1.0)
        self.assertAlmostEqual(metrics["mrr"], 0.75)


if __name__ == "__main__":
    unittest.main()
