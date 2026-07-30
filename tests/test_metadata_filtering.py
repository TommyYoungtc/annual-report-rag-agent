from __future__ import annotations

import unittest

import numpy as np

from annual_report_agent.retrieval import (
    BM25Retriever,
    expand_query_with_section_anchors,
    infer_allowed_document_ids,
)
from annual_report_agent.retrieval.dense import DenseRetriever
from annual_report_agent.schemas import Chunk


class ConstantEncoder:
    def encode_documents(self, texts):
        return np.asarray([[1.0, 0.0] for _ in texts], dtype=np.float32)

    def encode_queries(self, texts):
        return np.asarray([[1.0, 0.0] for _ in texts], dtype=np.float32)


class MetadataFilteringTests(unittest.TestCase):
    def setUp(self) -> None:
        self.chunks = [
            Chunk("a", "catl-2024", "宁德时代", 2024, "财务", 1, "营业收入"),
            Chunk("b", "catl-2025", "宁德时代", 2025, "财务", 1, "营业收入"),
            Chunk("c", "byd-2024", "比亚迪", 2024, "财务", 1, "营业收入"),
        ]

    def test_infers_company_and_year(self) -> None:
        allowed = infer_allowed_document_ids("宁德时代2025年的营业收入是多少？", self.chunks)
        self.assertEqual(allowed, {"catl-2025"})

    def test_supports_cross_company_queries(self) -> None:
        allowed = infer_allowed_document_ids("比较宁德时代和比亚迪2024年的营业收入", self.chunks)
        self.assertEqual(allowed, {"catl-2024", "byd-2024"})

    def test_returns_none_without_metadata(self) -> None:
        self.assertIsNone(infer_allowed_document_ids("研发投入情况", self.chunks))

    def test_expands_financial_query_with_section_anchor(self) -> None:
        expanded = expand_query_with_section_anchors("2024年的营业收入是多少？")
        self.assertIn("主要会计数据", expanded)
        self.assertIn("财务指标", expanded)

    def test_expands_rnd_staff_query_with_exact_table_label(self) -> None:
        expanded = expand_query_with_section_anchors("2024年有多少研发人员？")
        self.assertIn("公司研发人员情况", expanded)
        self.assertIn("研发人员数量", expanded)

    def test_leaves_unknown_query_unchanged(self) -> None:
        query = "公司的核心竞争力是什么？"
        self.assertEqual(expand_query_with_section_anchors(query), query)

    def test_bm25_respects_allowed_documents(self) -> None:
        results = BM25Retriever(self.chunks).search(
            "营业收入", top_k=3, allowed_document_ids={"byd-2024"}
        )
        self.assertEqual([result.chunk.chunk_id for result in results], ["c"])

    def test_dense_respects_allowed_documents(self) -> None:
        results = DenseRetriever(self.chunks, ConstantEncoder()).search(
            "营业收入", top_k=3, allowed_document_ids={"catl-2025"}
        )
        self.assertEqual([result.chunk.chunk_id for result in results], ["b"])

    def test_dense_can_use_precomputed_embeddings(self) -> None:
        embeddings = np.asarray([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]], dtype=np.float32)
        results = DenseRetriever(
            self.chunks,
            ConstantEncoder(),
            document_embeddings=embeddings,
        ).search("营业收入", top_k=1)
        self.assertEqual(results[0].chunk.chunk_id, "a")


if __name__ == "__main__":
    unittest.main()
