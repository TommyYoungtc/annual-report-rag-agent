from __future__ import annotations

import unittest

from annual_report_agent.ingestion import MarkdownDocument, chunk_markdown


class MarkdownChunkerTests(unittest.TestCase):
    def test_preserves_page_and_section(self) -> None:
        document = MarkdownDocument(
            document_id="demo-2024",
            company="示例公司",
            year=2024,
            markdown="""<!-- page: 3 -->
# 研发投入
这是一个足够长的研发投入说明，用于确认章节和页码能够被正确保留下来。
""",
        )
        chunks = chunk_markdown(document, max_chars=100, overlap_chars=10, min_chars=10)
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].page, 3)
        self.assertEqual(chunks[0].section, "研发投入")
        self.assertEqual(chunks[0].chunk_id, "demo-2024:0000")

    def test_rejects_invalid_overlap(self) -> None:
        document = MarkdownDocument("d", "c", 2024, "text")
        with self.assertRaises(ValueError):
            chunk_markdown(document, max_chars=100, overlap_chars=100)


if __name__ == "__main__":
    unittest.main()
