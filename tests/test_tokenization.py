from __future__ import annotations

import unittest

from annual_report_agent.retrieval.tokenization import tokenize


class TokenizationTests(unittest.TestCase):
    def test_chinese_unigrams_and_bigrams(self) -> None:
        tokens = tokenize("营业收入")
        self.assertIn("营", tokens)
        self.assertIn("营业", tokens)
        self.assertIn("收入", tokens)
        self.assertIn("营业收入", tokens)

    def test_english_and_numbers(self) -> None:
        self.assertEqual(tokenize("RAG 2025 16.12"), ["rag", "2025", "16.12"])


if __name__ == "__main__":
    unittest.main()
