from __future__ import annotations

import json
import unittest
from pathlib import Path
from urllib.parse import urlparse


class ManifestTests(unittest.TestCase):
    def test_manifest_has_two_years_for_three_companies(self) -> None:
        root = Path(__file__).resolve().parents[1]
        manifest = root / "data" / "raw" / "manifest.jsonl"
        with manifest.open("r", encoding="utf-8") as handle:
            rows = [json.loads(line) for line in handle if line.strip()]

        self.assertEqual(len(rows), 6)
        self.assertEqual(len({row["document_id"] for row in rows}), 6)
        by_company = {}
        for row in rows:
            by_company.setdefault(row["company"], set()).add(int(row["year"]))
            parsed = urlparse(row["url"])
            self.assertEqual(parsed.scheme, "https")
            self.assertTrue(parsed.netloc)
            self.assertTrue(row["filename"].endswith(".pdf"))

        self.assertEqual(
            by_company,
            {
                "宁德时代": {2024, 2025},
                "科大讯飞": {2024, 2025},
                "比亚迪": {2024, 2025},
            },
        )


if __name__ == "__main__":
    unittest.main()
