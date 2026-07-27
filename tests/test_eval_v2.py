from __future__ import annotations

import hashlib
import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class EvalV2Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        eval_dir = ROOT / "data" / "eval"
        cls.manifest = json.loads((eval_dir / "eval_v2_manifest.json").read_text(encoding="utf-8"))
        cls.rows = {
            name: read_jsonl(ROOT / metadata["path"])
            for name, metadata in cls.manifest["files"].items()
        }

    def test_expected_split_sizes(self) -> None:
        self.assertEqual(len(self.rows["dev"]), 60)
        self.assertEqual(len(self.rows["test"]), 30)
        self.assertEqual(len(self.rows["no_answer"]), 15)

    def test_query_ids_and_text_are_unique(self) -> None:
        rows = [row for split in self.rows.values() for row in split]
        self.assertEqual(len({row["query_id"] for row in rows}), 105)
        self.assertEqual(len({"".join(row["query"].split()) for row in rows}), 105)

    def test_frozen_splits_match_manifest_hashes(self) -> None:
        for name in ("test", "no_answer"):
            metadata = self.manifest["files"][name]
            self.assertTrue(metadata["frozen"])
            self.assertEqual(sha256(ROOT / metadata["path"]), metadata["sha256"])

    def test_answerability_labels(self) -> None:
        for row in [*self.rows["dev"], *self.rows["test"]]:
            self.assertTrue(row["relevant_chunk_ids"])
            self.assertIn("answer", row)
        for row in self.rows["no_answer"]:
            self.assertEqual(row["relevant_chunk_ids"], [])
            self.assertEqual(row["expected_action"], "refuse")


if __name__ == "__main__":
    unittest.main()
