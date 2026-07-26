from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path

from .schemas import Chunk, EvaluationQuery


def write_chunks(path: Path, chunks: Iterable[Chunk]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for chunk in chunks:
            handle.write(json.dumps(chunk.to_dict(), ensure_ascii=False) + "\n")


def read_chunks(path: Path) -> list[Chunk]:
    with path.open("r", encoding="utf-8") as handle:
        return [Chunk.from_dict(json.loads(line)) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_evaluation_queries(path: Path) -> list[EvaluationQuery]:
    with path.open("r", encoding="utf-8") as handle:
        return [EvaluationQuery.from_dict(json.loads(line)) for line in handle if line.strip()]
