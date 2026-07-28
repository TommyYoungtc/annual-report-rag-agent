from __future__ import annotations

import re
from collections.abc import Sequence

from ..agent import extract_value, infer_fact_spec
from ..schemas import Chunk, SearchResult

CHUNK_INDEX_PATTERN = re.compile(r"^(?P<document>.+):(?P<index>\d+)$")


def split_training_queries(
    rows: Sequence[dict],
    *,
    validation_stride: int = 5,
) -> tuple[list[dict], list[dict]]:
    if validation_stride < 2:
        raise ValueError("validation_stride must be at least 2")
    train = []
    validation = []
    for index, row in enumerate(rows, start=1):
        target = validation if index % validation_stride == 0 else train
        target.append(row)
    return train, validation


def is_adjacent_to_gold(candidate_id: str, gold_ids: Sequence[str], *, distance: int = 1) -> bool:
    candidate_match = CHUNK_INDEX_PATTERN.match(candidate_id)
    if candidate_match is None:
        return False
    candidate_document = candidate_match.group("document")
    candidate_index = int(candidate_match.group("index"))
    for gold_id in gold_ids:
        gold_match = CHUNK_INDEX_PATTERN.match(gold_id)
        if gold_match is None or gold_match.group("document") != candidate_document:
            continue
        if abs(candidate_index - int(gold_match.group("index"))) <= distance:
            return True
    return False


def _normalize_value(value: str) -> str:
    return re.sub(r"[\s（）()，,；;]", "", value)


def candidate_contains_labeled_answer(row: dict, chunk: Chunk) -> bool:
    spec = infer_fact_spec(str(row["query"]))
    if spec is None:
        return False
    result = SearchResult(chunk=chunk, score=0.0, rank=1, source="negative_guard")
    extracted = extract_value(result, spec)
    if extracted is None:
        return False
    return _normalize_value(extracted.value_text) in _normalize_value(str(row["answer"]))
