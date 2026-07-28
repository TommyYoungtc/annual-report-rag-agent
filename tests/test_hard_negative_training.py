from __future__ import annotations

from annual_report_agent.schemas import Chunk
from annual_report_agent.training import (
    candidate_contains_labeled_answer,
    is_adjacent_to_gold,
    split_training_queries,
)


def chunk(chunk_id: str, text: str) -> Chunk:
    document_id = chunk_id.split(":", 1)[0]
    return Chunk(
        chunk_id=chunk_id,
        document_id=document_id,
        company="宁德时代",
        year=2024,
        section="测试",
        page=9,
        text=text,
    )


def test_split_reserves_every_fifth_query() -> None:
    rows = [{"query_id": f"q{index}"} for index in range(1, 11)]
    train, validation = split_training_queries(rows)
    assert [row["query_id"] for row in validation] == ["q5", "q10"]
    assert len(train) == 8


def test_rejects_invalid_validation_stride() -> None:
    try:
        split_training_queries([], validation_stride=1)
    except ValueError as error:
        assert "at least 2" in str(error)
    else:
        raise AssertionError("expected ValueError")


def test_detects_adjacent_chunks_only_in_same_document() -> None:
    assert is_adjacent_to_gold("catl-2024:0010", ["catl-2024:0009"])
    assert not is_adjacent_to_gold("catl-2025:0010", ["catl-2024:0009"])
    assert not is_adjacent_to_gold("catl-2024:0012", ["catl-2024:0009"])


def test_filters_unlabeled_duplicate_answer_evidence() -> None:
    row = {
        "query": "宁德时代2024年的营业收入是多少？",
        "answer": "362,012,554千元",
    }
    duplicate = chunk(
        "catl-2024:0030",
        "主要会计数据和财务指标 单位：千元 营业收入 362,012,554",
    )
    unrelated = chunk("catl-2024:0031", "营业收入相关会计政策，不包含年度数值。")
    assert candidate_contains_labeled_answer(row, duplicate)
    assert not candidate_contains_labeled_answer(row, unrelated)
