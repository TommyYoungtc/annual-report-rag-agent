from annual_report_agent.agent import (
    EvidenceAssessment,
    PolicyConfig,
    RetrievalPolicy,
    decide_next_action,
    merge_ranked_results,
)
from annual_report_agent.schemas import Chunk, SearchResult


def assessment(status: str) -> EvidenceAssessment:
    return EvidenceAssessment(status, ("test",), 1.0, ())


def search_result(chunk_id: str, score: float) -> SearchResult:
    return SearchResult(
        chunk=Chunk(chunk_id, "doc", "公司", 2025, "section", 1, "text"),
        score=score,
        rank=1,
        source="reranker",
    )


def test_static_stops_after_first_round() -> None:
    config = PolicyConfig("A", RetrievalPolicy.STATIC, 1)
    decision = decide_next_action(config, assessment("insufficient"), round_index=1)
    assert decision.action == "stop"
    assert decision.reason == "round_budget_exhausted"


def test_always_agentic_escalates_even_when_sufficient() -> None:
    config = PolicyConfig("B", RetrievalPolicy.ALWAYS_AGENTIC, 3)
    decision = decide_next_action(config, assessment("sufficient"), round_index=1)
    assert decision.action == "retrieve_more"


def test_adaptive_stops_on_sufficient_and_escalates_on_insufficient() -> None:
    config = PolicyConfig("C", RetrievalPolicy.ADAPTIVE, 3)
    assert decide_next_action(config, assessment("sufficient"), round_index=1).action == "stop"
    assert (
        decide_next_action(config, assessment("insufficient"), round_index=1).action
        == "retrieve_more"
    )


def test_merge_keeps_best_score_and_reassigns_ranks() -> None:
    merged = merge_ranked_results(
        [search_result("doc:1", 0.4), search_result("doc:2", 0.8)],
        [search_result("doc:1", 0.9)],
    )
    assert [result.chunk.chunk_id for result in merged] == ["doc:1", "doc:2"]
    assert [result.rank for result in merged] == [1, 2]
