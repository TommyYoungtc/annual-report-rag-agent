from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class RoundCost:
    retrieval_ms: float = 0.0
    rerank_ms: float = 0.0
    amortized_embedding_load_ms: float = 0.0
    amortized_reranker_load_ms: float = 0.0
    dense_queries: int = 0
    bm25_queries: int = 0
    reranker_calls: int = 0
    candidates_retrieved: int = 0
    passages_reranked: int = 0
    planner_input_tokens: int = 0
    planner_output_tokens: int = 0
    estimated_api_cost_usd: float = 0.0

    @property
    def amortized_wall_clock_ms(self) -> float:
        return (
            self.retrieval_ms
            + self.rerank_ms
            + self.amortized_embedding_load_ms
            + self.amortized_reranker_load_ms
        )

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["amortized_wall_clock_ms"] = self.amortized_wall_clock_ms
        return value


@dataclass(frozen=True, slots=True)
class RetrievalStep:
    round_index: int
    search_query: str
    evidence_chunk_ids: tuple[str, ...]
    evidence_scores: tuple[float, ...]
    answer_status: str
    answer_reason: str | None
    assessment_status: str
    assessment_reasons: tuple[str, ...]
    action: str
    action_reason: str
    cost: RoundCost

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["cost"] = self.cost.to_dict()
        return value


@dataclass(frozen=True, slots=True)
class TrajectoryUsage:
    retrieval_rounds: int
    dense_queries: int
    bm25_queries: int
    reranker_calls: int
    candidates_retrieved: int
    passages_reranked: int
    planner_input_tokens: int
    planner_output_tokens: int
    estimated_api_cost_usd: float
    amortized_wall_clock_ms: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def summarize_usage(steps: list[RetrievalStep]) -> TrajectoryUsage:
    costs = [step.cost for step in steps]
    return TrajectoryUsage(
        retrieval_rounds=len(steps),
        dense_queries=sum(cost.dense_queries for cost in costs),
        bm25_queries=sum(cost.bm25_queries for cost in costs),
        reranker_calls=sum(cost.reranker_calls for cost in costs),
        candidates_retrieved=sum(cost.candidates_retrieved for cost in costs),
        passages_reranked=sum(cost.passages_reranked for cost in costs),
        planner_input_tokens=sum(cost.planner_input_tokens for cost in costs),
        planner_output_tokens=sum(cost.planner_output_tokens for cost in costs),
        estimated_api_cost_usd=sum(cost.estimated_api_cost_usd for cost in costs),
        amortized_wall_clock_ms=sum(cost.amortized_wall_clock_ms for cost in costs),
    )
