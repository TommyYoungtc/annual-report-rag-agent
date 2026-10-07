from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from ..schemas import SearchResult
from .evidence_verifier import EvidenceAssessment


class RetrievalPolicy(str, Enum):
    STATIC = "static_rag"
    ALWAYS_AGENTIC = "always_agentic"
    ADAPTIVE = "adaptive_budget_aware"


@dataclass(frozen=True, slots=True)
class PolicyConfig:
    name: str
    policy: RetrievalPolicy
    max_rounds: int

    def __post_init__(self) -> None:
        if self.max_rounds < 1:
            raise ValueError("max_rounds must be positive")


@dataclass(frozen=True, slots=True)
class ControllerDecision:
    action: str
    reason: str


def decide_next_action(
    config: PolicyConfig,
    assessment: EvidenceAssessment,
    *,
    round_index: int,
) -> ControllerDecision:
    if assessment.status == "terminal_refusal":
        return ControllerDecision("stop", "terminal_refusal")
    if round_index >= config.max_rounds:
        return ControllerDecision("stop", "round_budget_exhausted")
    if config.policy == RetrievalPolicy.STATIC:
        return ControllerDecision("stop", "static_single_round_policy")
    if config.policy == RetrievalPolicy.ALWAYS_AGENTIC:
        return ControllerDecision("retrieve_more", "always_agentic_forced_escalation")
    if assessment.status == "insufficient":
        return ControllerDecision("retrieve_more", "evidence_insufficient")
    return ControllerDecision("stop", "adaptive_evidence_sufficient")


def merge_ranked_results(
    current: list[SearchResult],
    new: list[SearchResult],
) -> list[SearchResult]:
    best: dict[str, SearchResult] = {}
    for result in [*current, *new]:
        existing = best.get(result.chunk.chunk_id)
        if existing is None or result.score > existing.score:
            best[result.chunk.chunk_id] = result
    ranked = sorted(best.values(), key=lambda result: result.score, reverse=True)
    return [
        SearchResult(
            chunk=result.chunk,
            score=result.score,
            rank=rank,
            source=result.source,
        )
        for rank, result in enumerate(ranked, start=1)
    ]
