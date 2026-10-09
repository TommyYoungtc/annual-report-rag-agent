from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ModelDecision:
    model_tier: str
    reason: str
    complexity_score: int


def choose_model(
    query: str,
    *,
    policy: str,
    retrieval_rounds: int = 1,
    evidence_count: int = 0,
) -> ModelDecision:
    if policy == "always_large":
        return ModelDecision("large", "baseline_always_large", 0)
    if policy != "adaptive":
        raise ValueError(f"unsupported model routing policy: {policy}")

    score = 0
    reasons: list[str] = []
    years = set(re.findall(r"20\d{2}", query))
    if len(years) >= 2 or any(term in query for term in ("跨年", "分别", "相比", "变化")):
        score += 2
        reasons.append("multi_period")
    if any(term in query for term in ("计算", "变化率", "增长率", "减少", "增加")):
        score += 2
        reasons.append("calculation")
    if any(term in query for term in ("季度", "半年度", "年度", "全年")):
        score += 1
        reasons.append("time_granularity")
    if retrieval_rounds > 1:
        score += 2
        reasons.append("retrieval_escalated")
    if evidence_count > 10:
        score += 1
        reasons.append("many_evidence_chunks")
    if score >= 2:
        return ModelDecision("large", "+".join(reasons), score)
    return ModelDecision("small", "single_fact_sufficient_evidence", score)
