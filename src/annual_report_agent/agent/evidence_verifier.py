from __future__ import annotations

import re
from dataclasses import asdict, dataclass, replace
from typing import Any

from ..schemas import SearchResult
from .answerer import (
    AgentAnswer,
    Citation,
    FactSpec,
    infer_evidence_constraints,
    infer_fact_spec,
)
from .router import QueryRoute

QUARTER_QUERY_MARKERS = ("季度", "第一季度", "第二季度", "第三季度", "第四季度", "Q1", "Q2", "Q3", "Q4")
QUARTER_EVIDENCE_MARKERS = (
    "分季度主要财务指标",
    "第一季度",
    "第二季度",
    "第三季度",
    "第四季度",
)
ANNUAL_EVIDENCE_MARKERS = (
    "主要会计数据和财务指标",
    "合并利润表",
    "合并现金流量表",
    "本年发生额",
)


@dataclass(frozen=True, slots=True)
class EvidenceAssessment:
    status: str
    reasons: tuple[str, ...]
    confidence: float
    selected_chunk_ids: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class EvidenceVerifier:
    """Deterministic evidence sufficiency checks used by retrieval policies.

    The verifier never reads evaluation labels. It only sees the user query,
    query route, retrieved evidence and the answer produced from that evidence.
    """

    def __init__(self, *, refusal_confirmation_rounds: int = 2) -> None:
        if refusal_confirmation_rounds < 1:
            raise ValueError("refusal_confirmation_rounds must be positive")
        self.refusal_confirmation_rounds = refusal_confirmation_rounds

    @staticmethod
    def _selected_results(
        results: list[SearchResult],
        answer: AgentAnswer,
    ) -> list[SearchResult]:
        selected_ids = {citation.chunk_id for citation in answer.citations}
        return [result for result in results if result.chunk.chunk_id in selected_ids]

    @staticmethod
    def _expects_annual_evidence(query: str, route: QueryRoute) -> bool:
        return bool(route.years) and not any(marker in query for marker in QUARTER_QUERY_MARKERS)

    @staticmethod
    def _is_quarter_only(result: SearchResult) -> bool:
        text = f"{result.chunk.section} {result.chunk.text}"
        has_quarter = any(marker in text for marker in QUARTER_EVIDENCE_MARKERS)
        has_annual = any(marker in text for marker in ANNUAL_EVIDENCE_MARKERS)
        return has_quarter and not has_annual

    @staticmethod
    def _citation_is_quarter_only(
        result: SearchResult,
        citation: Citation,
        spec: FactSpec,
    ) -> bool:
        normalized = re.sub(
            r"\s*([,.%/（）()])\s*",
            r"\1",
            result.chunk.text,
        )
        normalized = re.sub(r"\s+", " ", normalized)
        quote_start = normalized.find(citation.quote)
        label_match = re.search(spec.label_pattern, citation.quote)
        if quote_start < 0 or label_match is None:
            context = f"{result.chunk.section} {citation.quote}"
            quarter_position = max(
                (context.rfind(marker) for marker in QUARTER_EVIDENCE_MARKERS),
                default=-1,
            )
            annual_position = max(
                (context.rfind(marker) for marker in ANNUAL_EVIDENCE_MARKERS),
                default=-1,
            )
            return quarter_position > annual_position

        label_position = quote_start + label_match.start()
        evidence_prefix = normalized[:label_position]
        quarter_position = max(
            (evidence_prefix.rfind(marker) for marker in QUARTER_EVIDENCE_MARKERS),
            default=-1,
        )
        annual_position = max(
            (evidence_prefix.rfind(marker) for marker in ANNUAL_EVIDENCE_MARKERS),
            default=-1,
        )
        return quarter_position > annual_position

    def filter_incompatible_evidence(
        self,
        query: str,
        route: QueryRoute,
        results: list[SearchResult],
    ) -> list[SearchResult]:
        """Remove evidence already proven to use the wrong time granularity.

        This is intentionally called only after a prior verifier failure, so the
        first round remains identical to the frozen Static RAG baseline.
        """
        if not self._expects_annual_evidence(query, route):
            return results
        return [result for result in results if not self._is_quarter_only(result)]

    def prepare_escalated_evidence(
        self,
        query: str,
        route: QueryRoute,
        results: list[SearchResult],
    ) -> list[SearchResult]:
        """Filter wrong-granularity evidence and repair PDF line-break artifacts."""
        prepared = self.filter_incompatible_evidence(query, route, results)
        normalized = []
        for result in prepared:
            text = re.sub(
                r"(?<=[\u4e00-\u9fff])\s+(?=[\u4e00-\u9fff])",
                "",
                result.chunk.text,
            )
            text = text.replace(
                "归属于母公司股东的净利润",
                "归属于上市公司股东的净利润",
            )
            normalized.append(replace(result, chunk=replace(result.chunk, text=text)))
        return normalized

    def assess(
        self,
        query: str,
        route: QueryRoute,
        results: list[SearchResult],
        answer: AgentAnswer,
        *,
        round_index: int,
    ) -> EvidenceAssessment:
        if route.should_refuse:
            return EvidenceAssessment(
                status="terminal_refusal",
                reasons=(route.refusal_reason or "out_of_scope",),
                confidence=1.0,
                selected_chunk_ids=(),
            )

        spec = infer_fact_spec(query)
        if answer.status == "refused":
            terminal_reasons = {"unsupported_fact", "incompatible_units"}
            if answer.reason in terminal_reasons:
                return EvidenceAssessment(
                    status="terminal_refusal",
                    reasons=(answer.reason or "unsupported",),
                    confidence=1.0,
                    selected_chunk_ids=(),
                )
            constraints = infer_evidence_constraints(query, spec) if spec is not None else ()
            if constraints and round_index >= self.refusal_confirmation_rounds:
                return EvidenceAssessment(
                    status="terminal_refusal",
                    reasons=("constraint_not_disclosed_after_confirmation",),
                    confidence=0.9,
                    selected_chunk_ids=(),
                )
            return EvidenceAssessment(
                status="insufficient",
                reasons=(answer.reason or "answer_refused",),
                confidence=0.2,
                selected_chunk_ids=(),
            )

        selected = self._selected_results(results, answer)
        selected_ids = tuple(citation.chunk_id for citation in answer.citations)
        reasons: list[str] = []
        if not selected:
            reasons.append("missing_citation_evidence")

        selected_companies = {result.chunk.company for result in selected}
        selected_years = {result.chunk.year for result in selected}
        missing_companies = set(route.companies) - selected_companies
        missing_years = set(route.years) - selected_years
        if missing_companies:
            reasons.append("missing_company_coverage")
        if missing_years:
            reasons.append("missing_year_coverage")

        if self._expects_annual_evidence(query, route) and spec is not None:
            selected_by_id = {result.chunk.chunk_id: result for result in selected}
            for citation in answer.citations:
                result = selected_by_id.get(citation.chunk_id)
                if result is None:
                    continue
                if self._citation_is_quarter_only(result, citation, spec):
                    reasons.append("annual_quarter_mismatch")
                    break

        if reasons:
            return EvidenceAssessment(
                status="insufficient",
                reasons=tuple(dict.fromkeys(reasons)),
                confidence=0.1,
                selected_chunk_ids=selected_ids,
            )
        return EvidenceAssessment(
            status="sufficient",
            reasons=("answer_supported_by_complete_evidence",),
            confidence=1.0,
            selected_chunk_ids=selected_ids,
        )
