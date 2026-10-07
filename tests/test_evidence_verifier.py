from annual_report_agent.agent import (
    CorpusScope,
    EvidenceVerifier,
    answer_from_evidence,
    route_query,
)
from annual_report_agent.schemas import Chunk, SearchResult


def result(chunk_id: str, section: str, text: str, score: float = 0.9) -> SearchResult:
    return SearchResult(
        chunk=Chunk(
            chunk_id=chunk_id,
            document_id="luxshare-2025",
            company="立讯精密",
            year=2025,
            section=section,
            page=9,
            text=text,
        ),
        score=score,
        rank=1,
        source="reranker",
    )


def test_verifier_flags_quarterly_evidence_for_annual_question() -> None:
    scope = CorpusScope(("立讯精密",), (2025,))
    query = "立讯精密2025年归属于上市公司股东的净利润是多少？"
    route = route_query(query, scope)
    evidence = [
        result(
            "luxshare-2025:0009",
            "分季度主要财务指标",
            "立讯精密工业股份有限公司 2025 年年度报告全文 "
            "分季度主要财务指标 第一季度 "
            + "季度表头说明" * 12
            + " "
            "归属于上市公司股东的净利润 3,043,574,684.79",
        )
    ]
    answer = answer_from_evidence(query, route, evidence, scope)
    assessment = EvidenceVerifier().assess(
        query,
        route,
        evidence,
        answer,
        round_index=1,
    )
    assert answer.status == "answered"
    assert assessment.status == "insufficient"
    assert "annual_quarter_mismatch" in assessment.reasons


def test_verifier_filters_quarter_only_evidence_after_escalation() -> None:
    scope = CorpusScope(("立讯精密",), (2025,))
    query = "立讯精密2025年归属于上市公司股东的净利润是多少？"
    route = route_query(query, scope)
    quarterly = result(
        "luxshare-2025:0009",
        "分季度主要财务指标",
        "年度报告全文 分季度主要财务指标 第一季度 净利润 3,043,574,684.79",
        0.99,
    )
    annual = result(
        "luxshare-2025:0007",
        "主要会计数据和财务指标",
        "主要会计数据和财务指标 净利润 16,599,769,785.64",
        0.90,
    )
    filtered = EvidenceVerifier().filter_incompatible_evidence(
        query,
        route,
        [quarterly, annual],
    )
    assert [item.chunk.chunk_id for item in filtered] == ["luxshare-2025:0007"]


def test_verifier_repairs_pdf_line_breaks_for_escalated_evidence() -> None:
    scope = CorpusScope(("立讯精密",), (2025,))
    query = "立讯精密2025年归属于上市公司股东的净利润是多少？"
    route = route_query(query, scope)
    annual = result(
        "luxshare-2025:0007",
        "主要会计数据和财务指标",
        "归属于上市公司股东的净利\n润（元） 16,599,769,785.64",
    )
    prepared = EvidenceVerifier().prepare_escalated_evidence(
        query,
        route,
        [annual],
    )
    answer = answer_from_evidence(query, route, prepared, scope)
    assert answer.status == "answered"
    assert answer.answer == "16,599,769,785.64元"


def test_verifier_accepts_complete_annual_evidence() -> None:
    scope = CorpusScope(("立讯精密",), (2025,))
    query = "立讯精密2025年归属于上市公司股东的净利润是多少？"
    route = route_query(query, scope)
    evidence = [
        result(
            "luxshare-2025:0007",
            "主要会计数据和财务指标",
            "主要会计数据和财务指标 归属于上市公司股东的净利润 16,599,769,785.64元",
        )
    ]
    answer = answer_from_evidence(query, route, evidence, scope)
    assessment = EvidenceVerifier().assess(
        query,
        route,
        evidence,
        answer,
        round_index=1,
    )
    assert answer.status == "answered"
    assert assessment.status == "sufficient"


def test_verifier_does_not_reject_annual_value_when_quarter_table_is_later() -> None:
    scope = CorpusScope(("立讯精密",), (2025,))
    query = "立讯精密2025年归属于上市公司股东的净利润是多少？"
    route = route_query(query, scope)
    evidence = [
        result(
            "luxshare-2025:0007",
            "文档开头",
            "归属于上市公司股东的净利润（元） 16,599,769,785.64 "
            + "年度经营情况" * 15
            + "分季度主要财务指标 第一季度 3,043,574,684.79",
        )
    ]
    answer = answer_from_evidence(query, route, evidence, scope)
    assessment = EvidenceVerifier().assess(
        query,
        route,
        evidence,
        answer,
        round_index=1,
    )
    assert answer.status == "answered"
    assert assessment.status == "sufficient"


def test_constraint_refusal_requires_one_confirmation_round() -> None:
    scope = CorpusScope(("立讯精密",), (2025,))
    query = "立讯精密2025年女性研发人员有多少？"
    route = route_query(query, scope)
    evidence = [
        result(
            "luxshare-2025:0080",
            "公司研发人员情况",
            "公司研发人员情况 研发人员数量 34,357人",
        )
    ]
    answer = answer_from_evidence(query, route, evidence, scope)
    verifier = EvidenceVerifier(refusal_confirmation_rounds=2)
    first = verifier.assess(query, route, evidence, answer, round_index=1)
    second = verifier.assess(query, route, evidence, answer, round_index=2)
    assert answer.status == "refused"
    assert first.status == "insufficient"
    assert second.status == "terminal_refusal"
    assert second.reasons == ("constraint_not_disclosed_after_confirmation",)
