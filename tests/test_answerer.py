from __future__ import annotations

from decimal import Decimal

from annual_report_agent.agent import (
    CorpusScope,
    answer_from_evidence,
    calculate_change,
    route_query,
)
from annual_report_agent.schemas import Chunk, SearchResult

SCOPE = CorpusScope(("宁德时代", "比亚迪", "科大讯飞"), (2024, 2025))


def result(
    text: str,
    *,
    company: str = "宁德时代",
    year: int = 2024,
    page: int | None = 9,
    score: float = 0.9,
) -> SearchResult:
    chunk = Chunk(
        chunk_id=f"{company}-{year}:0001",
        document_id=f"{company}-{year}",
        company=company,
        year=year,
        section="主要财务指标",
        page=page,
        text=text,
    )
    return SearchResult(chunk=chunk, score=score, rank=1, source="reranker")


def test_answers_single_fact_with_citation() -> None:
    query = "宁德时代2024年的营业收入是多少？"
    route = route_query(query, SCOPE)
    answer = answer_from_evidence(
        query,
        route,
        [result("单位：千元\n营业收入 362,012,554 400,917,045 -9.70%")],
        SCOPE,
    )
    assert answer.status == "answered"
    assert answer.answer == "362,012,554千元"
    assert answer.citations[0].page == 9
    assert "362,012,554" in answer.citations[0].quote


def test_answers_basic_eps_with_per_share_unit() -> None:
    query = "宁德时代2024年的基本每股收益是多少？"
    route = route_query(query, SCOPE)
    answer = answer_from_evidence(
        query,
        route,
        [result("基本每股收益（元/股） 11.58 10.06 15.11%")],
        SCOPE,
    )
    assert answer.answer == "11.58元/股"


def test_answers_weighted_average_roe() -> None:
    query = "宁德时代2024年的加权平均净资产收益率是多少？"
    route = route_query(query, SCOPE)
    answer = answer_from_evidence(
        query,
        route,
        [result("加权平均净资产收益\n率 24.13% 24.04% 0.09%")],
        SCOPE,
    )
    assert answer.answer == "24.13%"


def test_answers_total_assets_for_both_common_labels() -> None:
    cases = (
        ("宁德时代", "资产总额（千元） 786,658,123", "786,658,123千元"),
        ("科大讯飞", "总资产（元） 41,478,899,803.20", "41,478,899,803.20元"),
    )
    for company, evidence, expected in cases:
        query = f"{company}2024年末的资产总额是多少？"
        route = route_query(query, SCOPE)
        answer = answer_from_evidence(
            query,
            route,
            [result(evidence, company=company)],
            SCOPE,
        )
        assert answer.answer == expected


def test_answers_rnd_staff_without_confusing_ratio() -> None:
    query = "比亚迪2024年有多少研发人员？"
    route = route_query(query, SCOPE)
    answer = answer_from_evidence(
        query,
        route,
        [result("研发人员数量（人） 121,598 102,844 18.24%", company="比亚迪")],
        SCOPE,
    )
    assert answer.answer == "121,598人"


def test_answers_combined_rnd_and_technical_staff_label() -> None:
    query = "海康威视2025年有多少研发人员？"
    scope = CorpusScope(companies=("海康威视",), years=(2025,))
    route = route_query(query, scope)
    answer = answer_from_evidence(
        query,
        route,
        [
            result(
                "研发及技术人员数量（人） 26,806 28,272 -5.19%",
                company="海康威视",
                year=2025,
            )
        ],
        scope,
    )
    assert answer.answer == "26,806人"


def test_answers_rnd_expense_ratio_synonym() -> None:
    query = "美的集团2025年研发投入占营业收入的比例是多少？"
    scope = CorpusScope(companies=("美的集团",), years=(2025,))
    route = route_query(query, scope)
    answer = answer_from_evidence(
        query,
        route,
        [
            result(
                "公司研发投入情况 研发费用占营业收入比例 3.90% 3.99%",
                company="美的集团",
                year=2025,
            )
        ],
        scope,
    )
    assert answer.answer == "3.90%"


def test_handles_wrapped_table_label() -> None:
    query = "科大讯飞2025年归属于上市公司股东的净利润是多少？"
    route = route_query(query, SCOPE)
    answer = answer_from_evidence(
        query,
        route,
        [
            result(
                "归属于上市公司股东\n的净利润（元） 839,390,861.36 560,162,663.16",
                company="科大讯飞",
                year=2025,
            )
        ],
        SCOPE,
    )
    assert answer.answer == "839,390,861.36元"


def test_rnd_amount_does_not_capture_year_from_heading() -> None:
    query = "宁德时代2024年的研发投入金额是多少？"
    route = route_query(query, SCOPE)
    answer = answer_from_evidence(
        query,
        route,
        [
            result(
                "近三年公司研发投入金额及占营业收入的比例 "
                "项目 2024年 2023年 研发投入金额（千元）18,606,756 18,356,108"
            )
        ],
        SCOPE,
    )
    assert answer.answer == "18,606,756千元"


def test_inherits_page_level_unit_for_wrapped_label() -> None:
    query = "宁德时代2025年末归属于上市公司股东的净资产是多少？"
    route = route_query(query, SCOPE)
    answer = answer_from_evidence(
        query,
        route,
        [
            result(
                "单位：千元 "
                + "其他表格内容 " * 30
                + "归属于上市公司股东\n的净资产 337,107,747 246,930,033",
                year=2025,
            )
        ],
        SCOPE,
    )
    assert answer.answer == "337,107,747千元"


def test_answers_cross_year_and_calculates_change() -> None:
    query = "比亚迪2024到2025年营业收入增长了多少？"
    route = route_query(query, SCOPE)
    answer = answer_from_evidence(
        query,
        route,
        [
            result("营业收入（元） 777,102,455,000.00", company="比亚迪", year=2024),
            result("营业收入（元） 803,964,958,000.00", company="比亚迪", year=2025),
        ],
        SCOPE,
    )
    assert answer.status == "answered"
    assert answer.answer is not None
    assert "26,862,503,000.00元" in answer.answer
    assert "3.46%" in answer.answer
    assert len(answer.citations) == 2


def test_refuses_when_one_year_is_missing() -> None:
    query = "宁德时代2024年和2025年的净利润分别是多少？"
    route = route_query(query, SCOPE)
    answer = answer_from_evidence(
        query,
        route,
        [result("归属于上市公司股东的净利润（千元）50,744,682")],
        SCOPE,
    )
    assert answer.status == "refused"
    assert answer.reason == "insufficient_evidence"


def test_refuses_low_reranker_score() -> None:
    query = "宁德时代2024年的营业收入是多少？"
    route = route_query(query, SCOPE)
    answer = answer_from_evidence(
        query,
        route,
        [result("营业收入（千元）362,012,554", score=0.1)],
        SCOPE,
        minimum_reranker_score=0.5,
    )
    assert answer.status == "refused"


def test_prefers_annual_summary_over_higher_reranker_score() -> None:
    query = "科大讯飞2024年归属于上市公司股东的净利润是多少？"
    route = route_query(query, SCOPE)
    wrong = result(
        "八、分季度主要财务指标 归属于上市公司股东的净利润（元）-300,468,030.20",
        company="科大讯飞",
        page=8,
        score=0.99,
    )
    gold = result(
        "六、主要会计数据和财务指标 归属于上市公司股东的净利润（元）560,162,663.16",
        company="科大讯飞",
        page=7,
        score=0.90,
    )
    answer = answer_from_evidence(query, route, [wrong, gold], SCOPE)
    assert answer.answer == "560,162,663.16元"
    assert answer.citations[0].page == 7


def test_refuses_total_rnd_when_query_requires_overseas_breakdown() -> None:
    query = "宁德时代2025年的海外研发投入金额是多少？"
    route = route_query(query, SCOPE)
    evidence = result(
        "近三年公司研发投入金额 项目2025年 研发投入金额（千元）22,146,581",
        year=2025,
    )
    baseline = answer_from_evidence(
        query,
        route,
        [evidence],
        SCOPE,
        enforce_query_constraints=False,
    )
    constrained = answer_from_evidence(query, route, [evidence], SCOPE)
    assert baseline.status == "answered"
    assert constrained.status == "refused"
    assert constrained.reason == "insufficient_evidence"


def test_accepts_value_when_required_qualifier_is_entailed() -> None:
    query = "宁德时代2025年的境外研发投入金额是多少？"
    route = route_query(query, SCOPE)
    answer = answer_from_evidence(
        query,
        route,
        [result("境外研发投入金额（千元）1,234", year=2025)],
        SCOPE,
    )
    assert answer.answer == "1,234千元"


def test_refuses_total_technical_staff_for_intersection_query() -> None:
    query = "比亚迪2024年女性技术人员有多少？"
    route = route_query(query, SCOPE)
    answer = answer_from_evidence(
        query,
        route,
        [result("技术人员 122,924 女性员工 20,000", company="比亚迪")],
        SCOPE,
    )
    assert answer.status == "refused"


def test_refuses_total_rnd_staff_for_intersection_query() -> None:
    query = "美的集团2025年女性研发人员数量是多少？"
    scope = CorpusScope(companies=("美的集团",), years=(2025,))
    route = route_query(query, scope)
    evidence = result(
        "研发人员数量（人） 23,926 研发人员数量占比 12.05%",
        company="美的集团",
        year=2025,
    )
    baseline = answer_from_evidence(
        query,
        route,
        [evidence],
        scope,
        enforce_query_constraints=False,
    )
    constrained = answer_from_evidence(query, route, [evidence], scope)
    assert baseline.answer == "23,926人"
    assert constrained.status == "refused"


def test_refuses_company_profit_for_segment_profit_query() -> None:
    query = "科大讯飞2025年智慧教育业务的净利润是多少？"
    route = route_query(query, SCOPE)
    answer = answer_from_evidence(
        query,
        route,
        [
            result(
                "智慧教育业务保持增长。主要会计数据和财务指标 "
                "归属于上市公司股东的净利润（元）839,390,861.36",
                company="科大讯飞",
                year=2025,
            )
        ],
        SCOPE,
    )
    assert answer.status == "refused"


def test_preserves_dividend_tax_note() -> None:
    query = "比亚迪2024年度每10股派发多少现金红利？"
    route = route_query(query, SCOPE)
    answer = answer_from_evidence(
        query,
        route,
        [
            result(
                "向全体股东每10股派发现金红利39.74元（含税），送红股0股。",
                company="比亚迪",
            )
        ],
        SCOPE,
    )
    assert answer.answer == "39.74元（含税）"


def test_calculator_handles_zero_baseline() -> None:
    change = calculate_change(Decimal(0), Decimal(10))
    assert change.absolute_change == Decimal(10)
    assert change.percentage_change is None
