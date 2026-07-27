from __future__ import annotations

import pytest

from annual_report_agent.agent import CorpusScope, refusal_message, route_query

SCOPE = CorpusScope(("宁德时代", "比亚迪", "科大讯飞"), (2024, 2025))


def test_routes_covered_single_fact() -> None:
    route = route_query("宁德时代2025年的营业收入是多少？", SCOPE)
    assert route.companies == ("宁德时代",)
    assert route.years == (2025,)
    assert route.task_type == "single_fact"
    assert not route.should_refuse


def test_refuses_uncovered_period() -> None:
    route = route_query("宁德时代2026年的营业收入是多少？", SCOPE)
    assert route.should_refuse
    assert route.refusal_reason == "period_not_covered"


def test_refuses_uncovered_company() -> None:
    route = route_query("请问贵州茅台2025年的营业收入是多少？", SCOPE)
    assert route.should_refuse
    assert route.refusal_reason == "company_not_covered"
    assert route.requested_company == "贵州茅台"


def test_routes_cross_year_and_calculation() -> None:
    route = route_query("比亚迪2024到2025年营业收入增长了多少？", SCOPE)
    assert route.task_type == "cross_year"
    assert route.years == (2024, 2025)
    assert route.requires_calculation


def test_routes_cross_company() -> None:
    route = route_query("比较宁德时代和比亚迪2025年的研发投入。", SCOPE)
    assert route.task_type == "cross_company"
    assert route.companies == ("宁德时代", "比亚迪")
    assert not route.should_refuse


def test_routes_single_period_calculation() -> None:
    route = route_query("宁德时代2025年研发投入占营业收入的比例是多少？", SCOPE)
    assert route.task_type == "single_fact"
    assert not route.requires_calculation


def test_generic_query_is_not_falsely_refused() -> None:
    route = route_query("2025年营业收入最高的是哪家公司？", SCOPE)
    assert not route.should_refuse
    assert route.companies == ()


def test_builds_scope_from_dicts() -> None:
    scope = CorpusScope.from_chunks(
        [
            {"company": "乙公司", "year": 2025},
            {"company": "甲公司", "year": 2024},
        ]
    )
    assert scope == CorpusScope(("乙公司", "甲公司"), (2024, 2025))


def test_refusal_message_is_structured() -> None:
    route = route_query("苹果2024年的研发投入是多少？", SCOPE)
    message = refusal_message(route, SCOPE)
    assert "苹果" in message
    assert "宁德时代" in message
    assert "无法" in message


def test_refusal_message_rejects_answerable_route() -> None:
    route = route_query("科大讯飞2024年的营业收入是多少？", SCOPE)
    with pytest.raises(ValueError):
        refusal_message(route, SCOPE)
