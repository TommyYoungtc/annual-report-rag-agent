from annual_report_agent.agent import CorpusScope, RuleBasedQueryRewriter, route_query


def test_rewriter_targets_annual_section_for_second_round() -> None:
    query = "立讯精密2025年经营活动产生的现金流量净额是多少？"
    route = route_query(query, CorpusScope(("立讯精密",), (2025,)))
    rewritten = RuleBasedQueryRewriter().rewrite(
        query,
        route,
        round_index=2,
        reasons=("annual_quarter_mismatch",),
    )
    assert query in rewritten
    assert "合并现金流量表" in rewritten
    assert "年度主要会计数据" in rewritten
    assert "非分季度" in rewritten


def test_third_round_adds_financial_statement_terms() -> None:
    query = "立讯精密2025年的营业收入是多少？"
    route = route_query(query, CorpusScope(("立讯精密",), (2025,)))
    rewritten = RuleBasedQueryRewriter().rewrite(query, route, round_index=3)
    assert "合并财务报表" in rewritten
    assert "本期金额" in rewritten
