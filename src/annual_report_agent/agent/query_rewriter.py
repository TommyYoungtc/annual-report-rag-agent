from __future__ import annotations

import re

from .answerer import infer_fact_spec
from .router import QueryRoute

FIELD_ANCHORS = {
    "revenue": "营业收入 主要会计数据和财务指标 合并利润表",
    "net_profit": "归属于上市公司股东的净利润 主要会计数据和财务指标 合并利润表",
    "operating_cash_flow": "经营活动产生的现金流量净额 主要会计数据和财务指标 合并现金流量表",
    "basic_eps": "基本每股收益 主要会计数据和财务指标",
    "roe": "加权平均净资产收益率 主要会计数据和财务指标",
    "total_assets": "总资产 主要会计数据和财务指标 合并资产负债表",
    "rnd_ratio": "研发投入占营业收入比例 公司研发投入情况",
    "rnd_staff": "研发人员数量 公司研发人员情况",
    "rnd_amount": "研发投入金额 公司研发投入情况",
    "net_assets": "归属于上市公司股东的净资产 主要会计数据和财务指标",
    "employees_total": "报告期末在职员工数量 公司员工情况",
    "technical_employees": "技术人员 公司员工情况 专业构成",
    "dividend_per_10": "每10股现金红利 利润分配预案",
}


class RuleBasedQueryRewriter:
    """Reproducible query rewriting without evaluation-label access."""

    name = "rule_based_financial_rewriter_v1"

    def rewrite(
        self,
        query: str,
        route: QueryRoute,
        *,
        round_index: int,
        reasons: tuple[str, ...] = (),
    ) -> str:
        if round_index < 2:
            return query
        spec = infer_fact_spec(query)
        anchors = FIELD_ANCHORS.get(spec.name, "年度报告 财务指标") if spec else "年度报告"
        additions = [anchors]
        if route.years and "季度" not in query:
            additions.append("年度 全年 报告期 年度报告全文 非分季度")
        if "annual_quarter_mismatch" in reasons:
            additions.append("年度主要会计数据 本年数 全年数据")
        if round_index >= 3:
            additions.append("合并财务报表 本期金额 审计报告附表")
        rewritten = " ".join([query, *additions])
        return re.sub(r"\s+", " ", rewritten).strip()
