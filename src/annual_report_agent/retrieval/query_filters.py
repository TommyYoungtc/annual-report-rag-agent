from __future__ import annotations

import re
from collections.abc import Sequence

from ..schemas import Chunk

YEAR_PATTERN = re.compile(r"(?:19|20)\d{2}")

QUERY_ANCHOR_GROUPS: tuple[tuple[tuple[str, ...], str], ...] = (
    (
        ("营业收入", "净利润", "现金流量净额", "总资产", "净资产", "每股收益"),
        "主要会计数据 财务指标",
    ),
    (
        ("研发投入", "研发人员", "研发资本化"),
        "公司研发投入情况 公司研发人员情况 研发人员数量",
    ),
    (
        ("员工总数", "在职员工", "技术人员", "生产人员", "硕士研究生"),
        "公司员工情况 员工数量 专业构成 教育程度",
    ),
    (
        ("现金分红", "现金红利", "每10股", "每 10 股"),
        "利润分配预案 现金分红",
    ),
)


def expand_query_with_section_anchors(query: str) -> str:
    """Append likely annual-report section headings for sparse retrieval."""

    anchors = [
        anchor
        for keywords, anchor in QUERY_ANCHOR_GROUPS
        if any(keyword in query for keyword in keywords)
    ]
    return " ".join([query, *anchors]) if anchors else query


def infer_allowed_document_ids(query: str, chunks: Sequence[Chunk]) -> set[str] | None:
    """Infer company/year constraints from a query using corpus metadata.

    Returning ``None`` means that no reliable constraint was found and the
    retriever should search the complete corpus.
    """

    companies = {chunk.company for chunk in chunks if chunk.company}
    mentioned_companies = {company for company in companies if company in query}
    mentioned_years = {int(value) for value in YEAR_PATTERN.findall(query)}
    if not mentioned_companies and not mentioned_years:
        return None

    allowed = {
        chunk.document_id
        for chunk in chunks
        if (not mentioned_companies or chunk.company in mentioned_companies)
        and (not mentioned_years or chunk.year in mentioned_years)
    }
    return allowed or None
