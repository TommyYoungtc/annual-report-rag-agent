from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

YEAR_PATTERN = re.compile(r"(?<!\d)(?:19|20)\d{2}(?!\d)")
COMPANY_BEFORE_YEAR_PATTERN = re.compile(r"([\u4e00-\u9fffA-Za-z·]{2,24})(?=(?:19|20)\d{2})")
CALCULATION_TERMS = ("同比", "增长", "下降", "增加", "减少", "差额", "变化", "高出", "低于", "相差")
QUESTION_PREFIXES = (
    "请问",
    "请查询",
    "查询",
    "请比较",
    "比较",
    "请告诉我",
    "告诉我",
    "想知道",
)


@dataclass(frozen=True, slots=True)
class CorpusScope:
    companies: tuple[str, ...]
    years: tuple[int, ...]

    @classmethod
    def from_chunks(cls, chunks: Iterable[Mapping[str, Any] | Any]) -> CorpusScope:
        companies: set[str] = set()
        years: set[int] = set()
        for chunk in chunks:
            if isinstance(chunk, Mapping):
                companies.add(str(chunk["company"]))
                years.add(int(chunk["year"]))
            else:
                companies.add(str(chunk.company))
                years.add(int(chunk.year))
        if not companies or not years:
            raise ValueError("Cannot build corpus scope from an empty corpus")
        return cls(tuple(sorted(companies)), tuple(sorted(years)))


@dataclass(frozen=True, slots=True)
class QueryRoute:
    companies: tuple[str, ...]
    years: tuple[int, ...]
    task_type: str
    requires_calculation: bool
    should_refuse: bool
    refusal_reason: str | None = None
    requested_company: str | None = None


def _possible_unknown_company(query: str) -> str | None:
    match = COMPANY_BEFORE_YEAR_PATTERN.search(query.replace(" ", ""))
    if match is None:
        return None
    candidate = match.group(1)
    for prefix in QUESTION_PREFIXES:
        if candidate.startswith(prefix):
            candidate = candidate[len(prefix) :]
            break
    return candidate or None


def route_query(query: str, scope: CorpusScope) -> QueryRoute:
    normalized = re.sub(r"\s+", "", query)
    companies = tuple(company for company in scope.companies if company in normalized)
    years = tuple(sorted({int(value) for value in YEAR_PATTERN.findall(normalized)}))
    outside_years = tuple(year for year in years if year not in scope.years)
    unknown_company = None if companies else _possible_unknown_company(normalized)
    requires_calculation = any(term in normalized for term in CALCULATION_TERMS)

    refusal_reason: str | None = None
    if outside_years:
        refusal_reason = "period_not_covered"
    elif unknown_company:
        refusal_reason = "company_not_covered"

    if refusal_reason:
        task_type = "out_of_scope"
    elif len(companies) > 1:
        task_type = "cross_company"
    elif len(years) > 1:
        task_type = "cross_year"
    elif requires_calculation:
        task_type = "calculation"
    else:
        task_type = "single_fact"

    return QueryRoute(
        companies=companies,
        years=years,
        task_type=task_type,
        requires_calculation=requires_calculation,
        should_refuse=refusal_reason is not None,
        refusal_reason=refusal_reason,
        requested_company=unknown_company,
    )


def refusal_message(route: QueryRoute, scope: CorpusScope) -> str:
    if not route.should_refuse:
        raise ValueError("A refusal message is only valid for an out-of-scope route")
    companies = "、".join(scope.companies)
    year_range = "、".join(str(year) for year in scope.years)
    if route.refusal_reason == "period_not_covered":
        requested = "、".join(str(year) for year in route.years if year not in scope.years)
        return (
            f"当前年报语料只覆盖 {year_range} 年，未覆盖 {requested} 年，"
            "因此无法基于现有资料可靠回答。"
        )
    if route.refusal_reason == "company_not_covered":
        requested = route.requested_company or "该公司"
        return (
            f"当前年报语料只覆盖 {companies}，未收录 {requested}，"
            "因此无法基于现有资料可靠回答。"
        )
    return "该问题超出当前年报语料范围，因此无法基于现有资料可靠回答。"
