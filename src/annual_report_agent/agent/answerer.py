from __future__ import annotations

import re
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation

from ..schemas import SearchResult
from .calculator import calculate_change, format_decimal
from .router import CorpusScope, QueryRoute, refusal_message

NUMBER = r"-?\d[\d,]*(?:\.\d+)?"
SPACE = r"\s*"


@dataclass(frozen=True, slots=True)
class Citation:
    chunk_id: str
    document_id: str
    company: str
    year: int
    page: int | None
    quote: str


@dataclass(frozen=True, slots=True)
class ExtractedValue:
    value_text: str
    numeric_value: Decimal
    unit: str
    citation: Citation
    score: float


@dataclass(frozen=True, slots=True)
class AgentAnswer:
    status: str
    answer: str | None
    reason: str | None
    citations: tuple[Citation, ...] = ()
    calculation: str | None = None


@dataclass(frozen=True, slots=True)
class FactSpec:
    name: str
    label_pattern: str
    default_unit: str | None = None
    dividend: bool = False
    require_explicit_unit: bool = False


@dataclass(frozen=True, slots=True)
class EvidenceConstraint:
    name: str
    patterns: tuple[str, ...]


FACT_SPECS: tuple[tuple[tuple[str, ...], FactSpec], ...] = (
    (
        ("研发投入占营业收入", "研发投入比例"),
        FactSpec("rnd_ratio", r"研发投入占营业收入(?:的)?比例", "%"),
    ),
    (
        ("归属于上市公司股东的净资产",),
        FactSpec(
            "net_assets",
            rf"归属于上市公司股东{SPACE}的净资产",
        ),
    ),
    (
        ("归属于上市公司股东的净利润", "净利润"),
        FactSpec(
            "net_profit",
            rf"归属于上市公司股东{SPACE}的净利润",
        ),
    ),
    (
        ("经营活动产生的现金流量净额",),
        FactSpec(
            "operating_cash_flow",
            rf"经营活动产生的现金{SPACE}流量净额",
        ),
    ),
    (
        ("研发投入金额", "研发投入是多少", "研发投入为多少"),
        FactSpec("rnd_amount", r"研发投入金额", require_explicit_unit=True),
    ),
    (
        ("在职员工总数", "在职员工的数量合计"),
        FactSpec(
            "employees_total",
            rf"报告期末在职员工的数量合计(?:{SPACE}（人）)?",
            "人",
        ),
    ),
    (
        ("技术人员",),
        FactSpec("technical_employees", r"技术人员", "人"),
    ),
    (
        ("每10股", "每 10 股"),
        FactSpec(
            "dividend_per_10",
            rf"每{SPACE}10{SPACE}股派发(?:现金)?(?:分红|红利)?",
            "元",
            dividend=True,
        ),
    ),
    (
        ("营业收入",),
        FactSpec("revenue", r"营业收入"),
    ),
)


def infer_fact_spec(query: str) -> FactSpec | None:
    normalized = re.sub(r"\s+", "", query)
    for keywords, spec in FACT_SPECS:
        if any(keyword.replace(" ", "") in normalized for keyword in keywords):
            return spec
    return None


def infer_evidence_constraints(query: str, spec: FactSpec) -> tuple[EvidenceConstraint, ...]:
    normalized = re.sub(r"\s+", "", query)
    constraints = []
    for term in ("海外", "境外", "欧洲", "北美"):
        if term in normalized:
            constraints.append(
                EvidenceConstraint(
                    f"region:{term}",
                    (
                        rf"{term}.{{0,40}}{spec.label_pattern}",
                        rf"{spec.label_pattern}.{{0,40}}{term}",
                    ),
                )
            )
    if spec.name == "technical_employees":
        for term in ("女性", "博士", "硕士"):
            if term in normalized:
                constraints.append(
                    EvidenceConstraint(
                        f"technical_employee_attribute:{term}",
                        (
                            rf"{term}.{{0,4}}技术人员",
                            rf"技术人员.{{0,4}}{term}",
                        ),
                    )
                )
    product_terms = (
        "储能电池系统",
        "动力电池系统",
        "智慧教育业务",
        "智慧医疗业务",
        "新能源汽车业务",
    )
    for term in product_terms:
        if term in normalized:
            constraints.append(
                EvidenceConstraint(
                    f"business_segment:{term}",
                    (
                        rf"{term}.{{0,12}}{spec.label_pattern}",
                        rf"{spec.label_pattern}.{{0,12}}{term}",
                    ),
                )
            )
    return tuple(constraints)


def evidence_satisfies_constraints(
    text: str,
    constraints: tuple[EvidenceConstraint, ...],
) -> bool:
    normalized = _normalize_text(text)
    return all(
        any(re.search(pattern, normalized) for pattern in constraint.patterns)
        for constraint in constraints
    )


def _normalize_text(text: str) -> str:
    normalized = re.sub(r"\s*([,.%])\s*", r"\1", text)
    return re.sub(r"\s+", " ", normalized)


def _unit_from_match(text: str, match: re.Match[str], spec: FactSpec) -> str:
    explicit = match.groupdict().get("unit")
    if explicit:
        return explicit
    if spec.default_unit:
        return spec.default_unit
    prefix = text[: match.start()]
    units = re.findall(r"单位[：:]?\s*(千元|万元|亿元|元)", prefix)
    return units[-1] if units else "元"


def _build_pattern(spec: FactSpec) -> re.Pattern[str]:
    if spec.dividend:
        return re.compile(
            rf"{spec.label_pattern}[^\d-]{{0,20}}(?P<value>{NUMBER}){SPACE}(?P<unit>元)"
        )
    if spec.require_explicit_unit:
        return re.compile(
            rf"{spec.label_pattern}"
            rf"{SPACE}[（(](?P<unit>千元|万元|亿元|元|人)[）)]"
            rf"[^\d-]{{0,30}}(?P<value>{NUMBER})(?P<percent>%?)"
        )
    return re.compile(
        rf"{spec.label_pattern}"
        rf"(?:{SPACE}[（(](?P<unit>千元|万元|亿元|元|人)[）)])?"
        rf"[^\d-]{{0,30}}(?P<value>{NUMBER})(?P<percent>%?)"
    )


def extract_value(result: SearchResult, spec: FactSpec) -> ExtractedValue | None:
    text = _normalize_text(result.chunk.text)
    match = _build_pattern(spec).search(text)
    if match is None:
        return None
    raw_value = match.group("value")
    try:
        numeric = Decimal(raw_value.replace(",", ""))
    except InvalidOperation:
        return None
    unit = _unit_from_match(text, match, spec)
    if match.groupdict().get("percent") == "%":
        unit = "%"
    suffix = text[match.end() : match.end() + 20]
    taxed = spec.dividend and ("含税" in suffix or "含税" in match.group(0))
    value_text = f"{raw_value}{unit}" + ("（含税）" if taxed else "")
    quote_start = max(0, match.start() - 45)
    quote_end = min(len(text), match.end() + 80)
    citation = Citation(
        chunk_id=result.chunk.chunk_id,
        document_id=result.chunk.document_id,
        company=result.chunk.company,
        year=result.chunk.year,
        page=result.chunk.page,
        quote=text[quote_start:quote_end].strip(),
    )
    return ExtractedValue(value_text, numeric, unit, citation, result.score)


def _evidence_priority(result: SearchResult, spec: FactSpec) -> tuple[int, int, float]:
    text = _normalize_text(result.chunk.text)
    preferred_markers = {
        "revenue": ("主要会计数据和财务指标",),
        "net_profit": ("主要会计数据和财务指标",),
        "operating_cash_flow": ("主要会计数据和财务指标",),
        "net_assets": ("主要会计数据和财务指标",),
        "rnd_amount": ("近三年公司研发投入金额",),
        "rnd_ratio": ("近三年公司研发投入金额",),
        "employees_total": ("公司员工情况", "员工数量、专业构成及教育程度"),
        "technical_employees": ("公司员工情况", "员工数量、专业构成及教育程度"),
        "dividend_per_10": ("董事会审议的报告期利润分配预案", "年度报告 第一节 重要提示"),
    }
    marker_score = sum(
        marker in text for marker in preferred_markers.get(spec.name, ())
    )
    page = result.chunk.page if result.chunk.page is not None else 10**9
    return marker_score, -page, result.score


def _refused(reason: str, message: str) -> AgentAnswer:
    return AgentAnswer(status="refused", answer=message, reason=reason)


def answer_from_evidence(
    query: str,
    route: QueryRoute,
    results: list[SearchResult],
    scope: CorpusScope,
    *,
    minimum_reranker_score: float = 0.0,
    enforce_query_constraints: bool = True,
) -> AgentAnswer:
    if route.should_refuse:
        return _refused(route.refusal_reason or "out_of_scope", refusal_message(route, scope))
    spec = infer_fact_spec(query)
    if spec is None:
        return _refused("unsupported_fact", "当前受控回答器尚不支持该问题所需的财务字段。")
    constraints = infer_evidence_constraints(query, spec) if enforce_query_constraints else ()

    requested_years = route.years
    requested_companies = set(route.companies)
    extracted: dict[int, ExtractedValue] = {}
    priorities: dict[int, tuple[int, int, float]] = {}
    for result in results:
        if requested_companies and result.chunk.company not in requested_companies:
            continue
        if requested_years and result.chunk.year not in requested_years:
            continue
        if result.source == "reranker" and result.score < minimum_reranker_score:
            continue
        if constraints and not evidence_satisfies_constraints(result.chunk.text, constraints):
            continue
        value = extract_value(result, spec)
        priority = _evidence_priority(result, spec)
        if value is not None and (
            result.chunk.year not in extracted or priority > priorities[result.chunk.year]
        ):
            extracted[result.chunk.year] = value
            priorities[result.chunk.year] = priority

    if requested_years and any(year not in extracted for year in requested_years):
        return _refused(
            "insufficient_evidence",
            "没有找到同时满足公司、年份、字段和页码要求的完整证据，因此拒绝作答。",
        )
    if not extracted:
        return _refused(
            "insufficient_evidence",
            "没有找到可核验的字段数值与页码证据，因此拒绝作答。",
        )

    ordered = [(year, extracted[year]) for year in sorted(extracted)]
    citations = tuple(value.citation for _, value in ordered)
    if len(ordered) == 1:
        answer = ordered[0][1].value_text
    else:
        answer = "；".join(f"{year}年{value.value_text}" for year, value in ordered)

    calculation = None
    if route.requires_calculation and len(ordered) >= 2:
        first_year, first = ordered[0]
        last_year, last = ordered[-1]
        if first.unit != last.unit:
            return _refused("incompatible_units", "证据中的单位不一致，无法进行可靠计算。")
        change = calculate_change(first.numeric_value, last.numeric_value)
        absolute = format_decimal(change.absolute_change)
        percentage = (
            format_decimal(change.percentage_change, places=2)
            if change.percentage_change is not None
            else "无法计算"
        )
        calculation = (
            f"{last_year}年-{first_year}年：{last.value_text}-{first.value_text}"
            f"={absolute}{first.unit}，变化率={percentage}%"
        )
        answer = f"{answer}；{calculation}"

    return AgentAnswer(
        status="answered",
        answer=answer,
        reason=None,
        citations=citations,
        calculation=calculation,
    )
