from __future__ import annotations

import hashlib
import json
import re
import sys
from collections import Counter
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from annual_report_agent.agent.answerer import (
    evidence_satisfies_constraints,
    infer_evidence_constraints,
    infer_fact_spec,
)
from annual_report_agent.agent.calculator import calculate_change, format_decimal

CANDIDATE_LOCK = ROOT / "configs" / "adaptive_retrieval_v1.lock.json"
RAW_MANIFEST = ROOT / "data" / "raw" / "manifest_v3_frozen.jsonl"
CORPUS = ROOT / "data" / "processed" / "pypdf_corpus_v3_frozen.jsonl"
EXTRACTION_REPORT = ROOT / "data" / "parsed" / "pypdf_v3_frozen" / "extraction_report.json"
EMBEDDING_CACHE = ROOT / "cache" / "embeddings" / "pypdf_bge_small_annual_report_v3_frozen_384.npz"
OLD_CORPUS = ROOT / "data" / "processed" / "pypdf_corpus_v2_blind.jsonl"
TEST = ROOT / "data" / "eval" / "annual_report_v3_frozen_test.jsonl"
MANIFEST = ROOT / "data" / "eval" / "annual_report_v3_frozen_manifest.json"
RUNNER = ROOT / "scripts" / "run_adaptive_v3_frozen_test.py"
EMBEDDING_BUILDER = ROOT / "scripts" / "build_adaptive_v3_embedding_cache.py"
SELF = Path(__file__).resolve()

EXPECTED_CATEGORY_COUNTS = {
    "single_year_fact": 20,
    "cross_year_comparison": 10,
    "calculation": 8,
    "annual_quarter_confusion": 8,
    "multi_evidence": 6,
    "no_answer": 8,
}
HOLDOUT_COMPANIES = {"贵州茅台", "海尔智家", "隆基绿能"}
HOLDOUT_DOCUMENTS = {
    "moutai-2024",
    "moutai-2025",
    "haier-2024",
    "haier-2025",
    "longi-2024",
    "longi-2025",
}
NUMBER_PATTERN = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


def fact(
    answer: str,
    chunk_id: str,
    page: int,
    document_id: str,
    company: str,
    year: int,
    unit: str,
) -> dict[str, Any]:
    return {
        "answer": answer,
        "chunk_id": chunk_id,
        "page": page,
        "document_id": document_id,
        "company": company,
        "year": year,
        "unit": unit,
    }


FACTS: dict[tuple[str, int, str], dict[str, Any]] = {
    ("贵州茅台", 2024, "revenue"): fact(
        "170,899,152,276.34元", "moutai-2024:0005", 5, "moutai-2024", "贵州茅台", 2024, "元"
    ),
    ("贵州茅台", 2024, "net_profit"): fact(
        "86,228,146,421.62元", "moutai-2024:0005", 5, "moutai-2024", "贵州茅台", 2024, "元"
    ),
    ("贵州茅台", 2024, "operating_cash_flow"): fact(
        "92,463,692,168.43元", "moutai-2024:0005", 5, "moutai-2024", "贵州茅台", 2024, "元"
    ),
    ("贵州茅台", 2024, "net_assets"): fact(
        "233,105,984,399.47元", "moutai-2024:0005", 5, "moutai-2024", "贵州茅台", 2024, "元"
    ),
    ("贵州茅台", 2024, "total_assets"): fact(
        "298,944,579,918.70元", "moutai-2024:0005", 5, "moutai-2024", "贵州茅台", 2024, "元"
    ),
    ("贵州茅台", 2024, "basic_eps"): fact(
        "68.64元/股", "moutai-2024:0006", 5, "moutai-2024", "贵州茅台", 2024, "元/股"
    ),
    ("贵州茅台", 2024, "roe"): fact(
        "36.02%", "moutai-2024:0006", 5, "moutai-2024", "贵州茅台", 2024, "%"
    ),
    ("贵州茅台", 2024, "rnd_amount"): fact(
        "695,376,735.81元", "moutai-2024:0018", 11, "moutai-2024", "贵州茅台", 2024, "元"
    ),
    ("贵州茅台", 2025, "revenue"): fact(
        "168,838,102,514.79元", "moutai-2025:0004", 6, "moutai-2025", "贵州茅台", 2025, "元"
    ),
    ("贵州茅台", 2025, "net_profit"): fact(
        "82,320,067,101.68元", "moutai-2025:0004", 6, "moutai-2025", "贵州茅台", 2025, "元"
    ),
    ("贵州茅台", 2025, "operating_cash_flow"): fact(
        "61,522,204,989.35元", "moutai-2025:0004", 6, "moutai-2025", "贵州茅台", 2025, "元"
    ),
    ("贵州茅台", 2025, "net_assets"): fact(
        "244,637,811,032.18元", "moutai-2025:0004", 6, "moutai-2025", "贵州茅台", 2025, "元"
    ),
    ("贵州茅台", 2025, "basic_eps"): fact(
        "65.66元/股", "moutai-2025:0005", 6, "moutai-2025", "贵州茅台", 2025, "元/股"
    ),
    ("贵州茅台", 2025, "rnd_staff"): fact(
        "806人", "moutai-2025:0016", 12, "moutai-2025", "贵州茅台", 2025, "人"
    ),
    ("海尔智家", 2024, "revenue"): fact(
        "285,981,225,203.93元", "haier-2024:0016", 10, "haier-2024", "海尔智家", 2024, "元"
    ),
    ("海尔智家", 2024, "net_profit"): fact(
        "18,741,120,122.93元", "haier-2024:0016", 10, "haier-2024", "海尔智家", 2024, "元"
    ),
    ("海尔智家", 2024, "operating_cash_flow"): fact(
        "26,543,081,911.96元", "haier-2024:0017", 10, "haier-2024", "海尔智家", 2024, "元"
    ),
    ("海尔智家", 2024, "total_assets"): fact(
        "290,113,822,824.61元", "haier-2024:0017", 10, "haier-2024", "海尔智家", 2024, "元"
    ),
    ("海尔智家", 2024, "basic_eps"): fact(
        "2.02元/股", "haier-2024:0018", 11, "haier-2024", "海尔智家", 2024, "元/股"
    ),
    ("海尔智家", 2024, "roe"): fact(
        "17.70%", "haier-2024:0018", 11, "haier-2024", "海尔智家", 2024, "%"
    ),
    ("海尔智家", 2024, "rnd_ratio"): fact(
        "3.95%", "haier-2024:0077", 44, "haier-2024", "海尔智家", 2024, "%"
    ),
    ("海尔智家", 2025, "revenue"): fact(
        "302,346,783,918.30元", "haier-2025:0017", 12, "haier-2025", "海尔智家", 2025, "元"
    ),
    ("海尔智家", 2025, "net_profit"): fact(
        "19,552,798,222.85元", "haier-2025:0017", 12, "haier-2025", "海尔智家", 2025, "元"
    ),
    ("海尔智家", 2025, "operating_cash_flow"): fact(
        "26,002,941,969.92元", "haier-2025:0017", 12, "haier-2025", "海尔智家", 2025, "元"
    ),
    ("海尔智家", 2025, "net_assets"): fact(
        "118,698,401,416.58元", "haier-2025:0017", 12, "haier-2025", "海尔智家", 2025, "元"
    ),
    ("海尔智家", 2025, "total_assets"): fact(
        "295,795,068,591.57元", "haier-2025:0017", 12, "haier-2025", "海尔智家", 2025, "元"
    ),
    ("海尔智家", 2025, "basic_eps"): fact(
        "2.12元/股", "haier-2025:0018", 13, "haier-2025", "海尔智家", 2025, "元/股"
    ),
    ("海尔智家", 2025, "roe"): fact(
        "16.98%", "haier-2025:0018", 13, "haier-2025", "海尔智家", 2025, "%"
    ),
    ("海尔智家", 2025, "rnd_staff"): fact(
        "25,913人", "haier-2025:0078", 45, "haier-2025", "海尔智家", 2025, "人"
    ),
    ("隆基绿能", 2024, "revenue"): fact(
        "82,582,273,118.72元", "longi-2024:0014", 11, "longi-2024", "隆基绿能", 2024, "元"
    ),
    ("隆基绿能", 2024, "net_profit"): fact(
        "-8,617,528,506.44元", "longi-2024:0014", 11, "longi-2024", "隆基绿能", 2024, "元"
    ),
    ("隆基绿能", 2024, "operating_cash_flow"): fact(
        "-4,724,978,931.84元", "longi-2024:0014", 11, "longi-2024", "隆基绿能", 2024, "元"
    ),
    ("隆基绿能", 2024, "net_assets"): fact(
        "60,895,314,122.52元", "longi-2024:0014", 11, "longi-2024", "隆基绿能", 2024, "元"
    ),
    ("隆基绿能", 2024, "total_assets"): fact(
        "152,844,602,368.05元", "longi-2024:0014", 11, "longi-2024", "隆基绿能", 2024, "元"
    ),
    ("隆基绿能", 2024, "basic_eps"): fact(
        "-1.14元/股", "longi-2024:0015", 11, "longi-2024", "隆基绿能", 2024, "元/股"
    ),
    ("隆基绿能", 2024, "roe"): fact(
        "-13.10%", "longi-2024:0015", 11, "longi-2024", "隆基绿能", 2024, "%"
    ),
    ("隆基绿能", 2025, "revenue"): fact(
        "70,347,049,950.42元", "longi-2025:0016", 13, "longi-2025", "隆基绿能", 2025, "元"
    ),
    ("隆基绿能", 2025, "net_profit"): fact(
        "-6,419,556,843.85元", "longi-2025:0016", 13, "longi-2025", "隆基绿能", 2025, "元"
    ),
    ("隆基绿能", 2025, "operating_cash_flow"): fact(
        "4,359,382,755.77元", "longi-2025:0016", 13, "longi-2025", "隆基绿能", 2025, "元"
    ),
    ("隆基绿能", 2025, "net_assets"): fact(
        "54,275,611,054.63元", "longi-2025:0017", 13, "longi-2025", "隆基绿能", 2025, "元"
    ),
    ("隆基绿能", 2025, "total_assets"): fact(
        "153,803,608,808.29元", "longi-2025:0017", 13, "longi-2025", "隆基绿能", 2025, "元"
    ),
    ("隆基绿能", 2025, "basic_eps"): fact(
        "-0.85元/股", "longi-2025:0018", 14, "longi-2025", "隆基绿能", 2025, "元/股"
    ),
    ("隆基绿能", 2025, "rnd_staff"): fact(
        "3,096人", "longi-2025:0042", 30, "longi-2025", "隆基绿能", 2025, "人"
    ),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def validate_candidate_lock() -> dict[str, Any]:
    lock = json.loads(CANDIDATE_LOCK.read_text(encoding="utf-8"))
    if lock["status"] != "candidate_locked_before_new_holdout_selection":
        raise ValueError("adaptive candidate is not in the expected locked state")
    for name, artifact in lock["artifacts"].items():
        path = ROOT / artifact["path"]
        if not path.is_file() or sha256(path) != artifact["sha256"]:
            raise ValueError(f"candidate artifact changed after lock: {name}")
    return lock


def get_fact(company: str, year: int, fact_key: str) -> dict[str, Any]:
    return FACTS[(company, year, fact_key)]


def base_row(
    query: str,
    category: str,
    fact_key: str,
    facts: list[dict[str, Any]],
    answer: str,
    *,
    distractor_chunk_ids: list[str] | None = None,
) -> dict[str, Any]:
    companies = list(dict.fromkeys(item["company"] for item in facts))
    years = sorted({item["year"] for item in facts})
    return {
        "query": query,
        "category": category,
        "question_type": category,
        "fact_key": fact_key,
        "company": companies[0] if len(companies) == 1 else None,
        "companies": companies,
        "years": years,
        "answer": answer,
        "expected_action": "answer",
        "relevant_chunk_ids": [item["chunk_id"] for item in facts],
        "source_pages": [item["page"] for item in facts],
        "document_ids": [item["document_id"] for item in facts],
        "gold_values": [item["answer"] for item in facts],
        "distractor_chunk_ids": distractor_chunk_ids or [],
        "split": "v3_frozen_test",
        "frozen": True,
        "manual_verification": {
            "status": "checked_against_extracted_annual_report_page",
            "answer_checked": True,
            "page_checked": True,
            "gold_chunk_checked": True,
        },
    }


def cross_year_answer(old: dict[str, Any], new: dict[str, Any]) -> str:
    return f"2024年{old['answer']}；2025年{new['answer']}"


def calculation_answer(old: dict[str, Any], new: dict[str, Any]) -> str:
    if old["unit"] != new["unit"]:
        raise ValueError("calculation units must match")
    old_value = Decimal(NUMBER_PATTERN.search(old["answer"]).group().replace(",", ""))
    new_value = Decimal(NUMBER_PATTERN.search(new["answer"]).group().replace(",", ""))
    change = calculate_change(old_value, new_value)
    percentage = (
        format_decimal(change.percentage_change, places=2)
        if change.percentage_change is not None
        else "无法计算"
    )
    calculation = (
        f"2025年-2024年：{new['answer']}-{old['answer']}="
        f"{format_decimal(change.absolute_change)}{old['unit']}，变化率={percentage}%"
    )
    return f"{cross_year_answer(old, new)}；{calculation}"


def build_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    singles = [
        ("贵州茅台2024年末的资产总额是多少？", "贵州茅台", 2024, "total_assets"),
        ("贵州茅台2024年的基本每股收益是多少？", "贵州茅台", 2024, "basic_eps"),
        ("贵州茅台2024年的加权平均净资产收益率是多少？", "贵州茅台", 2024, "roe"),
        ("贵州茅台2025年末归属于上市公司股东的净资产是多少？", "贵州茅台", 2025, "net_assets"),
        ("贵州茅台2025年的基本每股收益是多少？", "贵州茅台", 2025, "basic_eps"),
        ("贵州茅台2025年研发人员数量是多少？", "贵州茅台", 2025, "rnd_staff"),
        ("海尔智家2024年末的资产总额是多少？", "海尔智家", 2024, "total_assets"),
        ("海尔智家2024年的基本每股收益是多少？", "海尔智家", 2024, "basic_eps"),
        ("海尔智家2024年的加权平均净资产收益率是多少？", "海尔智家", 2024, "roe"),
        ("海尔智家2025年末归属于上市公司股东的净资产是多少？", "海尔智家", 2025, "net_assets"),
        ("海尔智家2025年的基本每股收益是多少？", "海尔智家", 2025, "basic_eps"),
        ("海尔智家2025年研发人员数量是多少？", "海尔智家", 2025, "rnd_staff"),
        ("隆基绿能2024年末的资产总额是多少？", "隆基绿能", 2024, "total_assets"),
        ("隆基绿能2024年的基本每股收益是多少？", "隆基绿能", 2024, "basic_eps"),
        ("隆基绿能2024年的加权平均净资产收益率是多少？", "隆基绿能", 2024, "roe"),
        ("隆基绿能2025年末归属于上市公司股东的净资产是多少？", "隆基绿能", 2025, "net_assets"),
        ("隆基绿能2025年的基本每股收益是多少？", "隆基绿能", 2025, "basic_eps"),
        ("隆基绿能2025年研发人员数量是多少？", "隆基绿能", 2025, "rnd_staff"),
        ("贵州茅台2024年的研发投入金额是多少？", "贵州茅台", 2024, "rnd_amount"),
        ("海尔智家2024年研发投入占营业收入的比例是多少？", "海尔智家", 2024, "rnd_ratio"),
    ]
    for query, company, year, fact_key in singles:
        item = get_fact(company, year, fact_key)
        rows.append(base_row(query, "single_year_fact", fact_key, [item], item["answer"]))

    comparisons = [
        ("贵州茅台2024年和2025年的营业收入分别是多少？", "贵州茅台", "revenue"),
        ("贵州茅台2024年和2025年归属于上市公司股东的净利润分别是多少？", "贵州茅台", "net_profit"),
        (
            "贵州茅台2024年和2025年经营活动产生的现金流量净额分别是多少？",
            "贵州茅台",
            "operating_cash_flow",
        ),
        ("海尔智家2024年和2025年的营业收入分别是多少？", "海尔智家", "revenue"),
        ("海尔智家2024年和2025年归属于上市公司股东的净利润分别是多少？", "海尔智家", "net_profit"),
        ("海尔智家2024年末和2025年末的资产总额分别是多少？", "海尔智家", "total_assets"),
        ("隆基绿能2024年和2025年的营业收入分别是多少？", "隆基绿能", "revenue"),
        ("隆基绿能2024年和2025年归属于上市公司股东的净利润分别是多少？", "隆基绿能", "net_profit"),
        (
            "隆基绿能2024年和2025年经营活动产生的现金流量净额分别是多少？",
            "隆基绿能",
            "operating_cash_flow",
        ),
        ("隆基绿能2024年末和2025年末的资产总额分别是多少？", "隆基绿能", "total_assets"),
    ]
    for query, company, fact_key in comparisons:
        old = get_fact(company, 2024, fact_key)
        new = get_fact(company, 2025, fact_key)
        rows.append(
            base_row(
                query,
                "cross_year_comparison",
                fact_key,
                [old, new],
                cross_year_answer(old, new),
            )
        )

    calculations = [
        ("贵州茅台2025年营业收入较2024年变化了多少？", "贵州茅台", "revenue"),
        ("贵州茅台2025年归母净利润较2024年变化了多少？", "贵州茅台", "net_profit"),
        ("海尔智家2025年营业收入较2024年变化了多少？", "海尔智家", "revenue"),
        ("海尔智家2025年归母净利润较2024年变化了多少？", "海尔智家", "net_profit"),
        (
            "海尔智家2025年经营活动现金流量净额较2024年变化了多少？",
            "海尔智家",
            "operating_cash_flow",
        ),
        ("隆基绿能2025年营业收入较2024年变化了多少？", "隆基绿能", "revenue"),
        ("隆基绿能2025年归母净利润较2024年变化了多少？", "隆基绿能", "net_profit"),
        (
            "隆基绿能2025年经营活动现金流量净额较2024年变化了多少？",
            "隆基绿能",
            "operating_cash_flow",
        ),
    ]
    for query, company, fact_key in calculations:
        old = get_fact(company, 2024, fact_key)
        new = get_fact(company, 2025, fact_key)
        rows.append(
            base_row(
                query,
                "calculation",
                fact_key,
                [old, new],
                calculation_answer(old, new),
            )
        )

    quarter_items = [
        (
            "贵州茅台2024年全年营业收入是多少（不要使用分季度数据）？",
            "贵州茅台",
            2024,
            "revenue",
            ["moutai-2024:0007"],
        ),
        (
            "贵州茅台2025年全年归属于上市公司股东的净利润是多少（不是单季度）？",
            "贵州茅台",
            2025,
            "net_profit",
            ["moutai-2025:0006"],
        ),
        (
            "海尔智家2024年全年归属于上市公司股东的净利润是多少（不要使用季度值）？",
            "海尔智家",
            2024,
            "net_profit",
            ["haier-2024:0018", "haier-2024:0019"],
        ),
        (
            "海尔智家2025年全年经营活动产生的现金流量净额是多少（不是单季度）？",
            "海尔智家",
            2025,
            "operating_cash_flow",
            ["haier-2025:0020"],
        ),
        (
            "隆基绿能2024年全年归属于上市公司股东的净利润是多少（不要使用季度值）？",
            "隆基绿能",
            2024,
            "net_profit",
            ["longi-2024:0016"],
        ),
        (
            "隆基绿能2024年全年经营活动产生的现金流量净额是多少（不是单季度）？",
            "隆基绿能",
            2024,
            "operating_cash_flow",
            ["longi-2024:0016"],
        ),
        (
            "隆基绿能2025年全年归属于上市公司股东的净利润是多少（不要使用季度值）？",
            "隆基绿能",
            2025,
            "net_profit",
            ["longi-2025:0019"],
        ),
        (
            "隆基绿能2025年全年经营活动产生的现金流量净额是多少（不是单季度）？",
            "隆基绿能",
            2025,
            "operating_cash_flow",
            ["longi-2025:0019"],
        ),
    ]
    for query, company, year, fact_key, distractors in quarter_items:
        item = get_fact(company, year, fact_key)
        rows.append(
            base_row(
                query,
                "annual_quarter_confusion",
                fact_key,
                [item],
                item["answer"],
                distractor_chunk_ids=distractors,
            )
        )

    multi_items = [
        (
            "贵州茅台2024年末和2025年末归属于上市公司股东的净资产分别是多少？",
            "贵州茅台",
            "net_assets",
        ),
        ("贵州茅台2024年和2025年的基本每股收益分别是多少？", "贵州茅台", "basic_eps"),
        ("海尔智家2024年和2025年的基本每股收益分别是多少？", "海尔智家", "basic_eps"),
        ("海尔智家2024年和2025年的加权平均净资产收益率分别是多少？", "海尔智家", "roe"),
        (
            "隆基绿能2024年末和2025年末归属于上市公司股东的净资产分别是多少？",
            "隆基绿能",
            "net_assets",
        ),
        ("隆基绿能2024年和2025年的基本每股收益分别是多少？", "隆基绿能", "basic_eps"),
    ]
    for query, company, fact_key in multi_items:
        old = get_fact(company, 2024, fact_key)
        new = get_fact(company, 2025, fact_key)
        rows.append(
            base_row(
                query,
                "multi_evidence",
                fact_key,
                [old, new],
                cross_year_answer(old, new),
            )
        )

    no_answers = [
        ("贵州茅台2024年海外研发投入金额是多少？", "贵州茅台", 2024, "regional_rnd_amount"),
        ("贵州茅台2025年北美营业收入是多少？", "贵州茅台", 2025, "regional_revenue"),
        ("海尔智家2024年北美研发人员数量是多少？", "海尔智家", 2024, "regional_rnd_staff"),
        ("海尔智家2025年女性技术人员有多少人？", "海尔智家", 2025, "staff_attribute"),
        ("海尔智家2025年智能家居业务净利润是多少？", "海尔智家", 2025, "business_segment"),
        ("隆基绿能2024年女性研发人员数量是多少？", "隆基绿能", 2024, "staff_attribute"),
        ("隆基绿能2025年欧洲研发投入金额是多少？", "隆基绿能", 2025, "regional_rnd_amount"),
        (
            "隆基绿能2024年海外经营活动产生的现金流量净额是多少？",
            "隆基绿能",
            2024,
            "regional_cash_flow",
        ),
    ]
    for query, company, year, constraint_type in no_answers:
        spec = infer_fact_spec(query)
        if spec is None:
            raise ValueError(f"no-answer query has no supported fact schema: {query}")
        constraints = infer_evidence_constraints(query, spec)
        if not constraints:
            raise ValueError(f"no-answer query has no explicit evidence constraint: {query}")
        rows.append(
            {
                "query": query,
                "category": "no_answer",
                "question_type": "no_answer",
                "fact_key": spec.name,
                "constraint_type": constraint_type,
                "required_qualifiers": [constraint.name for constraint in constraints],
                "company": company,
                "companies": [company],
                "years": [year],
                "answer": None,
                "expected_action": "refuse",
                "relevant_chunk_ids": [],
                "source_pages": [],
                "document_ids": [
                    f"{ {'贵州茅台': 'moutai', '海尔智家': 'haier', '隆基绿能': 'longi'}[company] }-{year}"
                ],
                "gold_values": [],
                "distractor_chunk_ids": [],
                "split": "v3_frozen_test",
                "frozen": True,
                "manual_verification": {
                    "status": "checked_no_matching_constrained_disclosure_in_scoped_corpus",
                    "answer_checked": True,
                    "page_checked": True,
                    "gold_chunk_checked": True,
                },
            }
        )

    for index, row in enumerate(rows, start=1):
        row["query_id"] = f"v3-frozen-test-{index:03d}"
    return rows


def validate_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    corpus_rows = read_jsonl(CORPUS)
    corpus = {row["chunk_id"]: row for row in corpus_rows}
    quarter_heading_pages: dict[str, set[int]] = {}
    for chunk in corpus_rows:
        if "分季度主要财务" in chunk["text"]:
            quarter_heading_pages.setdefault(chunk["document_id"], set()).add(chunk["page"])
    old_companies = {row["company"] for row in read_jsonl(OLD_CORPUS)}
    overlap = HOLDOUT_COMPANIES & old_companies
    if overlap:
        raise ValueError(f"holdout companies overlap prior development corpus: {overlap}")
    if {row["company"] for row in corpus_rows} != HOLDOUT_COMPANIES:
        raise ValueError("V3 corpus companies do not exactly match the selected holdout")
    if {row["document_id"] for row in corpus_rows} != HOLDOUT_DOCUMENTS:
        raise ValueError("V3 corpus documents do not exactly match the six holdout reports")
    if len(rows) != 60:
        raise ValueError(f"expected 60 questions, found {len(rows)}")
    counts = Counter(row["category"] for row in rows)
    if dict(counts) != EXPECTED_CATEGORY_COUNTS:
        raise ValueError(f"unexpected category counts: {dict(counts)}")
    if len({re.sub(r"\s+", "", row["query"]) for row in rows}) != len(rows):
        raise ValueError("duplicate normalized query in V3 frozen test")

    no_answer_checks = []
    for row in rows:
        if row["expected_action"] == "answer":
            if len(row["relevant_chunk_ids"]) != len(row["source_pages"]):
                raise ValueError(f"{row['query_id']}: gold chunk/page count mismatch")
            for chunk_id, page, document_id, value in zip(
                row["relevant_chunk_ids"],
                row["source_pages"],
                row["document_ids"],
                row["gold_values"],
                strict=True,
            ):
                chunk = corpus.get(chunk_id)
                if chunk is None:
                    raise ValueError(f"{row['query_id']}: missing gold chunk {chunk_id}")
                if chunk["document_id"] != document_id or chunk["page"] != page:
                    raise ValueError(f"{row['query_id']}: gold metadata/page mismatch")
                if chunk["company"] not in row["companies"] or chunk["year"] not in row["years"]:
                    raise ValueError(f"{row['query_id']}: gold company/year mismatch")
                value_number = NUMBER_PATTERN.search(value)
                if value_number is None or value_number.group() not in chunk["text"]:
                    raise ValueError(
                        f"{row['query_id']}: value {value} absent from gold chunk {chunk_id}"
                    )
            for chunk_id in row["distractor_chunk_ids"]:
                distractor = corpus.get(chunk_id)
                if distractor is None:
                    raise ValueError(f"{row['query_id']}: missing distractor {chunk_id}")
                heading_pages = quarter_heading_pages.get(distractor["document_id"], set())
                near_quarter_heading = any(
                    abs(distractor["page"] - page) <= 1 for page in heading_pages
                )
                if not near_quarter_heading:
                    raise ValueError(f"{row['query_id']}: invalid quarterly distractor {chunk_id}")
        else:
            spec = infer_fact_spec(row["query"])
            if spec is None:
                raise ValueError(f"{row['query_id']}: unsupported no-answer fact")
            constraints = infer_evidence_constraints(row["query"], spec)
            matches = [
                chunk["chunk_id"]
                for chunk in corpus_rows
                if chunk["company"] in row["companies"]
                and chunk["year"] in row["years"]
                and evidence_satisfies_constraints(chunk["text"], constraints)
            ]
            if matches:
                raise ValueError(
                    f"{row['query_id']}: constrained evidence unexpectedly found: {matches[:5]}"
                )
            no_answer_checks.append(
                {
                    "query_id": row["query_id"],
                    "scoped_chunks_checked": sum(
                        chunk["company"] in row["companies"] and chunk["year"] in row["years"]
                        for chunk in corpus_rows
                    ),
                    "constraint_match_count": 0,
                }
            )
    return {
        "category_counts": dict(counts),
        "no_answer_checks": no_answer_checks,
        "prior_companies": sorted(old_companies),
        "company_overlap": sorted(overlap),
    }


def artifact(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.relative_to(ROOT)).replace("\\", "/"),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def main() -> None:
    if TEST.exists() or MANIFEST.exists():
        raise FileExistsError("V3 test is already frozen; refusing to overwrite it")
    if not RUNNER.is_file() or not EMBEDDING_BUILDER.is_file():
        raise FileNotFoundError("freeze-time runner and embedding builder must exist first")
    required = [RAW_MANIFEST, CORPUS, EXTRACTION_REPORT, EMBEDDING_CACHE, OLD_CORPUS]
    if missing := [str(path) for path in required if not path.is_file()]:
        raise FileNotFoundError(f"missing required freeze artifacts: {missing}")

    lock = validate_candidate_lock()
    rows = build_rows()
    validation = validate_rows(rows)
    TEST.parent.mkdir(parents=True, exist_ok=True)
    TEST.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )

    raw_rows = read_jsonl(RAW_MANIFEST)
    pdfs = {
        row["document_id"]: artifact(ROOT / "data" / "raw" / row["filename"]) for row in raw_rows
    }
    source_urls = {row["document_id"]: row["url"] for row in raw_rows}
    manifest = {
        "dataset_id": "annual-report-agent-adaptive-v3-frozen-eval",
        "status": "frozen_before_first_model_run",
        "frozen_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "candidate_lock": {
            "path": str(CANDIDATE_LOCK.relative_to(ROOT)).replace("\\", "/"),
            "locked_at": lock["locked_at"],
            "sha256": sha256(CANDIDATE_LOCK),
            "status": lock["status"],
        },
        "holdout_selected_after_candidate_lock": True,
        "labels_built_after_candidate_lock": True,
        "holdout_companies": sorted(HOLDOUT_COMPANIES),
        "holdout_documents": sorted(HOLDOUT_DOCUMENTS),
        "development_companies": validation["prior_companies"],
        "development_company_overlap": validation["company_overlap"],
        "test_questions": len(rows),
        "category_counts": validation["category_counts"],
        "answerable_questions": sum(row["expected_action"] == "answer" for row in rows),
        "no_answer_questions": sum(row["expected_action"] == "refuse" for row in rows),
        "manual_verification": {
            "answer_page_and_gold_chunk_checked": True,
            "method": "direct inspection of extracted annual-report page text plus consistency checks",
            "no_answer_constraint_checks": validation["no_answer_checks"],
        },
        "source_urls": source_urls,
        "artifacts": {
            "candidate_lock": artifact(CANDIDATE_LOCK),
            "source_manifest": artifact(RAW_MANIFEST),
            "source_pdfs": pdfs,
            "extraction_report": artifact(EXTRACTION_REPORT),
            "corpus": artifact(CORPUS),
            "embedding_cache": artifact(EMBEDDING_CACHE),
            "frozen_test": artifact(TEST),
            "freeze_builder": artifact(SELF),
            "embedding_builder": artifact(EMBEDDING_BUILDER),
            "single_run_evaluator": artifact(RUNNER),
        },
        "successful_model_runs_at_freeze": 0,
        "maximum_successful_frozen_test_runs": 1,
        "post_run_policy": (
            "Seal immediately after the first successful A/B/C run; do not modify verifier, "
            "rewriter, models, budgets, scoring rules, corpus, labels or parameters based on results."
        ),
    }
    MANIFEST.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
