from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from annual_report_agent.io_utils import write_jsonl

DOCUMENTS = [
    ("catl-2024", "宁德时代", 2024),
    ("catl-2025", "宁德时代", 2025),
    ("iflytek-2024", "科大讯飞", 2024),
    ("iflytek-2025", "科大讯飞", 2025),
    ("byd-2024", "比亚迪", 2024),
    ("byd-2025", "比亚迪", 2025),
]

FACTS = {
    "catl-2024": {
        "revenue": ("362,012,554千元", "catl-2024:0009", 9),
        "net_profit": ("50,744,682千元", "catl-2024:0009", 9),
        "operating_cash_flow": ("96,990,345千元", "catl-2024:0009", 9),
        "basic_eps": ("11.58元/股", "catl-2024:0009", 9),
        "roe": ("24.13%", "catl-2024:0009", 9),
        "total_assets": ("786,658,123千元", "catl-2024:0009", 9),
        "net_assets": ("246,930,033千元", "catl-2024:0010", 9),
        "rnd_amount": ("18,606,756千元", "catl-2024:0039", 25),
        "rnd_ratio": ("5.14%", "catl-2024:0039", 25),
        "rnd_staff": ("20,346人", "catl-2024:0038", 24),
        "employees_total": ("131,988人", "catl-2024:0097", 57),
        "technical_employees": ("20,346人", "catl-2024:0097", 57),
        "dividend_per_10": ("45.53元（含税）", "catl-2024:0001", 3),
    },
    "catl-2025": {
        "revenue": ("423,701,834千元", "catl-2025:0013", 11),
        "net_profit": ("72,201,282千元", "catl-2025:0013", 11),
        "operating_cash_flow": ("133,219,982千元", "catl-2025:0013", 11),
        "basic_eps": ("16.14元/股", "catl-2025:0013", 11),
        "roe": ("24.91%", "catl-2025:0013", 11),
        "total_assets": ("974,827,544千元", "catl-2025:0013", 11),
        "net_assets": ("337,107,747千元", "catl-2025:0013", 11),
        "rnd_amount": ("22,146,581千元", "catl-2025:0049", 30),
        "rnd_ratio": ("5.23%", "catl-2025:0049", 30),
        "rnd_staff": ("22,901人", "catl-2025:0048", 29),
        "employees_total": ("185,839人", "catl-2025:0097", 56),
        "technical_employees": ("22,901人", "catl-2025:0097", 56),
        "dividend_per_10": ("69.57元（含税）", "catl-2025:0006", 6),
    },
    "iflytek-2024": {
        "revenue": ("23,343,093,018.69元", "iflytek-2024:0008", 7),
        "net_profit": ("560,162,663.16元", "iflytek-2024:0008", 7),
        "operating_cash_flow": ("2,495,173,454.92元", "iflytek-2024:0008", 7),
        "basic_eps": ("0.24元/股", "iflytek-2024:0008", 7),
        "roe": ("3.21%", "iflytek-2024:0008", 7),
        "total_assets": ("41,478,899,803.20元", "iflytek-2024:0008", 7),
        "net_assets": ("17,793,386,034.67元", "iflytek-2024:0008", 7),
        "rnd_amount": ("4,580,118,447.89元", "iflytek-2024:0084", 42),
        "rnd_ratio": ("19.62%", "iflytek-2024:0084", 42),
        "rnd_staff": ("9,752人", "iflytek-2024:0084", 42),
        "employees_total": ("15,551人", "iflytek-2024:0154", 76),
        "technical_employees": ("9,752人", "iflytek-2024:0154", 76),
        "dividend_per_10": ("1元（含税）", "iflytek-2024:0000", 2),
    },
    "iflytek-2025": {
        "revenue": ("27,105,390,547.66元", "iflytek-2025:0007", 7),
        "net_profit": ("839,390,861.36元", "iflytek-2025:0007", 7),
        "operating_cash_flow": ("3,208,095,271.78元", "iflytek-2025:0007", 7),
        "basic_eps": ("0.36元/股", "iflytek-2025:0007", 7),
        "roe": ("4.57%", "iflytek-2025:0007", 7),
        "total_assets": ("44,857,109,349.34元", "iflytek-2025:0008", 7),
        "net_assets": ("18,794,462,168.69元", "iflytek-2025:0008", 7),
        "rnd_amount": ("5,364,151,772.27元", "iflytek-2025:0091", 46),
        "rnd_ratio": ("19.79%", "iflytek-2025:0091", 46),
        "rnd_staff": ("10,040人", "iflytek-2025:0091", 46),
        "employees_total": ("16,818人", "iflytek-2025:0147", 72),
        "technical_employees": ("10,040人", "iflytek-2025:0147", 72),
        "dividend_per_10": ("1元（含税）", "iflytek-2025:0000", 2),
    },
    "byd-2024": {
        "revenue": ("777,102,455,000.00元", "byd-2024:0013", 10),
        "net_profit": ("40,254,346,000.00元", "byd-2024:0014", 11),
        "operating_cash_flow": ("133,453,873,000.00元", "byd-2024:0014", 11),
        "basic_eps": ("13.84元/股", "byd-2024:0014", 11),
        "roe": ("26.05%", "byd-2024:0014", 11),
        "total_assets": ("783,355,855,000.00元", "byd-2024:0014", 11),
        "net_assets": ("185,251,104,000.00元", "byd-2024:0014", 11),
        "rnd_amount": ("54,160,964,000.00元", "byd-2024:0064", 34),
        "rnd_ratio": ("6.97%", "byd-2024:0064", 34),
        "rnd_staff": ("121,598人", "byd-2024:0063", 33),
        "employees_total": ("968,872人", "byd-2024:0127", 66),
        "technical_employees": ("122,924人", "byd-2024:0127", 66),
        "dividend_per_10": ("39.74元（含税）", "byd-2024:0006", 5),
    },
    "byd-2025": {
        "revenue": ("803,964,958,000.00元", "byd-2025:0013", 11),
        "net_profit": ("32,619,022,000.00元", "byd-2025:0013", 11),
        "operating_cash_flow": ("59,135,544,000.00元", "byd-2025:0013", 11),
        "basic_eps": ("3.58元/股", "byd-2025:0013", 11),
        "roe": ("15.31%", "byd-2025:0013", 11),
        "total_assets": ("883,729,883,000.00元", "byd-2025:0013", 11),
        "net_assets": ("246,274,606,000.00元", "byd-2025:0013", 11),
        "rnd_amount": ("63,441,379,000.00元", "byd-2025:0073", 39),
        "rnd_ratio": ("7.89%", "byd-2025:0073", 39),
        "rnd_staff": ("127,665人", "byd-2025:0072", 39),
        "employees_total": ("869,622人", "byd-2025:0132", 69),
        "technical_employees": ("127,665人", "byd-2025:0132", 69),
        "dividend_per_10": ("3.58元（含税）", "byd-2025:0005", 5),
    },
}

QUESTION_TEMPLATES = {
    "revenue": ("{company}{year}年的营业收入是多少？", "single_fact"),
    "net_profit": (
        "{company}{year}年归属于上市公司股东的净利润是多少？",
        "single_fact",
    ),
    "operating_cash_flow": (
        "{company}{year}年经营活动产生的现金流量净额是多少？",
        "single_fact",
    ),
    "basic_eps": ("{company}{year}年的基本每股收益是多少？", "single_fact"),
    "roe": ("{company}{year}年的加权平均净资产收益率是多少？", "single_fact"),
    "total_assets": ("{company}{year}年末的资产总额是多少？", "single_fact"),
    "net_assets": (
        "{company}{year}年末归属于上市公司股东的净资产是多少？",
        "single_fact",
    ),
    "rnd_amount": ("{company}{year}年的研发投入金额是多少？", "research_and_development"),
    "rnd_ratio": (
        "{company}{year}年的研发投入占营业收入比例是多少？",
        "research_and_development",
    ),
    "rnd_staff": ("{company}{year}年有多少研发人员？", "research_and_development"),
    "employees_total": ("{company}{year}年末在职员工总数是多少？", "workforce"),
    "technical_employees": (
        "{company}{year}年末专业构成中的技术人员有多少？",
        "workforce",
    ),
    "dividend_per_10": (
        "{company}{year}年度利润分配预案每10股派发多少现金红利？",
        "dividend",
    ),
}

DEV_FACT_KEYS = (
    "revenue",
    "net_profit",
    "operating_cash_flow",
    "net_assets",
    "rnd_amount",
    "rnd_ratio",
    "employees_total",
    "technical_employees",
    "dividend_per_10",
)
TEST_FACT_KEYS = ("basic_eps", "roe", "total_assets", "rnd_staff")


def single_fact_row(document_id: str, company: str, year: int, fact_key: str) -> dict:
    answer, chunk_id, page = FACTS[document_id][fact_key]
    template, question_type = QUESTION_TEMPLATES[fact_key]
    return {
        "query": template.format(company=company, year=year),
        "relevant_chunk_ids": [chunk_id],
        "question_type": question_type,
        "fact_key": fact_key,
        "company": company,
        "years": [year],
        "answer": answer,
        "source_pages": [page],
    }


def cross_year_row(company: str, prefix: str, fact_key: str) -> dict:
    first = FACTS[f"{prefix}-2024"][fact_key]
    second = FACTS[f"{prefix}-2025"][fact_key]
    labels = {
        "revenue": "营业收入",
        "net_profit": "归属于上市公司股东的净利润",
        "rnd_amount": "研发投入金额",
        "employees_total": "年末在职员工总数",
    }
    return {
        "query": f"{company}2024年和2025年的{labels[fact_key]}分别是多少？",
        "relevant_chunk_ids": [first[1], second[1]],
        "question_type": "cross_year",
        "fact_key": f"cross_year_{fact_key}",
        "company": company,
        "years": [2024, 2025],
        "answer": f"2024年{first[0]}；2025年{second[0]}",
        "source_pages": [first[2], second[2]],
    }


def add_split_metadata(rows: list[dict], prefix: str, split: str, frozen: bool) -> None:
    for index, row in enumerate(rows, start=1):
        row["query_id"] = f"{prefix}-{index:03d}"
        row["split"] = split
        row["frozen"] = frozen


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    output_dir = ROOT / "data" / "eval"
    dev_rows = []
    test_rows = []
    for document_id, company, year in DOCUMENTS:
        dev_rows.extend(single_fact_row(document_id, company, year, key) for key in DEV_FACT_KEYS)
        test_rows.extend(single_fact_row(document_id, company, year, key) for key in TEST_FACT_KEYS)

    companies = (("宁德时代", "catl"), ("科大讯飞", "iflytek"), ("比亚迪", "byd"))
    for company, prefix in companies:
        dev_rows.append(cross_year_row(company, prefix, "net_profit"))
        dev_rows.append(cross_year_row(company, prefix, "employees_total"))
        test_rows.append(cross_year_row(company, prefix, "revenue"))
        test_rows.append(cross_year_row(company, prefix, "rnd_amount"))

    no_answer_rows = []
    for company in ("宁德时代", "科大讯飞", "比亚迪"):
        for fact_key, label in (
            ("future_revenue", "2026年的营业收入是多少？"),
            ("future_rnd", "2026年的研发投入金额是多少？"),
            ("future_workforce", "2026年末在职员工总数是多少？"),
        ):
            no_answer_rows.append(
                {
                    "query": f"{company}{label}",
                    "relevant_chunk_ids": [],
                    "question_type": "no_answer",
                    "fact_key": fact_key,
                    "company": company,
                    "years": [2026],
                    "expected_action": "refuse",
                    "reason": "period_not_covered",
                }
            )
    no_answer_rows.extend(
        {
            "query": query,
            "relevant_chunk_ids": [],
            "question_type": "no_answer",
            "fact_key": fact_key,
            "company": company,
            "years": [year],
            "expected_action": "refuse",
            "reason": "company_not_covered",
        }
        for query, fact_key, company, year in (
            ("贵州茅台2025年的营业收入是多少？", "outside_revenue", "贵州茅台", 2025),
            ("腾讯2025年的研发投入金额是多少？", "outside_rnd", "腾讯", 2025),
            ("阿里巴巴2024年末在职员工总数是多少？", "outside_workforce", "阿里巴巴", 2024),
            ("特斯拉2025年归属于股东的净利润是多少？", "outside_profit", "特斯拉", 2025),
            ("苹果2024年的研发投入金额是多少？", "outside_rnd", "苹果", 2024),
            ("华为2025年度每10股派发多少现金红利？", "outside_dividend", "华为", 2025),
        )
    )

    assert len(dev_rows) == 60
    assert len(test_rows) == 30
    assert len(no_answer_rows) == 15
    add_split_metadata(dev_rows, "dev", "dev", False)
    add_split_metadata(test_rows, "test", "test", True)
    add_split_metadata(no_answer_rows, "na", "no_answer", True)

    paths = {
        "dev": output_dir / "annual_report_dev_v2.jsonl",
        "test": output_dir / "annual_report_test_v2.jsonl",
        "no_answer": output_dir / "annual_report_no_answer_v2.jsonl",
    }
    rows_by_name = {"dev": dev_rows, "test": test_rows, "no_answer": no_answer_rows}
    for name, path in paths.items():
        write_jsonl(path, rows_by_name[name])

    manifest = {
        "dataset_id": "annual-report-eval-v2",
        "created_at": "2026-07-27",
        "total_questions": 105,
        "answerable_questions": 90,
        "files": {
            name: {
                "path": str(path.relative_to(ROOT)).replace("\\", "/"),
                "count": len(rows_by_name[name]),
                "frozen": name in {"test", "no_answer"},
                "sha256": sha256(path),
            }
            for name, path in paths.items()
        },
    }
    manifest_path = output_dir / "eval_v2_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
