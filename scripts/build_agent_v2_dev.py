from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NUMBER_PATTERN = re.compile(r"-?\d[\d,]*(?:\.\d+)?%?")

FIELD_QUERIES = {
    "revenue": "{company}{year}年的营业收入是多少？",
    "net_profit": "{company}{year}年归属于上市公司股东的净利润是多少？",
    "operating_cash_flow": "{company}{year}年经营活动产生的现金流量净额是多少？",
    "basic_eps": "{company}{year}年的基本每股收益是多少？",
    "roe": "{company}{year}年的加权平均净资产收益率是多少？",
    "total_assets": "{company}{year}年末的总资产是多少？",
    "rnd_ratio": "{company}{year}年研发投入占营业收入的比例是多少？",
    "rnd_staff": "{company}{year}年有多少研发人员？",
}

NEW_DOCUMENT_ROWS = (
    {
        "document_id": "hikvision-2024",
        "company": "海康威视",
        "year": 2024,
        "facts": {
            "revenue": ("92,495,525,118.30元", "hikvision-2024:0011", 9),
            "net_profit": ("11,977,327,023.54元", "hikvision-2024:0011", 9),
            "operating_cash_flow": ("13,264,092,022.73元", "hikvision-2024:0011", 9),
            "basic_eps": ("1.297元/股", "hikvision-2024:0011", 9),
            "roe": ("15.34%", "hikvision-2024:0011", 9),
            "total_assets": ("132,016,200,156.14元", "hikvision-2024:0012", 9),
            "rnd_ratio": ("12.83%", "hikvision-2024:0125", 94),
            "rnd_staff": ("28,272人", "hikvision-2024:0125", 94),
        },
    },
    {
        "document_id": "hikvision-2025",
        "company": "海康威视",
        "year": 2025,
        "facts": {
            "revenue": ("92,507,796,069.94元", "hikvision-2025:0010", 9),
            "net_profit": ("14,195,371,894.42元", "hikvision-2025:0010", 9),
            "operating_cash_flow": ("25,339,411,083.10元", "hikvision-2025:0010", 9),
            "basic_eps": ("1.546元/股", "hikvision-2025:0010", 9),
            "roe": ("17.30%", "hikvision-2025:0010", 9),
            "total_assets": ("138,049,655,334.62元", "hikvision-2025:0010", 9),
            "rnd_ratio": ("12.70%", "hikvision-2025:0117", 92),
            "rnd_staff": ("26,806人", "hikvision-2025:0117", 92),
        },
    },
    {
        "document_id": "midea-2024",
        "company": "美的集团",
        "year": 2024,
        "facts": {
            "revenue": ("407,149,600千元", "midea-2024:0009", 9),
            "net_profit": ("38,537,237千元", "midea-2024:0009", 9),
            "operating_cash_flow": ("60,511,572千元", "midea-2024:0009", 9),
            "basic_eps": ("5.44元/股", "midea-2024:0009", 9),
            "roe": ("21.29%", "midea-2024:0009", 9),
            "total_assets": ("604,351,853千元", "midea-2024:0010", 10),
            "rnd_ratio": ("3.99%", "midea-2024:0092", 53),
            "rnd_staff": ("23,693人", "midea-2024:0091", 52),
        },
    },
    {
        "document_id": "midea-2025",
        "company": "美的集团",
        "year": 2025,
        "facts": {
            "revenue": ("456,451,731千元", "midea-2025:0009", 9),
            "net_profit": ("43,945,411千元", "midea-2025:0009", 9),
            "operating_cash_flow": ("53,345,930千元", "midea-2025:0009", 9),
            "basic_eps": ("5.80元/股", "midea-2025:0009", 9),
            "roe": ("19.70%", "midea-2025:0010", 10),
            "total_assets": ("608,791,766千元", "midea-2025:0010", 10),
            "rnd_ratio": ("3.90%", "midea-2025:0090", 53),
            "rnd_staff": ("23,926人", "midea-2025:0090", 53),
        },
    },
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the V2 development and hard no-answer sets"
    )
    parser.add_argument(
        "--legacy-dev",
        type=Path,
        default=ROOT / "data" / "eval" / "annual_report_dev_v2.jsonl",
    )
    parser.add_argument(
        "--legacy-exposed-test",
        type=Path,
        default=ROOT / "data" / "eval" / "annual_report_test_v2.jsonl",
    )
    parser.add_argument(
        "--legacy-hard-no-answer",
        type=Path,
        default=ROOT / "data" / "eval" / "annual_report_hard_no_answer_dev_v1.jsonl",
    )
    parser.add_argument(
        "--corpus",
        type=Path,
        default=ROOT / "data" / "processed" / "pypdf_corpus_v2.jsonl",
    )
    parser.add_argument(
        "--dev-output",
        type=Path,
        default=ROOT / "data" / "eval" / "annual_report_v2_dev.jsonl",
    )
    parser.add_argument(
        "--hard-no-answer-output",
        type=Path,
        default=ROOT
        / "data"
        / "eval"
        / "annual_report_v2_hard_no_answer_dev.jsonl",
    )
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    path.write_text(text, encoding="utf-8")


def build_new_dev_rows() -> list[dict]:
    rows = []
    for document in NEW_DOCUMENT_ROWS:
        company = document["company"]
        year = document["year"]
        for fact_key, (answer, chunk_id, page) in document["facts"].items():
            rows.append(
                {
                    "query": FIELD_QUERIES[fact_key].format(company=company, year=year),
                    "relevant_chunk_ids": [chunk_id],
                    "question_type": "single_fact",
                    "fact_key": fact_key,
                    "company": company,
                    "years": [year],
                    "answer": answer,
                    "source_pages": [page],
                    "origin": "v2_schema_development",
                }
            )
    return rows


def build_dev_rows(legacy_dev: list[dict], exposed_test: list[dict]) -> list[dict]:
    rows = []
    sources = (
        ("v1_dev", legacy_dev),
        ("v1_exposed_test_demoted_to_dev", exposed_test),
        ("v2_new_company_dev", build_new_dev_rows()),
    )
    for origin, source_rows in sources:
        for row in source_rows:
            value = dict(row)
            value["origin"] = value.get("origin", origin)
            value["split"] = "v2_dev"
            value["frozen"] = False
            rows.append(value)
    for index, row in enumerate(rows, start=1):
        row["query_id"] = f"v2-dev-{index:03d}"
    return rows


def build_hard_no_answer_rows(legacy_rows: list[dict]) -> list[dict]:
    rows = [dict(row) for row in legacy_rows]
    for document in NEW_DOCUMENT_ROWS:
        company = document["company"]
        year = document["year"]
        segment = "主业产品及服务" if company == "海康威视" else "智能家居业务"
        rows.extend(
            [
                {
                    "query": f"{company}{year}年的海外研发投入金额是多少？",
                    "company": company,
                    "years": [year],
                    "question_type": "hard_no_answer",
                    "constraint_type": "regional_rnd_amount",
                    "required_qualifiers": ["海外", "研发投入金额"],
                    "relevant_chunk_ids": [],
                    "expected_action": "refuse",
                    "reason": "granularity_not_disclosed",
                    "verification": "corpus_keyword_search_and_manual_review",
                },
                {
                    "query": f"{company}{year}年女性研发人员数量是多少？",
                    "company": company,
                    "years": [year],
                    "question_type": "hard_no_answer",
                    "constraint_type": "intersection_workforce",
                    "required_qualifiers": ["女性", "研发人员数量"],
                    "relevant_chunk_ids": [],
                    "expected_action": "refuse",
                    "reason": "granularity_not_disclosed",
                    "verification": "corpus_keyword_search_and_manual_review",
                },
                {
                    "query": f"{company}{year}年{segment}的净利润是多少？",
                    "company": company,
                    "years": [year],
                    "question_type": "hard_no_answer",
                    "constraint_type": "segment_net_profit",
                    "required_qualifiers": [segment, "净利润"],
                    "relevant_chunk_ids": [],
                    "expected_action": "refuse",
                    "reason": "granularity_not_disclosed",
                    "verification": "corpus_keyword_search_and_manual_review",
                },
            ]
        )
    for index, row in enumerate(rows, start=1):
        row["query_id"] = f"v2-hard-na-{index:03d}"
        row["split"] = "v2_hard_no_answer_dev"
        row["frozen"] = False
    return rows


def validate_dev_rows(rows: list[dict], corpus_rows: list[dict]) -> None:
    corpus = {row["chunk_id"]: row for row in corpus_rows}
    seen_queries = set()
    for row in rows:
        normalized_query = re.sub(r"\s+", "", row["query"])
        if normalized_query in seen_queries:
            raise ValueError(f"duplicate query: {row['query']}")
        seen_queries.add(normalized_query)
        evidence = []
        pages = []
        for chunk_id in row["relevant_chunk_ids"]:
            chunk = corpus.get(chunk_id)
            if chunk is None:
                raise ValueError(f"{row['query_id']}: missing chunk {chunk_id}")
            evidence.append(chunk["text"])
            pages.append(chunk["page"])
        if pages != row["source_pages"]:
            raise ValueError(f"{row['query_id']}: source page mismatch")
        joined = " ".join(evidence)
        for number in NUMBER_PATTERN.findall(row["answer"]):
            if number not in joined:
                raise ValueError(
                    f"{row['query_id']}: answer token {number} absent from evidence"
                )


def main() -> None:
    args = parse_args()
    legacy_dev = read_jsonl(args.legacy_dev)
    exposed_test = read_jsonl(args.legacy_exposed_test)
    legacy_hard = read_jsonl(args.legacy_hard_no_answer)
    corpus = read_jsonl(args.corpus)
    dev_rows = build_dev_rows(legacy_dev, exposed_test)
    hard_rows = build_hard_no_answer_rows(legacy_hard)
    validate_dev_rows(dev_rows, corpus)
    write_jsonl(args.dev_output, dev_rows)
    write_jsonl(args.hard_no_answer_output, hard_rows)
    print(
        json.dumps(
            {
                "dev_questions": len(dev_rows),
                "hard_no_answer_questions": len(hard_rows),
                "exposed_test_demoted": len(exposed_test),
                "new_company_dev_questions": len(build_new_dev_rows()),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
