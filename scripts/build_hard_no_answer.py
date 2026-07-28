from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "data" / "eval" / "annual_report_hard_no_answer_dev_v1.jsonl"

COMPANIES = {
    "宁德时代": "储能电池系统",
    "科大讯飞": "智慧教育业务",
    "比亚迪": "新能源汽车业务",
}


def build_rows() -> list[dict]:
    rows = []
    index = 1
    for company, segment in COMPANIES.items():
        for year in (2024, 2025):
            questions = (
                (
                    f"{company}{year}年的海外研发投入金额是多少？",
                    "regional_rnd_amount",
                    ["海外", "研发投入金额"],
                ),
                (
                    f"{company}{year}年女性技术人员有多少？",
                    "intersection_workforce",
                    ["女性", "技术人员"],
                ),
                (
                    f"{company}{year}年{segment}的净利润是多少？",
                    "segment_net_profit",
                    [segment, "净利润"],
                ),
            )
            for query, constraint_type, qualifiers in questions:
                rows.append(
                    {
                        "query_id": f"hard-na-{index:03d}",
                        "query": query,
                        "company": company,
                        "years": [year],
                        "question_type": "hard_no_answer",
                        "constraint_type": constraint_type,
                        "required_qualifiers": qualifiers,
                        "relevant_chunk_ids": [],
                        "expected_action": "refuse",
                        "reason": "granularity_not_disclosed",
                        "verification": "corpus_keyword_search_and_manual_review",
                        "split": "hard_no_answer_dev",
                        "frozen": False,
                    }
                )
                index += 1
    return rows


def main() -> None:
    rows = build_rows()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"questions={len(rows)}")
    print(f"output={OUTPUT}")


if __name__ == "__main__":
    main()
