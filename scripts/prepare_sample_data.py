from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from annual_report_agent.ingestion import MarkdownDocument, chunk_markdown
from annual_report_agent.io_utils import write_chunks, write_jsonl

DOCUMENTS = [
    MarkdownDocument(
        document_id="demo-tech-2024",
        company="示例科技",
        year=2024,
        markdown="""<!-- page: 10 -->
# 主要会计数据
2024年，示例科技实现营业收入120亿元，同比增长20%；归属于母公司股东的净利润为12亿元。

<!-- page: 12 -->
# 研发投入
公司2024年研发投入为18亿元，占营业收入的15%。研发人员数量为3,200人。

<!-- page: 18 -->
# 经营活动现金流
2024年经营活动产生的现金流量净额为15亿元，主要受回款效率改善影响。
""",
    ),
    MarkdownDocument(
        document_id="demo-tech-2023",
        company="示例科技",
        year=2023,
        markdown="""<!-- page: 9 -->
# 主要会计数据
2023年，示例科技实现营业收入100亿元；归属于母公司股东的净利润为11亿元。

<!-- page: 11 -->
# 研发投入
公司2023年研发投入为14亿元，占营业收入的14%。研发人员数量为2,800人。

<!-- page: 17 -->
# 经营活动现金流
2023年经营活动产生的现金流量净额为10亿元。
""",
    ),
]


EVALUATION_ROWS = [
    {
        "query_id": "q001",
        "query": "示例科技2024年的营业收入是多少？",
        "relevant_chunk_ids": ["demo-tech-2024:0000"],
        "question_type": "single_fact",
    },
    {
        "query_id": "q002",
        "query": "示例科技2024年的研发投入是多少？",
        "relevant_chunk_ids": ["demo-tech-2024:0001"],
        "question_type": "single_fact",
    },
    {
        "query_id": "q003",
        "query": "示例科技2023年的经营活动现金流量净额是多少？",
        "relevant_chunk_ids": ["demo-tech-2023:0002"],
        "question_type": "single_fact",
    },
    {
        "query_id": "q004",
        "query": "示例科技2023年和2024年的研发投入分别是多少？",
        "relevant_chunk_ids": ["demo-tech-2023:0001", "demo-tech-2024:0001"],
        "question_type": "cross_year",
    },
]


def main() -> None:
    chunks = []
    for document in DOCUMENTS:
        chunks.extend(chunk_markdown(document, max_chars=500, overlap_chars=50, min_chars=20))

    corpus_path = ROOT / "data" / "samples" / "corpus.jsonl"
    eval_path = ROOT / "data" / "samples" / "eval.jsonl"
    write_chunks(corpus_path, chunks)
    write_jsonl(eval_path, EVALUATION_ROWS)
    print(f"Wrote {len(chunks)} chunks to {corpus_path}")
    print(f"Wrote {len(EVALUATION_ROWS)} evaluation rows to {eval_path}")


if __name__ == "__main__":
    main()
