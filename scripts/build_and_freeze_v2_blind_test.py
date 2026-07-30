from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
CANDIDATE_LOCK = ROOT / "configs" / "v2_blind_candidate.lock.json"
CORPUS = ROOT / "data" / "processed" / "pypdf_corpus_v2_blind.jsonl"
DEV = ROOT / "data" / "eval" / "annual_report_v2_dev.jsonl"
TEST = ROOT / "data" / "eval" / "annual_report_v2_blind_test.jsonl"
MANIFEST = ROOT / "data" / "eval" / "annual_report_v2_blind_manifest.json"
NUMBER_PATTERN = re.compile(r"-?\d[\d,]*(?:\.\d+)?%?")

FIELD_QUERIES = {
    "revenue": "{company}{year}年的营业收入是多少？",
    "net_profit": "{company}{year}年归属于上市公司股东的净利润是多少？",
    "operating_cash_flow": "{company}{year}年经营活动产生的现金流量净额是多少？",
    "basic_eps": "{company}{year}年的基本每股收益是多少？",
    "roe": "{company}{year}年的加权平均净资产收益率是多少？",
    "total_assets": "{company}{year}年末的资产总额是多少？",
    "rnd_ratio": "{company}{year}年研发投入占营业收入的比例是多少？",
    "rnd_staff": "{company}{year}年有多少研发人员？",
}

HOLDOUT_DOCUMENTS = (
    {
        "document_id": "luxshare-2024",
        "company": "立讯精密",
        "year": 2024,
        "facts": {
            "revenue": ("268,794,737,612.58元", "luxshare-2024:0009", 8),
            "net_profit": ("13,365,651,026.16元", "luxshare-2024:0009", 8),
            "operating_cash_flow": (
                "27,116,908,208.53元",
                "luxshare-2024:0009",
                8,
            ),
            "basic_eps": ("1.86元/股", "luxshare-2024:0009", 8),
            "roe": ("21.34%", "luxshare-2024:0009", 8),
            "total_assets": ("223,827,584,433.26元", "luxshare-2024:0009", 8),
            "rnd_ratio": ("3.18%", "luxshare-2024:0066", 37),
            "rnd_staff": ("22,583人", "luxshare-2024:0065", 37),
        },
    },
    {
        "document_id": "luxshare-2025",
        "company": "立讯精密",
        "year": 2025,
        "facts": {
            "revenue": ("332,344,443,143.39元", "luxshare-2025:0007", 8),
            "net_profit": ("16,599,769,785.64元", "luxshare-2025:0007", 8),
            "operating_cash_flow": (
                "17,325,329,533.97元",
                "luxshare-2025:0007",
                8,
            ),
            "basic_eps": ("2.29元/股", "luxshare-2025:0007", 8),
            "roe": ("21.10%", "luxshare-2025:0007", 8),
            "total_assets": ("306,537,675,786.42元", "luxshare-2025:0007", 8),
            "rnd_ratio": ("3.44%", "luxshare-2025:0081", 41),
            "rnd_staff": ("34,357人", "luxshare-2025:0080", 41),
        },
    },
    {
        "document_id": "mindray-2024",
        "company": "迈瑞医疗",
        "year": 2024,
        "facts": {
            "revenue": ("36,725,749,548.00元", "mindray-2024:0027", 21),
            "net_profit": ("11,668,487,164.00元", "mindray-2024:0027", 21),
            "operating_cash_flow": (
                "12,432,041,281.00元",
                "mindray-2024:0027",
                21,
            ),
            "basic_eps": ("9.6356元/股", "mindray-2024:0028", 22),
            "roe": ("32.58%", "mindray-2024:0028", 22),
            "total_assets": ("56,643,545,143.00元", "mindray-2024:0028", 22),
            "rnd_ratio": ("10.91%", "mindray-2024:0157", 93),
            "rnd_staff": ("5,259人", "mindray-2024:0156", 92),
        },
    },
    {
        "document_id": "mindray-2025",
        "company": "迈瑞医疗",
        "year": 2025,
        "facts": {
            "revenue": ("33,282,159,404.00元", "mindray-2025:0027", 20),
            "net_profit": ("8,135,775,409.00元", "mindray-2025:0027", 20),
            "operating_cash_flow": (
                "10,144,968,535.00元",
                "mindray-2025:0027",
                20,
            ),
            "basic_eps": ("6.7147元/股", "mindray-2025:0027", 20),
            "roe": ("21.58%", "mindray-2025:0028", 21),
            "total_assets": ("59,266,767,707.00元", "mindray-2025:0028", 21),
            "rnd_ratio": ("11.80%", "mindray-2025:0192", 112),
            "rnd_staff": ("5,212人", "mindray-2025:0191", 111),
        },
    },
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def validate_candidate_lock() -> dict:
    lock = json.loads(CANDIDATE_LOCK.read_text(encoding="utf-8"))
    if lock["status"] != "candidate_locked_before_holdout_ingestion":
        raise ValueError("candidate configuration is not locked")
    for name, artifact in lock["artifacts"].items():
        path = ROOT / artifact["path"]
        if not path.is_file() or sha256(path) != artifact["sha256"]:
            raise ValueError(f"candidate artifact changed after lock: {name}")
    return lock


def build_rows() -> list[dict]:
    rows = []
    for document in HOLDOUT_DOCUMENTS:
        for fact_key, (answer, chunk_id, page) in document["facts"].items():
            rows.append(
                {
                    "query": FIELD_QUERIES[fact_key].format(
                        company=document["company"],
                        year=document["year"],
                    ),
                    "relevant_chunk_ids": [chunk_id],
                    "question_type": "single_fact",
                    "fact_key": fact_key,
                    "company": document["company"],
                    "years": [document["year"]],
                    "answer": answer,
                    "source_pages": [page],
                    "document_id": document["document_id"],
                    "split": "v2_blind_test",
                    "frozen": True,
                }
            )
    for index, row in enumerate(rows, start=1):
        row["query_id"] = f"v2-blind-test-{index:03d}"
    return rows


def validate_rows(rows: list[dict]) -> None:
    corpus = {row["chunk_id"]: row for row in read_jsonl(CORPUS)}
    dev_rows = read_jsonl(DEV)
    holdout_companies = {row["company"] for row in rows}
    holdout_documents = {row["document_id"] for row in rows}
    if holdout_companies & {row.get("company") for row in dev_rows}:
        raise ValueError("holdout company is present in the development set")
    dev_chunks = {
        chunk_id for row in dev_rows for chunk_id in row["relevant_chunk_ids"]
    }
    if any(chunk_id.split(":", 1)[0] in holdout_documents for chunk_id in dev_chunks):
        raise ValueError("holdout document is referenced by the development set")
    if len({re.sub(r"\s+", "", row["query"]) for row in rows}) != len(rows):
        raise ValueError("duplicate normalized query in frozen test")
    for row in rows:
        chunk = corpus.get(row["relevant_chunk_ids"][0])
        if chunk is None:
            raise ValueError(f"{row['query_id']}: missing gold chunk")
        if chunk["company"] != row["company"] or chunk["year"] not in row["years"]:
            raise ValueError(f"{row['query_id']}: metadata mismatch")
        if [chunk["page"]] != row["source_pages"]:
            raise ValueError(f"{row['query_id']}: source page mismatch")
        for number in NUMBER_PATTERN.findall(row["answer"]):
            if number not in chunk["text"]:
                raise ValueError(
                    f"{row['query_id']}: answer token {number} absent from evidence"
                )


def main() -> None:
    if TEST.exists() or MANIFEST.exists():
        raise FileExistsError("blind test is already frozen; refusing to overwrite it")
    lock = validate_candidate_lock()
    rows = build_rows()
    validate_rows(rows)
    TEST.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    created_at = datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()
    manifest = {
        "dataset_id": "annual-report-agent-v2-blind-eval",
        "status": "frozen_before_first_model_run",
        "frozen_at": created_at,
        "candidate_lock": {
            "path": str(CANDIDATE_LOCK.relative_to(ROOT)),
            "locked_at": lock["locked_at"],
            "sha256": sha256(CANDIDATE_LOCK),
        },
        "holdout_selected_after_candidate_lock": True,
        "holdout_companies": sorted({row["company"] for row in rows}),
        "holdout_documents": sorted({row["document_id"] for row in rows}),
        "development_company_overlap": [],
        "test_questions": len(rows),
        "fact_keys": sorted({row["fact_key"] for row in rows}),
        "frozen_test": {
            "path": str(TEST.relative_to(ROOT)),
            "sha256": sha256(TEST),
        },
        "successful_model_runs_at_freeze": 0,
    }
    MANIFEST.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
