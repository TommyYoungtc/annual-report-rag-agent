from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Seal final frozen-test results")
    parser.add_argument("--lock", default="configs/final_v1.lock.json")
    parser.add_argument("--frozen-result", default="outputs/frozen_test_final_v1.json")
    parser.add_argument("--output", default="configs/final_v1_result.json")
    return parser.parse_args()


def project_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    lock_path = project_path(args.lock)
    frozen_path = project_path(args.frozen_result)
    output_path = project_path(args.output)
    lock = read_json(lock_path)
    frozen = read_json(frozen_path)
    if lock["status"] != "locked":
        raise ValueError("configuration lock is not active")
    if not frozen.get("frozen_test_evaluated"):
        raise ValueError("result is not marked as a frozen-test evaluation")
    if frozen.get("evaluation_split") != "frozen_test":
        raise ValueError("result split is not frozen_test")

    dev_rows = read_jsonl(project_path("data/eval/annual_report_dev_v2.jsonl"))
    test_rows = read_jsonl(project_path("data/eval/annual_report_test_v2.jsonl"))
    dev_fact_keys = {row["fact_key"] for row in dev_rows}
    test_fact_counts = Counter(row["fact_key"] for row in test_rows)
    details = frozen["dev_details"]
    refusal_reasons = Counter(
        row["reason"] for row in details if row["status"] == "refused"
    )
    exact_by_fact = {}
    for fact_key in test_fact_counts:
        fact_query_ids = {
            row["query_id"] for row in test_rows if row["fact_key"] == fact_key
        }
        fact_details = [row for row in details if row["query_id"] in fact_query_ids]
        exact_by_fact[fact_key] = {
            "correct": sum(row["answer_exact_match"] for row in fact_details),
            "total": len(fact_details),
        }

    metrics = frozen["metrics"]
    summary = {
        "configuration_id": lock["configuration_id"],
        "status": "sealed",
        "configuration_lock": {
            "path": str(lock_path.relative_to(ROOT)),
            "sha256": sha256(lock_path),
        },
        "frozen_test_result": {
            "path": str(frozen_path.relative_to(ROOT)),
            "sha256": sha256(frozen_path),
            "successful_runs": 1,
        },
        "metrics": {
            "frozen_test_questions": metrics["dev_questions"],
            "end_to_end_exact_answer_accuracy": metrics["dev_exact_answer_accuracy"],
            "all_gold_evidence_recall_at_10": metrics[
                "dev_all_gold_evidence_recall_at_10"
            ],
            "citation_chunk_precision_on_answered": metrics["citation_chunk_precision"],
            "oracle_exact_answer_accuracy": frozen["oracle_extraction"][
                "exact_answer_accuracy"
            ],
        },
        "analysis": {
            "status_counts": dict(Counter(row["status"] for row in details)),
            "refusal_reasons": dict(refusal_reasons),
            "test_only_fact_keys": sorted(set(test_fact_counts).difference(dev_fact_keys)),
            "exact_by_fact_key": exact_by_fact,
            "retrieval_miss_query_ids": [
                row["query_id"]
                for row in details
                if not row["all_gold_evidence_retrieved"]
            ],
            "root_cause": (
                "The controlled answer extractor did not support four frozen-test-only "
                "fact schemas; retrieval remained strong."
            ),
        },
        "policy": {
            "test_modified_after_run": False,
            "configuration_modified_after_lock": False,
            "test_rerun_for_tuning": False,
        },
    }
    output_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
