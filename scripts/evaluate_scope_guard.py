from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from annual_report_agent.agent import CorpusScope, refusal_message, route_query

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate deterministic scope routing")
    parser.add_argument("--corpus", default="data/processed/pypdf_corpus.jsonl")
    parser.add_argument("--dev", default="data/eval/annual_report_dev_v2.jsonl")
    parser.add_argument("--no-answer", default="data/eval/annual_report_no_answer_v2.jsonl")
    parser.add_argument("--output", default="outputs/scope_guard_metrics_v2.json")
    return parser.parse_args()


def project_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def evaluate(corpus_rows: list[dict], dev_rows: list[dict], no_answer_rows: list[dict]) -> dict:
    scope = CorpusScope.from_chunks(corpus_rows)
    details = []
    route_counts: Counter[str] = Counter()
    reason_totals: Counter[str] = Counter()
    reason_correct: Counter[str] = Counter()

    for split, rows in (("dev", dev_rows), ("no_answer", no_answer_rows)):
        for row in rows:
            route = route_query(row["query"], scope)
            expected_refusal = split == "no_answer"
            expected_reason = row.get("reason") if expected_refusal else None
            correct = route.should_refuse == expected_refusal
            reason_match = route.refusal_reason == expected_reason
            if expected_reason:
                reason_totals[expected_reason] += 1
                if reason_match:
                    reason_correct[expected_reason] += 1
            route_counts[route.task_type] += 1
            details.append(
                {
                    "query_id": row["query_id"],
                    "split": split,
                    "query": row["query"],
                    "expected_refusal": expected_refusal,
                    "predicted_refusal": route.should_refuse,
                    "expected_reason": expected_reason,
                    "predicted_reason": route.refusal_reason,
                    "correct": correct,
                    "reason_match": reason_match,
                    "route": route.task_type,
                    "companies": list(route.companies),
                    "years": list(route.years),
                    "requires_calculation": route.requires_calculation,
                    "response": refusal_message(route, scope) if route.should_refuse else None,
                }
            )

    dev_details = [row for row in details if row["split"] == "dev"]
    no_answer_details = [row for row in details if row["split"] == "no_answer"]
    false_refusals = [row["query_id"] for row in dev_details if row["predicted_refusal"]]
    missed_refusals = [
        row["query_id"] for row in no_answer_details if not row["predicted_refusal"]
    ]
    reason_accuracy = {
        reason: reason_correct[reason] / total for reason, total in sorted(reason_totals.items())
    }
    return {
        "scope": {"companies": list(scope.companies), "years": list(scope.years)},
        "evaluated_splits": ["dev", "no_answer"],
        "frozen_test_evaluated": False,
        "counts": {"dev": len(dev_details), "no_answer": len(no_answer_details)},
        "metrics": {
            "dev_accept_rate": (
                sum(not row["predicted_refusal"] for row in dev_details) / len(dev_details)
            ),
            "no_answer_refusal_accuracy": (
                sum(row["predicted_refusal"] for row in no_answer_details)
                / len(no_answer_details)
            ),
            "no_answer_reason_accuracy": (
                sum(row["reason_match"] for row in no_answer_details)
                / len(no_answer_details)
            ),
            "overall_decision_accuracy": (
                sum(row["correct"] for row in details) / len(details)
            ),
        },
        "reason_accuracy": reason_accuracy,
        "route_counts": dict(sorted(route_counts.items())),
        "false_refusal_query_ids": false_refusals,
        "missed_refusal_query_ids": missed_refusals,
        "details": details,
    }


def main() -> None:
    args = parse_args()
    result = evaluate(
        read_jsonl(project_path(args.corpus)),
        read_jsonl(project_path(args.dev)),
        read_jsonl(project_path(args.no_answer)),
    )
    output_path = project_path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result["scope"], ensure_ascii=False))
    print(json.dumps(result["counts"], ensure_ascii=False))
    print(json.dumps(result["metrics"], ensure_ascii=False))
    print(f"frozen_test_evaluated={result['frozen_test_evaluated']}")
    print(f"output={output_path}")


if __name__ == "__main__":
    main()
