from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from annual_report_agent.agent import (
    CorpusScope,
    evidence_satisfies_constraints,
    infer_evidence_constraints,
    infer_fact_spec,
    route_query,
)
from annual_report_agent.io_utils import read_chunks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate in-scope hard no-answer set")
    parser.add_argument("--corpus", default="data/processed/pypdf_corpus.jsonl")
    parser.add_argument(
        "--eval",
        default="data/eval/annual_report_hard_no_answer_dev_v1.jsonl",
    )
    return parser.parse_args()


def project_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main() -> None:
    args = parse_args()
    chunks = read_chunks(project_path(args.corpus))
    rows = read_jsonl(project_path(args.eval))
    scope = CorpusScope.from_chunks(chunks)
    issues = []
    query_ids = set()
    queries = set()

    for row in rows:
        if row["query_id"] in query_ids:
            issues.append(f"{row['query_id']}: duplicate query_id")
        if row["query"] in queries:
            issues.append(f"{row['query_id']}: duplicate query")
        query_ids.add(row["query_id"])
        queries.add(row["query"])
        route = route_query(row["query"], scope)
        if route.should_refuse:
            issues.append(f"{row['query_id']}: query is not inside corpus scope")
        if row["relevant_chunk_ids"] or row["expected_action"] != "refuse":
            issues.append(f"{row['query_id']}: invalid no-answer labels")
        spec = infer_fact_spec(row["query"])
        if spec is None:
            issues.append(f"{row['query_id']}: query does not exercise a supported fact")
            continue
        constraints = infer_evidence_constraints(row["query"], spec)
        if not constraints:
            issues.append(f"{row['query_id']}: no evidence constraint inferred")
            continue
        candidate_chunks = [
            chunk
            for chunk in chunks
            if chunk.company == row["company"]
            and chunk.year in row["years"]
            and evidence_satisfies_constraints(chunk.text, constraints)
        ]
        if candidate_chunks:
            ids = ", ".join(chunk.chunk_id for chunk in candidate_chunks[:3])
            issues.append(f"{row['query_id']}: possible entailing evidence: {ids}")

    if len(rows) != 18:
        issues.append(f"expected 18 rows, got {len(rows)}")
    if issues:
        raise SystemExit("\n".join(issues))
    print(f"validated_questions={len(rows)}")
    print(
        "constraint_counts="
        + json.dumps(Counter(row["constraint_type"] for row in rows), ensure_ascii=False)
    )
    print("possible_entailing_chunks=0")
    print("frozen_test_read=False")


if __name__ == "__main__":
    main()
