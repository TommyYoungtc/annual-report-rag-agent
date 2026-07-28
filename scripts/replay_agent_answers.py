from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from annual_report_agent.agent import CorpusScope, answer_from_evidence, route_query
from annual_report_agent.io_utils import read_chunks
from annual_report_agent.schemas import SearchResult


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replay deterministic answer selection from saved reranker outputs"
    )
    parser.add_argument("--input", default="outputs/agent_dev_metrics_v2.json")
    parser.add_argument("--output", default="outputs/agent_dev_metrics_v2.json")
    parser.add_argument("--corpus", default="data/processed/pypdf_corpus.jsonl")
    parser.add_argument("--dev", default="data/eval/annual_report_dev_v2.jsonl")
    return parser.parse_args()


def project_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def main() -> None:
    args = parse_args()
    input_path = project_path(args.input)
    output_path = project_path(args.output)
    saved = json.loads(input_path.read_text(encoding="utf-8"))
    chunks = read_chunks(project_path(args.corpus))
    chunk_by_id = {chunk.chunk_id: chunk for chunk in chunks}
    dev_by_id = {row["query_id"]: row for row in read_jsonl(project_path(args.dev))}
    scope = CorpusScope.from_chunks(chunks)
    details = []

    for previous in saved["dev_details"]:
        row = dev_by_id[previous["query_id"]]
        reranked = [
            SearchResult(
                chunk=chunk_by_id[chunk_id],
                score=float(score),
                rank=rank,
                source="reranker",
            )
            for rank, (chunk_id, score) in enumerate(
                zip(previous["reranked_chunk_ids"], previous["reranked_scores"]),
                start=1,
            )
        ]
        route = route_query(row["query"], scope)
        answer = answer_from_evidence(
            row["query"],
            route,
            reranked,
            scope,
            minimum_reranker_score=saved["configuration"]["minimum_reranker_score"],
        )
        reranked_ids = [result.chunk.chunk_id for result in reranked]
        citation_ids = [citation.chunk_id for citation in answer.citations]
        citation_pages = [citation.page for citation in answer.citations]
        details.append(
            {
                **previous,
                "status": answer.status,
                "reason": answer.reason,
                "predicted_answer": answer.answer,
                "answer_exact_match": answer.answer == row["answer"],
                "all_gold_evidence_retrieved": all(
                    chunk_id in reranked_ids for chunk_id in row["relevant_chunk_ids"]
                ),
                "citation_chunk_ids": citation_ids,
                "citation_pages": citation_pages,
                "citation_count": len(citation_ids),
                "correct_citation_count": sum(
                    chunk_id in row["relevant_chunk_ids"] for chunk_id in citation_ids
                ),
                "citation_pages_exact_match": sorted(citation_pages)
                == sorted(row["source_pages"]),
                "calculation": answer.calculation,
            }
        )

    answered = [row for row in details if row["status"] == "answered"]
    no_answer_details = saved["no_answer_details"]
    saved["metrics"] = {
        "dev_questions": len(details),
        "dev_answer_rate": ratio(len(answered), len(details)),
        "dev_exact_answer_accuracy": ratio(
            sum(row["answer_exact_match"] for row in details), len(details)
        ),
        "dev_all_gold_evidence_recall_at_10": ratio(
            sum(row["all_gold_evidence_retrieved"] for row in details), len(details)
        ),
        "citation_chunk_precision": ratio(
            sum(row["correct_citation_count"] for row in details),
            sum(row["citation_count"] for row in details),
        ),
        "citation_page_set_accuracy": ratio(
            sum(row["citation_pages_exact_match"] for row in details), len(details)
        ),
        "no_answer_questions": len(no_answer_details),
        "no_answer_refusal_accuracy": ratio(
            sum(row["status"] == "refused" for row in no_answer_details),
            len(no_answer_details),
        ),
    }
    saved["dev_details"] = details
    saved["answer_selection_replayed"] = True
    saved["reranker_was_not_rerun"] = True
    output_path.write_text(
        json.dumps(saved, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(saved["metrics"], ensure_ascii=False, indent=2))
    print("reranker_was_not_rerun=True")
    print(f"output={output_path}")


if __name__ == "__main__":
    main()
