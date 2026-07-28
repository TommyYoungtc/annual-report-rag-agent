from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from annual_report_agent.io_utils import read_chunks
from annual_report_agent.retrieval import (
    BM25Retriever,
    expand_query_with_section_anchors,
    infer_allowed_document_ids,
)
from annual_report_agent.training import (
    candidate_contains_labeled_answer,
    is_adjacent_to_gold,
    split_training_queries,
)

Candidate = tuple[str, str, int]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build hard-negative embedding training data")
    parser.add_argument("--corpus", default="data/processed/pypdf_corpus.jsonl")
    parser.add_argument("--dev", default="data/eval/annual_report_dev_v2.jsonl")
    parser.add_argument("--reranker-results", default="outputs/agent_dev_metrics_v2.json")
    parser.add_argument("--negatives-per-query", type=int, default=24)
    parser.add_argument("--bm25-candidates", type=int, default=80)
    parser.add_argument("--output-dir", default="data/training")
    return parser.parse_args()


def project_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def add_candidate(
    candidates: list[Candidate],
    seen: set[str],
    chunk_id: str,
    source: str,
    rank: int,
) -> None:
    if chunk_id in seen:
        return
    seen.add(chunk_id)
    candidates.append((chunk_id, source, rank))


def main() -> None:
    args = parse_args()
    if args.negatives_per_query <= 0:
        raise ValueError("--negatives-per-query must be positive")
    chunks = read_chunks(project_path(args.corpus))
    chunk_by_id = {chunk.chunk_id: chunk for chunk in chunks}
    dev_rows = read_jsonl(project_path(args.dev))
    train_rows, validation_rows = split_training_queries(dev_rows)
    train_ids = {row["query_id"] for row in train_rows}
    reranker_saved = json.loads(project_path(args.reranker_results).read_text(encoding="utf-8"))
    reranker_by_id = {row["query_id"]: row for row in reranker_saved["dev_details"]}
    bm25 = BM25Retriever(chunks)
    triples = []

    for row in train_rows:
        candidates: list[Candidate] = []
        seen: set[str] = set()

        saved = reranker_by_id[row["query_id"]]
        for rank, chunk_id in enumerate(saved["reranked_chunk_ids"], start=1):
            add_candidate(candidates, seen, chunk_id, "qwen3_reranker_top10", rank)

        allowed = infer_allowed_document_ids(row["query"], chunks)
        bm25_results = bm25.search(
            expand_query_with_section_anchors(row["query"]),
            top_k=args.bm25_candidates,
            allowed_document_ids=allowed,
        )
        for result in bm25_results:
            add_candidate(
                candidates,
                seen,
                result.chunk.chunk_id,
                "bm25_top80",
                result.rank,
            )

        safe_candidates = []
        for chunk_id, source, rank in candidates:
            if chunk_id in row["relevant_chunk_ids"]:
                continue
            if is_adjacent_to_gold(chunk_id, row["relevant_chunk_ids"]):
                continue
            chunk = chunk_by_id[chunk_id]
            if candidate_contains_labeled_answer(row, chunk):
                continue
            safe_candidates.append((chunk_id, source, rank))
            if len(safe_candidates) == args.negatives_per_query:
                break
        if len(safe_candidates) != args.negatives_per_query:
            raise ValueError(
                f"{row['query_id']}: expected {args.negatives_per_query} safe negatives, "
                f"found {len(safe_candidates)}"
            )

        positives = row["relevant_chunk_ids"]
        for negative_index, (negative_id, source, source_rank) in enumerate(safe_candidates):
            positive_id = positives[negative_index % len(positives)]
            triples.append(
                {
                    "triple_id": f"{row['query_id']}-hn-{negative_index + 1:02d}",
                    "query_id": row["query_id"],
                    "query": row["query"],
                    "positive_chunk_id": positive_id,
                    "negative_chunk_id": negative_id,
                    "negative_source": source,
                    "negative_source_rank": source_rank,
                    "question_type": row["question_type"],
                    "company": row["company"],
                    "years": row["years"],
                }
            )

    output_dir = project_path(args.output_dir)
    train_path = output_dir / "embedding_train_queries_v1.jsonl"
    validation_path = output_dir / "embedding_validation_queries_v1.jsonl"
    triples_path = output_dir / "embedding_hard_negatives_v1.jsonl"
    manifest_path = output_dir / "embedding_training_manifest_v1.json"
    write_jsonl(train_path, train_rows)
    write_jsonl(validation_path, validation_rows)
    write_jsonl(triples_path, triples)
    manifest = {
        "dataset_id": "annual-report-embedding-training-v1",
        "source_eval": str(project_path(args.dev)),
        "validation_rule": "every fifth query in source order",
        "train_query_count": len(train_rows),
        "validation_query_count": len(validation_rows),
        "triples_count": len(triples),
        "negatives_per_query": args.negatives_per_query,
        "train_and_validation_disjoint": train_ids.isdisjoint(
            row["query_id"] for row in validation_rows
        ),
        "files": {
            "train_queries": {
                "path": str(train_path.relative_to(ROOT)),
                "sha256": sha256(train_path),
            },
            "validation_queries": {
                "path": str(validation_path.relative_to(ROOT)),
                "sha256": sha256(validation_path),
            },
            "triples": {
                "path": str(triples_path.relative_to(ROOT)),
                "sha256": sha256(triples_path),
            },
        },
        "negative_sources": dict(Counter(row["negative_source"] for row in triples)),
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
