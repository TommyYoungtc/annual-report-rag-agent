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
from annual_report_agent.training import (
    candidate_contains_labeled_answer,
    is_adjacent_to_gold,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate embedding hard-negative data")
    parser.add_argument("--corpus", default="data/processed/pypdf_corpus.jsonl")
    parser.add_argument("--data-dir", default="data/training")
    return parser.parse_args()


def project_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def main() -> None:
    args = parse_args()
    data_dir = project_path(args.data_dir)
    train_path = data_dir / "embedding_train_queries_v1.jsonl"
    validation_path = data_dir / "embedding_validation_queries_v1.jsonl"
    triples_path = data_dir / "embedding_hard_negatives_v1.jsonl"
    manifest_path = data_dir / "embedding_training_manifest_v1.json"

    chunks = read_chunks(project_path(args.corpus))
    chunk_by_id = {chunk.chunk_id: chunk for chunk in chunks}
    train = read_jsonl(train_path)
    validation = read_jsonl(validation_path)
    triples = read_jsonl(triples_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    train_by_id = {row["query_id"]: row for row in train}
    validation_ids = {row["query_id"] for row in validation}
    require(len(train_by_id) == len(train), "duplicate training query IDs")
    require(train_by_id.keys().isdisjoint(validation_ids), "training/validation query overlap")
    require(len(train) == manifest["train_query_count"], "training count differs from manifest")
    require(
        len(validation) == manifest["validation_query_count"],
        "validation count differs from manifest",
    )
    require(len(triples) == manifest["triples_count"], "triple count differs from manifest")

    files = manifest["files"]
    for key, path in (
        ("train_queries", train_path),
        ("validation_queries", validation_path),
        ("triples", triples_path),
    ):
        require(sha256(path) == files[key]["sha256"], f"{key} hash differs from manifest")

    counts: Counter[str] = Counter()
    triple_ids: set[str] = set()
    for triple in triples:
        query_id = triple["query_id"]
        require(triple["triple_id"] not in triple_ids, "duplicate triple ID")
        triple_ids.add(triple["triple_id"])
        require(query_id in train_by_id, f"{query_id}: not a training query")
        require(query_id not in validation_ids, f"{query_id}: leaked validation query")
        row = train_by_id[query_id]
        positive_id = triple["positive_chunk_id"]
        negative_id = triple["negative_chunk_id"]
        require(positive_id in row["relevant_chunk_ids"], f"{query_id}: invalid positive")
        require(positive_id in chunk_by_id, f"{query_id}: positive chunk missing")
        require(negative_id in chunk_by_id, f"{query_id}: negative chunk missing")
        require(negative_id not in row["relevant_chunk_ids"], f"{query_id}: gold used as negative")
        require(
            not is_adjacent_to_gold(negative_id, row["relevant_chunk_ids"]),
            f"{query_id}: adjacent chunk used as negative",
        )
        require(
            not candidate_contains_labeled_answer(row, chunk_by_id[negative_id]),
            f"{query_id}: answer-bearing chunk used as negative",
        )
        counts[query_id] += 1

    expected = manifest["negatives_per_query"]
    require(set(counts) == set(train_by_id), "some training queries have no triples")
    require(all(count == expected for count in counts.values()), "uneven negatives per query")
    print(
        json.dumps(
            {
                "status": "valid",
                "train_queries": len(train),
                "validation_queries": len(validation),
                "triples": len(triples),
                "negatives_per_query": expected,
                "validation_leakage": 0,
                "unsafe_negatives": 0,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
