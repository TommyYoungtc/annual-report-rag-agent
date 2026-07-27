from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NUMBER_PATTERN = re.compile(r"-?\d[\d,]*(?:\.\d+)?%?")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate annual-report eval v2")
    parser.add_argument("--corpus", default="data/processed/pypdf_corpus.jsonl")
    parser.add_argument("--manifest", default="data/eval/eval_v2_manifest.json")
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


def main() -> None:
    args = parse_args()
    corpus_path = project_path(args.corpus)
    manifest_path = project_path(args.manifest)
    corpus_rows = read_jsonl(corpus_path)
    corpus = {row["chunk_id"]: row for row in corpus_rows}
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    issues = []
    all_rows = []

    expected = {"dev": 60, "test": 30, "no_answer": 15}
    for name, expected_count in expected.items():
        metadata = manifest["files"][name]
        path = ROOT / metadata["path"]
        rows = read_jsonl(path)
        all_rows.extend(rows)
        if len(rows) != expected_count or metadata["count"] != expected_count:
            issues.append(f"{name}: expected {expected_count} rows")
        if sha256(path) != metadata["sha256"]:
            issues.append(f"{name}: SHA-256 does not match frozen manifest")

        for row in rows:
            relevant_ids = row["relevant_chunk_ids"]
            if name == "no_answer":
                if relevant_ids:
                    issues.append(f"{row['query_id']}: no-answer row has evidence")
                if row.get("expected_action") != "refuse":
                    issues.append(f"{row['query_id']}: expected_action must be refuse")
                continue
            if not relevant_ids:
                issues.append(f"{row['query_id']}: answerable row has no evidence")
                continue
            evidence = []
            evidence_pages = []
            for chunk_id in relevant_ids:
                chunk = corpus.get(chunk_id)
                if chunk is None:
                    issues.append(f"{row['query_id']}: missing chunk {chunk_id}")
                    continue
                evidence.append(chunk["text"])
                evidence_pages.append(chunk["page"])
            joined = " ".join(evidence)
            for number in NUMBER_PATTERN.findall(row["answer"]):
                if number not in joined:
                    issues.append(f"{row['query_id']}: answer token {number} absent from evidence")
            if sorted(row["source_pages"]) != sorted(evidence_pages):
                issues.append(f"{row['query_id']}: source pages do not match chunks")

    query_ids = [row["query_id"] for row in all_rows]
    queries = [re.sub(r"\s+", "", row["query"]) for row in all_rows]
    if len(query_ids) != len(set(query_ids)):
        issues.append("duplicate query_id across splits")
    if len(queries) != len(set(queries)):
        issues.append("duplicate normalized query text across splits")
    if len(all_rows) != manifest["total_questions"]:
        issues.append("total question count does not match manifest")

    if issues:
        raise SystemExit("\n".join(issues))

    print(f"validated_questions={len(all_rows)}")
    print(f"validated_chunks={len(corpus)}")
    print("split_counts=" + json.dumps(Counter(row["split"] for row in all_rows)))
    print(
        "type_counts="
        + json.dumps(Counter(row["question_type"] for row in all_rows), ensure_ascii=False)
    )
    print("frozen_hashes=verified")


if __name__ == "__main__":
    main()
