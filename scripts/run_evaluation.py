from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from annual_report_agent.evaluation import evaluate_retrieval
from annual_report_agent.io_utils import read_chunks, read_evaluation_queries
from annual_report_agent.retrieval import BM25Retriever


def main() -> None:
    corpus_path = ROOT / "data" / "samples" / "corpus.jsonl"
    eval_path = ROOT / "data" / "samples" / "eval.jsonl"
    output_path = ROOT / "outputs" / "sample_bm25_metrics.json"
    if not corpus_path.exists() or not eval_path.exists():
        raise SystemExit("Sample data missing. Run: python scripts/prepare_sample_data.py")

    chunks = read_chunks(corpus_path)
    queries = read_evaluation_queries(eval_path)
    retriever = BM25Retriever(chunks)
    rows = []
    details = []

    for item in queries:
        results = retriever.search(item.query, top_k=5)
        retrieved_ids = [result.chunk.chunk_id for result in results]
        rows.append((retrieved_ids, item.relevant_chunk_ids))
        details.append(
            {
                "query_id": item.query_id,
                "retrieved_chunk_ids": retrieved_ids,
                "relevant_chunk_ids": list(item.relevant_chunk_ids),
            }
        )

    metrics = evaluate_retrieval(rows, recall_ks=(1, 3, 5), ndcg_k=3)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps({"metrics": metrics, "details": details}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"Saved details to {output_path}")


if __name__ == "__main__":
    main()
