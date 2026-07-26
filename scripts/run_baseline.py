from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from annual_report_agent.io_utils import read_chunks, read_evaluation_queries
from annual_report_agent.retrieval import BM25Retriever


def main() -> None:
    corpus_path = ROOT / "data" / "samples" / "corpus.jsonl"
    eval_path = ROOT / "data" / "samples" / "eval.jsonl"
    if not corpus_path.exists() or not eval_path.exists():
        raise SystemExit("Sample data missing. Run: python scripts/prepare_sample_data.py")

    chunks = read_chunks(corpus_path)
    queries = read_evaluation_queries(eval_path)
    retriever = BM25Retriever(chunks)

    for item in queries:
        print(f"\n[{item.query_id}] {item.query}")
        for result in retriever.search(item.query, top_k=3):
            page = result.chunk.page if result.chunk.page is not None else "?"
            print(
                f"  {result.rank}. {result.chunk.chunk_id} "
                f"score={result.score:.4f} page={page} section={result.chunk.section}"
            )


if __name__ == "__main__":
    main()
