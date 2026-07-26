from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("HF_HOME", str(ROOT / "cache" / "huggingface"))
os.environ.setdefault("TORCH_HOME", str(ROOT / "cache" / "torch"))
sys.path.insert(0, str(ROOT / "src"))

from annual_report_agent.evaluation import evaluate_retrieval
from annual_report_agent.io_utils import read_chunks, read_evaluation_queries
from annual_report_agent.retrieval import (
    BM25Retriever,
    expand_query_with_section_anchors,
    infer_allowed_document_ids,
    reciprocal_rank_fusion,
)
from annual_report_agent.retrieval.dense import (
    DenseRetriever,
    SentenceTransformerEncoder,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate dense and hybrid retrieval")
    parser.add_argument("--model", default="Qwen/Qwen3-Embedding-0.6B")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=768)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--candidate-k", type=int, default=30)
    parser.add_argument(
        "--bm25-weight",
        type=float,
        default=1.0,
        help="BM25 weight in reciprocal-rank fusion",
    )
    parser.add_argument(
        "--metadata-filter",
        action="store_true",
        help="Constrain retrieval using company and year explicitly mentioned in a query",
    )
    parser.add_argument(
        "--query-expansion",
        action="store_true",
        help="Append annual-report section anchors to the BM25 query",
    )
    parser.add_argument(
        "--embedding-cache",
        default=None,
        help="Optional NPZ cache for document embeddings",
    )
    parser.add_argument(
        "--corpus",
        default="data/samples/corpus.jsonl",
        help="Corpus JSONL path, relative paths are resolved from the project root",
    )
    parser.add_argument(
        "--eval",
        default="data/samples/eval.jsonl",
        help="Evaluation JSONL path, relative paths are resolved from the project root",
    )
    parser.add_argument(
        "--output",
        default="outputs/sample_dense_metrics.json",
        help="Metrics JSON path, relative paths are resolved from the project root",
    )
    return parser.parse_args()


def project_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def percentile(values: list[float], quantile: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(np.asarray(values, dtype=np.float64), quantile))


def main() -> None:
    args = parse_args()
    if args.top_k <= 0:
        raise ValueError("--top-k must be positive")
    if args.candidate_k < args.top_k:
        raise ValueError("--candidate-k must be greater than or equal to --top-k")
    if args.bm25_weight < 0:
        raise ValueError("--bm25-weight cannot be negative")

    corpus_path = project_path(args.corpus)
    eval_path = project_path(args.eval)
    output_path = project_path(args.output)
    chunks = read_chunks(corpus_path)
    queries = read_evaluation_queries(eval_path)
    if not queries:
        raise ValueError("evaluation set cannot be empty")

    encoder = SentenceTransformerEncoder(
        args.model,
        device=args.device,
        batch_size=args.batch_size,
        max_length=args.max_length,
        query_instruction=(
            "Given a Chinese annual report question, retrieve passages "
            "that directly support the answer."
        ),
    )

    import torch

    if args.device.startswith("cuda"):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    started = time.perf_counter()
    embedding_cache_path = project_path(args.embedding_cache) if args.embedding_cache else None
    cached_embeddings = None
    embedding_cache_hit = False
    if embedding_cache_path and embedding_cache_path.exists():
        with np.load(embedding_cache_path, allow_pickle=False) as cache:
            cached_ids = cache["chunk_ids"].astype(str).tolist()
            cached_model = str(cache["model"].item())
            cached_max_length = int(cache["max_length"].item())
            if cached_ids != [chunk.chunk_id for chunk in chunks]:
                raise ValueError("embedding cache does not match corpus chunk IDs")
            if cached_model != args.model or cached_max_length != args.max_length:
                raise ValueError("embedding cache does not match model configuration")
            cached_embeddings = cache["embeddings"]
        embedding_cache_hit = True

    dense = DenseRetriever(
        chunks,
        encoder,
        document_embeddings=cached_embeddings,
    )
    if embedding_cache_path and not embedding_cache_hit:
        embedding_cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            embedding_cache_path,
            embeddings=dense.document_embeddings,
            chunk_ids=np.asarray([chunk.chunk_id for chunk in chunks], dtype=np.str_),
            model=np.asarray(args.model),
            max_length=np.asarray(args.max_length, dtype=np.int64),
        )
    bm25 = BM25Retriever(chunks)
    index_seconds = time.perf_counter() - started
    dense_rows = []
    bm25_rows = []
    hybrid_rows = []
    rows_by_type: dict[str, dict[str, list]] = defaultdict(
        lambda: {"dense": [], "bm25": [], "hybrid": []}
    )
    query_latencies_ms = []
    query_results = []

    for item in queries:
        query_started = time.perf_counter()
        allowed_document_ids = (
            infer_allowed_document_ids(item.query, chunks) if args.metadata_filter else None
        )
        dense_results = dense.search(
            item.query,
            top_k=args.candidate_k,
            allowed_document_ids=allowed_document_ids,
        )
        bm25_query = (
            expand_query_with_section_anchors(item.query) if args.query_expansion else item.query
        )
        bm25_results = bm25.search(
            bm25_query,
            top_k=args.candidate_k,
            allowed_document_ids=allowed_document_ids,
        )
        hybrid_results = reciprocal_rank_fusion(
            [dense_results, bm25_results],
            top_k=args.top_k,
            weights=[1.0, args.bm25_weight],
        )
        latency_ms = (time.perf_counter() - query_started) * 1000
        query_latencies_ms.append(latency_ms)
        dense_row = (
            [result.chunk.chunk_id for result in dense_results[: args.top_k]],
            item.relevant_chunk_ids,
        )
        bm25_row = (
            [result.chunk.chunk_id for result in bm25_results[: args.top_k]],
            item.relevant_chunk_ids,
        )
        hybrid_row = (
            [result.chunk.chunk_id for result in hybrid_results],
            item.relevant_chunk_ids,
        )
        dense_rows.append(dense_row)
        bm25_rows.append(bm25_row)
        hybrid_rows.append(hybrid_row)
        rows_by_type[item.question_type]["dense"].append(dense_row)
        rows_by_type[item.question_type]["bm25"].append(bm25_row)
        rows_by_type[item.question_type]["hybrid"].append(hybrid_row)
        query_results.append(
            {
                "query_id": item.query_id,
                "question_type": item.question_type,
                "latency_ms": latency_ms,
                "allowed_document_ids": (
                    sorted(allowed_document_ids) if allowed_document_ids is not None else None
                ),
                "relevant_chunk_ids": list(item.relevant_chunk_ids),
                "dense_top_ids": dense_row[0],
                "bm25_top_ids": bm25_row[0],
                "hybrid_top_ids": hybrid_row[0],
            }
        )

    recall_ks = tuple(k for k in (1, 3, 5, 10) if k <= args.top_k)
    gpu = None
    if args.device.startswith("cuda"):
        gpu = {
            "name": torch.cuda.get_device_name(0),
            "peak_memory_mb": round(torch.cuda.max_memory_allocated() / 1024 / 1024, 2),
            "torch_version": torch.__version__,
            "compiled_cuda": torch.version.cuda,
        }

    result = {
        "model": args.model,
        "device": args.device,
        "runtime": {
            "python": platform.python_version(),
            "gpu": gpu,
            "batch_size": args.batch_size,
            "max_length": args.max_length,
            "metadata_filter": args.metadata_filter,
            "query_expansion": args.query_expansion,
            "bm25_weight": args.bm25_weight,
            "embedding_cache": (str(embedding_cache_path) if embedding_cache_path else None),
            "embedding_cache_hit": embedding_cache_hit,
        },
        "dataset": {
            "corpus": str(corpus_path),
            "evaluation": str(eval_path),
            "num_chunks": len(chunks),
            "num_queries": len(queries),
        },
        "index_seconds": index_seconds,
        "mean_query_latency_ms": sum(query_latencies_ms) / len(query_latencies_ms),
        "p50_query_latency_ms": percentile(query_latencies_ms, 50),
        "p95_query_latency_ms": percentile(query_latencies_ms, 95),
        "dense": evaluate_retrieval(dense_rows, recall_ks=recall_ks, ndcg_k=args.top_k),
        "bm25": evaluate_retrieval(bm25_rows, recall_ks=recall_ks, ndcg_k=args.top_k),
        "hybrid": evaluate_retrieval(hybrid_rows, recall_ks=recall_ks, ndcg_k=args.top_k),
        "by_question_type": {
            question_type: {
                "num_queries": len(group["dense"]),
                "dense": evaluate_retrieval(group["dense"], recall_ks=recall_ks, ndcg_k=args.top_k),
                "bm25": evaluate_retrieval(group["bm25"], recall_ks=recall_ks, ndcg_k=args.top_k),
                "hybrid": evaluate_retrieval(
                    group["hybrid"], recall_ks=recall_ks, ndcg_k=args.top_k
                ),
            }
            for question_type, group in sorted(rows_by_type.items())
        },
        "queries": query_results,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
