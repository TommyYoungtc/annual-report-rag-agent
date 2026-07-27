from __future__ import annotations

import argparse
import gc
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
    SentenceTransformerReranker,
    expand_query_with_section_anchors,
    infer_allowed_document_ids,
    reciprocal_rank_fusion,
    rerank_results,
)
from annual_report_agent.retrieval.dense import (
    DenseRetriever,
    SentenceTransformerEncoder,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate sequential hybrid retrieval and CrossEncoder reranking"
    )
    parser.add_argument("--embedding-model", default="Qwen/Qwen3-Embedding-0.6B")
    parser.add_argument("--reranker-model", default="Qwen/Qwen3-Reranker-0.6B")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--embedding-batch-size", type=int, default=4)
    parser.add_argument("--reranker-batch-size", type=int, default=1)
    parser.add_argument("--embedding-max-length", type=int, default=768)
    parser.add_argument("--reranker-max-length", type=int, default=1024)
    parser.add_argument("--candidate-k", type=int, default=30)
    parser.add_argument("--rerank-candidates", type=int, default=20)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--bm25-weight", type=float, default=10.0)
    parser.add_argument(
        "--metadata-filter",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--query-expansion",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--embedding-cache",
        default="cache/embeddings/pypdf_qwen3_0.6b_768.npz",
    )
    parser.add_argument("--corpus", default="data/processed/pypdf_corpus.jsonl")
    parser.add_argument("--eval", default="data/eval/annual_report_eval_v1.jsonl")
    parser.add_argument("--output", default="outputs/real_reranker_metrics_v1.json")
    return parser.parse_args()


def project_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def percentile(values: list[float], quantile: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(np.asarray(values, dtype=np.float64), quantile))


def metrics_by_type(rows, question_types, *, top_k: int) -> dict:
    grouped = defaultdict(list)
    for row, question_type in zip(rows, question_types):
        grouped[question_type].append(row)
    recall_ks = tuple(k for k in (1, 3, 5, 10) if k <= top_k)
    return {
        name: {
            "num_queries": len(group),
            **evaluate_retrieval(group, recall_ks=recall_ks, ndcg_k=top_k),
        }
        for name, group in sorted(grouped.items())
    }


def main() -> None:
    args = parse_args()
    if not 0 < args.top_k <= args.rerank_candidates <= args.candidate_k:
        raise ValueError("expected 0 < top-k <= rerank-candidates <= candidate-k")
    if args.bm25_weight < 0:
        raise ValueError("--bm25-weight cannot be negative")

    corpus_path = project_path(args.corpus)
    eval_path = project_path(args.eval)
    output_path = project_path(args.output)
    embedding_cache_path = project_path(args.embedding_cache)
    chunks = read_chunks(corpus_path)
    queries = read_evaluation_queries(eval_path)
    if not queries:
        raise ValueError("evaluation set cannot be empty")
    if not embedding_cache_path.exists():
        raise FileNotFoundError(
            f"embedding cache not found: {embedding_cache_path}; "
            "run scripts/run_dense_evaluation.py first"
        )

    with np.load(embedding_cache_path, allow_pickle=False) as cache:
        cached_ids = cache["chunk_ids"].astype(str).tolist()
        cached_model = str(cache["model"].item())
        cached_max_length = int(cache["max_length"].item())
        if cached_ids != [chunk.chunk_id for chunk in chunks]:
            raise ValueError("embedding cache does not match corpus chunk IDs")
        if cached_model != args.embedding_model or cached_max_length != args.embedding_max_length:
            raise ValueError("embedding cache does not match embedding configuration")
        cached_embeddings = np.asarray(cache["embeddings"], dtype=np.float32)

    import torch

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    retrieval_started = time.perf_counter()
    encoder = SentenceTransformerEncoder(
        args.embedding_model,
        device=args.device,
        batch_size=args.embedding_batch_size,
        max_length=args.embedding_max_length,
        query_instruction=(
            "Given a Chinese annual report question, retrieve passages "
            "that directly support the answer."
        ),
    )
    dense = DenseRetriever(
        chunks,
        encoder,
        document_embeddings=cached_embeddings,
    )
    bm25 = BM25Retriever(chunks)
    candidates = []
    hybrid_rows = []
    question_types = []

    for item in queries:
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
            top_k=args.candidate_k,
            weights=[1.0, args.bm25_weight],
        )
        candidates.append(hybrid_results)
        hybrid_rows.append(
            (
                [result.chunk.chunk_id for result in hybrid_results[: args.top_k]],
                item.relevant_chunk_ids,
            )
        )
        question_types.append(item.question_type)

    retrieval_seconds = time.perf_counter() - retrieval_started
    retrieval_peak_memory_mb = torch.cuda.max_memory_allocated() / 1024 / 1024

    del dense, encoder, bm25, cached_embeddings
    gc.collect()
    torch.cuda.empty_cache()
    memory_after_unload_mb = torch.cuda.memory_allocated() / 1024 / 1024

    torch.cuda.reset_peak_memory_stats()
    reranker_load_started = time.perf_counter()
    reranker = SentenceTransformerReranker(
        args.reranker_model,
        device=args.device,
        batch_size=args.reranker_batch_size,
        max_length=args.reranker_max_length,
        instruction=(
            "Given a Chinese annual report question, judge whether the passage "
            "directly contains evidence that answers the question."
        ),
    )
    reranker_load_seconds = time.perf_counter() - reranker_load_started

    reranked_rows = []
    latencies_ms = []
    query_details = []
    for item, candidate_results in zip(queries, candidates):
        started = time.perf_counter()
        reranked = rerank_results(
            item.query,
            candidate_results[: args.rerank_candidates],
            reranker,
            top_k=args.top_k,
        )
        latency_ms = (time.perf_counter() - started) * 1000
        latencies_ms.append(latency_ms)
        reranked_ids = [result.chunk.chunk_id for result in reranked]
        reranked_rows.append((reranked_ids, item.relevant_chunk_ids))
        query_details.append(
            {
                "query_id": item.query_id,
                "question_type": item.question_type,
                "rerank_latency_ms": latency_ms,
                "relevant_chunk_ids": list(item.relevant_chunk_ids),
                "hybrid_top_ids": [
                    result.chunk.chunk_id for result in candidate_results[: args.top_k]
                ],
                "reranked_top_ids": reranked_ids,
            }
        )

    recall_ks = tuple(k for k in (1, 3, 5, 10) if k <= args.top_k)
    result = {
        "models": {
            "embedding": args.embedding_model,
            "reranker": args.reranker_model,
        },
        "runtime": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "compiled_cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
            "retrieval_peak_memory_mb": round(retrieval_peak_memory_mb, 2),
            "memory_after_embedding_unload_mb": round(memory_after_unload_mb, 2),
            "reranker_peak_memory_mb": round(torch.cuda.max_memory_allocated() / 1024 / 1024, 2),
        },
        "dataset": {
            "corpus": str(corpus_path),
            "evaluation": str(eval_path),
            "num_chunks": len(chunks),
            "num_queries": len(queries),
        },
        "configuration": {
            "candidate_k": args.candidate_k,
            "rerank_candidates": args.rerank_candidates,
            "top_k": args.top_k,
            "bm25_weight": args.bm25_weight,
            "metadata_filter": args.metadata_filter,
            "query_expansion": args.query_expansion,
            "embedding_max_length": args.embedding_max_length,
            "reranker_max_length": args.reranker_max_length,
            "reranker_batch_size": args.reranker_batch_size,
        },
        "timing": {
            "retrieval_seconds": retrieval_seconds,
            "reranker_load_seconds": reranker_load_seconds,
            "mean_rerank_latency_ms": sum(latencies_ms) / len(latencies_ms),
            "p50_rerank_latency_ms": percentile(latencies_ms, 50),
            "p95_rerank_latency_ms": percentile(latencies_ms, 95),
        },
        "hybrid": evaluate_retrieval(hybrid_rows, recall_ks=recall_ks, ndcg_k=args.top_k),
        "reranked": evaluate_retrieval(reranked_rows, recall_ks=recall_ks, ndcg_k=args.top_k),
        "hybrid_by_question_type": metrics_by_type(hybrid_rows, question_types, top_k=args.top_k),
        "reranked_by_question_type": metrics_by_type(
            reranked_rows, question_types, top_k=args.top_k
        ),
        "queries": query_details,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
