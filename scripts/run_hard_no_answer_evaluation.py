from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("HF_HOME", str(ROOT / "cache" / "huggingface"))
os.environ.setdefault("TORCH_HOME", str(ROOT / "cache" / "torch"))
sys.path.insert(0, str(ROOT / "src"))

from annual_report_agent.agent import CorpusScope, answer_from_evidence, route_query
from annual_report_agent.io_utils import read_chunks
from annual_report_agent.retrieval import (
    BM25Retriever,
    Qwen3Reranker,
    expand_query_with_section_anchors,
    infer_allowed_document_ids,
    reciprocal_rank_fusion,
    rerank_results,
)
from annual_report_agent.retrieval.dense import DenseRetriever, SentenceTransformerEncoder
from annual_report_agent.schemas import SearchResult

THRESHOLDS = (0.0, 0.5, 0.9, 0.95, 0.98, 0.99, 0.995, 0.999)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate in-scope hard no-answer queries")
    parser.add_argument(
        "--embedding-model",
        default=r".\cache\models\bge-small-zh-v1.5-annual-report-v1",
    )
    parser.add_argument(
        "--reranker-model",
        default=r".\cache\models\Qwen3-Reranker-0.6B-modelscope",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--corpus", default="data/processed/pypdf_corpus.jsonl")
    parser.add_argument(
        "--hard-no-answer",
        default="data/eval/annual_report_hard_no_answer_dev_v1.jsonl",
    )
    parser.add_argument(
        "--dev-results",
        default="outputs/agent_dev_metrics_bge_final_v1.json",
    )
    parser.add_argument(
        "--embedding-cache",
        default="cache/embeddings/pypdf_bge_small_annual_report_v1_384.npz",
    )
    parser.add_argument("--embedding-batch-size", type=int, default=32)
    parser.add_argument("--embedding-max-length", type=int, default=384)
    parser.add_argument(
        "--embedding-query-instruction",
        default="为这个句子生成表示以用于检索相关文章：",
    )
    parser.add_argument(
        "--embedding-query-template",
        default="{instruction}{query}",
    )
    parser.add_argument("--candidate-k", type=int, default=30)
    parser.add_argument("--rerank-candidates", type=int, default=10)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--bm25-weight", type=float, default=10.0)
    parser.add_argument(
        "--output",
        default="outputs/hard_no_answer_metrics_bge_final_v1.json",
    )
    return parser.parse_args()


def project_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def rebuild_results(detail: dict, chunk_by_id: dict[str, object]) -> list[SearchResult]:
    return [
        SearchResult(
            chunk=chunk_by_id[chunk_id],
            score=float(score),
            rank=rank,
            source="reranker",
        )
        for rank, (chunk_id, score) in enumerate(
            zip(detail["reranked_chunk_ids"], detail["reranked_scores"]),
            start=1,
        )
    ]


def threshold_sweep(
    dev_saved: dict,
    hard_rows: list[dict],
    hard_results: list[list[SearchResult]],
    chunk_by_id: dict[str, object],
    scope: CorpusScope,
) -> list[dict]:
    rows = []
    for threshold in THRESHOLDS:
        dev_answers = []
        for detail in dev_saved["dev_details"]:
            results = rebuild_results(detail, chunk_by_id)
            route = route_query(detail["query"], scope)
            answer = answer_from_evidence(
                detail["query"],
                route,
                results,
                scope,
                minimum_reranker_score=threshold,
            )
            dev_answers.append((answer.status, answer.answer == detail["expected_answer"]))
        hard_refusals = []
        for row, results in zip(hard_rows, hard_results):
            route = route_query(row["query"], scope)
            answer = answer_from_evidence(
                row["query"],
                route,
                results,
                scope,
                minimum_reranker_score=threshold,
                enforce_query_constraints=False,
            )
            hard_refusals.append(answer.status == "refused")
        rows.append(
            {
                "threshold": threshold,
                "dev_answer_rate": sum(status == "answered" for status, _ in dev_answers)
                / len(dev_answers),
                "dev_exact_answer_accuracy": sum(exact for _, exact in dev_answers)
                / len(dev_answers),
                "hard_refusal_accuracy_score_only": sum(hard_refusals)
                / len(hard_refusals),
            }
        )
    return rows


def main() -> None:
    args = parse_args()
    chunks = read_chunks(project_path(args.corpus))
    hard_rows = read_jsonl(project_path(args.hard_no_answer))
    dev_saved = json.loads(project_path(args.dev_results).read_text(encoding="utf-8"))
    scope = CorpusScope.from_chunks(chunks)
    chunk_by_id = {chunk.chunk_id: chunk for chunk in chunks}
    embedding_cache_path = project_path(args.embedding_cache)
    with np.load(embedding_cache_path, allow_pickle=False) as cache:
        if cache["chunk_ids"].astype(str).tolist() != [chunk.chunk_id for chunk in chunks]:
            raise ValueError("embedding cache does not match corpus")
        cached_model_path = project_path(str(cache["model"].item()))
        requested_model_path = project_path(args.embedding_model)
        if cached_model_path.resolve() != requested_model_path.resolve():
            raise ValueError("embedding cache does not match model")
        if int(cache["max_length"].item()) != args.embedding_max_length:
            raise ValueError("embedding cache does not match max length")
        cached_embeddings = np.asarray(cache["embeddings"], dtype=np.float32)

    import torch

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    encoder = SentenceTransformerEncoder(
        args.embedding_model,
        device=args.device,
        batch_size=args.embedding_batch_size,
        max_length=args.embedding_max_length,
        query_instruction=args.embedding_query_instruction or None,
        query_template=args.embedding_query_template,
    )
    dense = DenseRetriever(chunks, encoder, document_embeddings=cached_embeddings)
    bm25 = BM25Retriever(chunks)
    candidates = []
    for row in hard_rows:
        allowed = infer_allowed_document_ids(row["query"], chunks)
        dense_results = dense.search(
            row["query"], top_k=args.candidate_k, allowed_document_ids=allowed
        )
        bm25_results = bm25.search(
            expand_query_with_section_anchors(row["query"]),
            top_k=args.candidate_k,
            allowed_document_ids=allowed,
        )
        candidates.append(
            reciprocal_rank_fusion(
                [dense_results, bm25_results],
                top_k=args.candidate_k,
                weights=[1.0, args.bm25_weight],
            )
        )
    retrieval_seconds = time.perf_counter() - started
    embedding_peak_memory_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
    del encoder, dense, bm25, cached_embeddings
    gc.collect()
    torch.cuda.empty_cache()
    memory_after_embedding_unload_mb = torch.cuda.memory_allocated() / 1024 / 1024

    torch.cuda.reset_peak_memory_stats()
    reranker = Qwen3Reranker(
        args.reranker_model,
        device=args.device,
        batch_size=1,
        max_length=1024,
        instruction=(
            "Given a Chinese annual report question, judge whether the passage "
            "directly contains evidence that answers the question."
        ),
    )
    hard_results = []
    latencies = []
    for row, candidate_results in zip(hard_rows, candidates):
        query_started = time.perf_counter()
        hard_results.append(
            rerank_results(
                row["query"],
                candidate_results[: args.rerank_candidates],
                reranker,
                top_k=args.top_k,
            )
        )
        latencies.append((time.perf_counter() - query_started) * 1000)

    details = []
    for row, results in zip(hard_rows, hard_results):
        route = route_query(row["query"], scope)
        baseline = answer_from_evidence(
            row["query"],
            route,
            results,
            scope,
            enforce_query_constraints=False,
        )
        constrained = answer_from_evidence(row["query"], route, results, scope)
        details.append(
            {
                "query_id": row["query_id"],
                "query": row["query"],
                "constraint_type": row["constraint_type"],
                "baseline_status": baseline.status,
                "baseline_answer": baseline.answer,
                "constrained_status": constrained.status,
                "constrained_reason": constrained.reason,
                "reranked_chunk_ids": [result.chunk.chunk_id for result in results],
                "reranked_scores": [result.score for result in results],
            }
        )

    sweep = threshold_sweep(dev_saved, hard_rows, hard_results, chunk_by_id, scope)
    result = {
        "dataset": {
            "hard_no_answer": str(project_path(args.hard_no_answer)),
            "num_queries": len(hard_rows),
            "frozen": False,
        },
        "frozen_test_evaluated": False,
        "models": {
            "embedding": args.embedding_model,
            "reranker": args.reranker_model,
        },
        "configuration": {
            "candidate_k": args.candidate_k,
            "rerank_candidates": args.rerank_candidates,
            "top_k": args.top_k,
            "bm25_weight": args.bm25_weight,
            "embedding_max_length": args.embedding_max_length,
            "embedding_query_instruction": args.embedding_query_instruction,
            "embedding_query_template": args.embedding_query_template,
        },
        "metrics": {
            "baseline_hard_refusal_accuracy": sum(
                row["baseline_status"] == "refused" for row in details
            )
            / len(details),
            "constraint_aware_hard_refusal_accuracy": sum(
                row["constrained_status"] == "refused" for row in details
            )
            / len(details),
        },
        "threshold_sweep": sweep,
        "runtime": {
            "gpu": torch.cuda.get_device_name(0),
            "embedding_peak_memory_mb": round(embedding_peak_memory_mb, 2),
            "memory_after_embedding_unload_mb": round(memory_after_embedding_unload_mb, 2),
            "reranker_peak_memory_mb": round(
                torch.cuda.max_memory_allocated() / 1024 / 1024, 2
            ),
            "retrieval_seconds": retrieval_seconds,
            "mean_rerank_latency_ms": float(np.mean(latencies)),
            "p95_rerank_latency_ms": float(np.percentile(latencies, 95)),
        },
        "details": details,
    }
    output_path = project_path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result["metrics"], ensure_ascii=False, indent=2))
    print(json.dumps(result["threshold_sweep"], ensure_ascii=False, indent=2))
    print(json.dumps(result["runtime"], ensure_ascii=False, indent=2))
    print("frozen_test_evaluated=False")
    print(f"output={output_path}")


if __name__ == "__main__":
    main()
