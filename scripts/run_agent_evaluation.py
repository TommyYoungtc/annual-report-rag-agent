from __future__ import annotations

import argparse
import gc
import json
import os
import platform
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate routed Hybrid + Reranker evidence answers on Eval v2"
    )
    parser.add_argument(
        "--embedding-model",
        default=r".\cache\models\Qwen3-Embedding-0.6B-modelscope",
    )
    parser.add_argument(
        "--reranker-model",
        default=r".\cache\models\Qwen3-Reranker-0.6B-modelscope",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--embedding-batch-size", type=int, default=4)
    parser.add_argument("--reranker-batch-size", type=int, default=1)
    parser.add_argument("--embedding-max-length", type=int, default=768)
    parser.add_argument("--reranker-max-length", type=int, default=1024)
    parser.add_argument("--candidate-k", type=int, default=30)
    parser.add_argument("--rerank-candidates", type=int, default=10)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--bm25-weight", type=float, default=10.0)
    parser.add_argument("--minimum-reranker-score", type=float, default=0.0)
    parser.add_argument(
        "--embedding-cache",
        default="cache/embeddings/pypdf_qwen3_0.6b_768.npz",
    )
    parser.add_argument("--corpus", default="data/processed/pypdf_corpus.jsonl")
    parser.add_argument("--dev", default="data/eval/annual_report_dev_v2.jsonl")
    parser.add_argument("--no-answer", default="data/eval/annual_report_no_answer_v2.jsonl")
    parser.add_argument("--output", default="outputs/agent_dev_metrics_v2.json")
    parser.add_argument("--oracle-only", action="store_true")
    return parser.parse_args()


def project_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def evaluate_predictions(details: list[dict], no_answer_details: list[dict]) -> dict:
    answered = [row for row in details if row["status"] == "answered"]
    return {
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


def oracle_extraction(
    dev_rows: list[dict],
    chunk_by_id: dict[str, object],
    scope: CorpusScope,
) -> dict:
    details = []
    for row in dev_rows:
        route = route_query(row["query"], scope)
        gold_results = [
            SearchResult(
                chunk=chunk_by_id[chunk_id],
                score=1.0,
                rank=rank,
                source="gold_evidence",
            )
            for rank, chunk_id in enumerate(row["relevant_chunk_ids"], start=1)
        ]
        answer = answer_from_evidence(row["query"], route, gold_results, scope)
        details.append(
            {
                "query_id": row["query_id"],
                "status": answer.status,
                "predicted_answer": answer.answer,
                "expected_answer": row["answer"],
                "exact_match": answer.answer == row["answer"],
                "reason": answer.reason,
            }
        )
    return {
        "num_questions": len(details),
        "answer_rate": ratio(sum(row["status"] == "answered" for row in details), len(details)),
        "exact_answer_accuracy": ratio(sum(row["exact_match"] for row in details), len(details)),
        "details": details,
    }


def main() -> None:
    args = parse_args()
    if not 0 < args.top_k <= args.rerank_candidates <= args.candidate_k:
        raise ValueError("expected 0 < top-k <= rerank-candidates <= candidate-k")
    if not 0 <= args.minimum_reranker_score <= 1:
        raise ValueError("--minimum-reranker-score must be between 0 and 1")

    corpus_path = project_path(args.corpus)
    dev_path = project_path(args.dev)
    no_answer_path = project_path(args.no_answer)
    output_path = project_path(args.output)
    chunks = read_chunks(corpus_path)
    dev_rows = read_jsonl(dev_path)
    no_answer_rows = read_jsonl(no_answer_path)
    scope = CorpusScope.from_chunks(chunks)
    chunk_by_id = {chunk.chunk_id: chunk for chunk in chunks}
    oracle = oracle_extraction(dev_rows, chunk_by_id, scope)

    base_result = {
        "scope": {"companies": list(scope.companies), "years": list(scope.years)},
        "dataset": {
            "corpus": str(corpus_path),
            "dev": str(dev_path),
            "no_answer": str(no_answer_path),
            "num_chunks": len(chunks),
            "num_dev_queries": len(dev_rows),
            "num_no_answer_queries": len(no_answer_rows),
        },
        "frozen_test_evaluated": False,
        "oracle_extraction": oracle,
    }
    if args.oracle_only:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(base_result, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(json.dumps({**base_result["dataset"], **oracle}, ensure_ascii=False, indent=2))
        print("oracle_only=True")
        print(f"output={output_path}")
        return

    embedding_cache_path = project_path(args.embedding_cache)
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
    dense = DenseRetriever(chunks, encoder, document_embeddings=cached_embeddings)
    bm25 = BM25Retriever(chunks)
    routed_candidates = []
    for row in dev_rows:
        route = route_query(row["query"], scope)
        if route.should_refuse:
            routed_candidates.append((route, []))
            continue
        allowed = infer_allowed_document_ids(row["query"], chunks)
        dense_results = dense.search(row["query"], top_k=args.candidate_k, allowed_document_ids=allowed)
        bm25_query = expand_query_with_section_anchors(row["query"])
        bm25_results = bm25.search(
            bm25_query,
            top_k=args.candidate_k,
            allowed_document_ids=allowed,
        )
        hybrid = reciprocal_rank_fusion(
            [dense_results, bm25_results],
            top_k=args.candidate_k,
            weights=[1.0, args.bm25_weight],
        )
        routed_candidates.append((route, hybrid))
    retrieval_seconds = time.perf_counter() - retrieval_started
    embedding_peak_memory_mb = torch.cuda.max_memory_allocated() / 1024 / 1024

    del dense, encoder, bm25, cached_embeddings
    gc.collect()
    torch.cuda.empty_cache()
    memory_after_embedding_unload_mb = torch.cuda.memory_allocated() / 1024 / 1024

    torch.cuda.reset_peak_memory_stats()
    reranker_load_started = time.perf_counter()
    reranker = Qwen3Reranker(
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

    details = []
    rerank_latencies_ms = []
    for row, (route, candidates) in zip(dev_rows, routed_candidates):
        started = time.perf_counter()
        reranked = rerank_results(
            row["query"],
            candidates[: args.rerank_candidates],
            reranker,
            top_k=args.top_k,
        )
        rerank_latencies_ms.append((time.perf_counter() - started) * 1000)
        answer = answer_from_evidence(
            row["query"],
            route,
            reranked,
            scope,
            minimum_reranker_score=args.minimum_reranker_score,
        )
        reranked_ids = [result.chunk.chunk_id for result in reranked]
        citation_ids = [citation.chunk_id for citation in answer.citations]
        citation_pages = [citation.page for citation in answer.citations]
        details.append(
            {
                "query_id": row["query_id"],
                "query": row["query"],
                "status": answer.status,
                "reason": answer.reason,
                "predicted_answer": answer.answer,
                "expected_answer": row["answer"],
                "answer_exact_match": answer.answer == row["answer"],
                "relevant_chunk_ids": row["relevant_chunk_ids"],
                "reranked_chunk_ids": reranked_ids,
                "reranked_scores": [result.score for result in reranked],
                "all_gold_evidence_retrieved": all(
                    chunk_id in reranked_ids for chunk_id in row["relevant_chunk_ids"]
                ),
                "citation_chunk_ids": citation_ids,
                "citation_pages": citation_pages,
                "expected_pages": row["source_pages"],
                "citation_count": len(citation_ids),
                "correct_citation_count": sum(
                    chunk_id in row["relevant_chunk_ids"] for chunk_id in citation_ids
                ),
                "citation_pages_exact_match": sorted(citation_pages) == sorted(row["source_pages"]),
                "calculation": answer.calculation,
            }
        )

    no_answer_details = []
    for row in no_answer_rows:
        route = route_query(row["query"], scope)
        answer = answer_from_evidence(row["query"], route, [], scope)
        no_answer_details.append(
            {
                "query_id": row["query_id"],
                "status": answer.status,
                "reason": answer.reason,
                "expected_reason": row["reason"],
                "reason_match": answer.reason == row["reason"],
                "answer": answer.answer,
            }
        )

    result = {
        **base_result,
        "models": {"embedding": args.embedding_model, "reranker": args.reranker_model},
        "configuration": {
            "candidate_k": args.candidate_k,
            "rerank_candidates": args.rerank_candidates,
            "top_k": args.top_k,
            "bm25_weight": args.bm25_weight,
            "minimum_reranker_score": args.minimum_reranker_score,
            "embedding_max_length": args.embedding_max_length,
            "reranker_max_length": args.reranker_max_length,
        },
        "runtime": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "compiled_cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
            "embedding_peak_memory_mb": round(embedding_peak_memory_mb, 2),
            "memory_after_embedding_unload_mb": round(memory_after_embedding_unload_mb, 2),
            "reranker_peak_memory_mb": round(
                torch.cuda.max_memory_allocated() / 1024 / 1024, 2
            ),
        },
        "timing": {
            "retrieval_seconds": retrieval_seconds,
            "reranker_load_seconds": reranker_load_seconds,
            "mean_rerank_latency_ms": float(np.mean(rerank_latencies_ms)),
            "p50_rerank_latency_ms": float(np.percentile(rerank_latencies_ms, 50)),
            "p95_rerank_latency_ms": float(np.percentile(rerank_latencies_ms, 95)),
        },
        "metrics": evaluate_predictions(details, no_answer_details),
        "dev_details": details,
        "no_answer_details": no_answer_details,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result["runtime"], ensure_ascii=False, indent=2))
    print(json.dumps(result["timing"], ensure_ascii=False, indent=2))
    print(json.dumps(result["metrics"], ensure_ascii=False, indent=2))
    print(f"oracle_exact_answer_accuracy={oracle['exact_answer_accuracy']:.4f}")
    print("frozen_test_evaluated=False")
    print(f"output={output_path}")


if __name__ == "__main__":
    main()
