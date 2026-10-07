from __future__ import annotations

import argparse
import gc
import json
import os
import platform
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("HF_HOME", str(ROOT / "cache" / "huggingface"))
os.environ.setdefault("TORCH_HOME", str(ROOT / "cache" / "torch"))
sys.path.insert(0, str(ROOT / "src"))

from annual_report_agent.agent import (
    CorpusScope,
    EvidenceVerifier,
    PolicyConfig,
    RetrievalPolicy,
    RetrievalStep,
    RoundCost,
    RuleBasedQueryRewriter,
    answer_from_evidence,
    decide_next_action,
    merge_ranked_results,
    route_query,
    summarize_usage,
)
from annual_report_agent.io_utils import read_chunks
from annual_report_agent.retrieval import (
    BM25Retriever,
    Qwen3Reranker,
    expand_query_with_section_anchors,
    infer_allowed_document_ids,
    reciprocal_rank_fusion,
    rerank_results,
)
from annual_report_agent.retrieval.dense import (
    DenseRetriever,
    SentenceTransformerEncoder,
)
from annual_report_agent.schemas import SearchResult


@dataclass(frozen=True, slots=True)
class EvalItem:
    query_id: str
    query: str
    group: str
    question_type: str
    expected_answer: str | None
    relevant_chunk_ids: tuple[str, ...]
    expected_pages: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class RoundRecord:
    search_query: str
    results: tuple[SearchResult, ...]
    cost: RoundCost


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare Static, Always-Agentic and Adaptive retrieval on development data"
    )
    parser.add_argument("--config", default="configs/adaptive_retrieval_v1.yaml")
    parser.add_argument("--output", default=None)
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def percentile(values: list[float], value: int) -> float:
    return float(np.percentile(values, value)) if values else 0.0


def ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def load_items(config: dict[str, Any], limit: int | None) -> list[EvalItem]:
    evaluation = config["development_evaluation"]
    answerable_rows = read_jsonl(project_path(evaluation["answerable_dev"]))
    no_answer_rows = read_jsonl(project_path(evaluation["hard_no_answer_dev"]))
    if limit is not None:
        answerable_rows = answerable_rows[:limit]
        no_answer_rows = no_answer_rows[:limit]
    items = [
        EvalItem(
            query_id=row["query_id"],
            query=row["query"],
            group="answerable_dev",
            question_type=row.get("question_type", "unknown"),
            expected_answer=row["answer"],
            relevant_chunk_ids=tuple(row["relevant_chunk_ids"]),
            expected_pages=tuple(row["source_pages"]),
        )
        for row in answerable_rows
    ]
    items.extend(
        EvalItem(
            query_id=row["query_id"],
            query=row["query"],
            group="hard_no_answer_dev",
            question_type=row.get("constraint_type", "hard_no_answer"),
            expected_answer=None,
            relevant_chunk_ids=(),
            expected_pages=(),
        )
        for row in no_answer_rows
    )
    canary_ids = set(evaluation.get("known_failure_canary_ids", ()))
    if canary_ids:
        canary_rows = read_jsonl(project_path(evaluation["known_failure_canary"]))
        items.extend(
            EvalItem(
                query_id=row["query_id"],
                query=row["query"],
                group="known_failure_canary",
                question_type=row.get("question_type", "unknown"),
                expected_answer=row["answer"],
                relevant_chunk_ids=tuple(row["relevant_chunk_ids"]),
                expected_pages=tuple(row["source_pages"]),
            )
            for row in canary_rows
            if row["query_id"] in canary_ids
        )
    return items


def build_policies(config: dict[str, Any]) -> dict[str, PolicyConfig]:
    policies = {}
    for key, value in config["systems"].items():
        policies[key] = PolicyConfig(
            name=value["name"],
            policy=RetrievalPolicy(value["policy"]),
            max_rounds=int(value["max_rounds"]),
        )
    return policies


def load_embedding_cache(
    config: dict[str, Any],
    chunks: list[Any],
) -> np.ndarray:
    base = config["base_pipeline"]
    cache_path = project_path(base["embedding_cache"])
    model_path = project_path(base["embedding_model"])
    with np.load(cache_path, allow_pickle=False) as cache:
        cached_ids = cache["chunk_ids"].astype(str).tolist()
        cached_model = Path(str(cache["model"].item()))
        cached_length = int(cache["max_length"].item())
        if cached_ids != [chunk.chunk_id for chunk in chunks]:
            raise ValueError("embedding cache does not match corpus chunk IDs")
        if not cached_model.is_absolute():
            cached_model = ROOT / cached_model
        if cached_model.resolve() != model_path.resolve():
            raise ValueError("embedding cache model does not match configuration")
        if cached_length != int(base["embedding_max_length"]):
            raise ValueError("embedding cache max_length does not match configuration")
        return np.asarray(cache["embeddings"], dtype=np.float32)


def build_search_queries(
    items: list[EvalItem],
    routes: dict[str, Any],
    max_rounds: int,
) -> dict[int, dict[str, str]]:
    rewriter = RuleBasedQueryRewriter()
    queries: dict[int, dict[str, str]] = {}
    for round_index in range(1, max_rounds + 1):
        queries[round_index] = {}
        for item in items:
            queries[round_index][item.query_id] = rewriter.rewrite(
                item.query,
                routes[item.query_id],
                round_index=round_index,
            )
    return queries


def execute_round(
    *,
    round_index: int,
    items: list[EvalItem],
    routes: dict[str, Any],
    search_queries: dict[str, str],
    chunks: list[Any],
    document_embeddings: np.ndarray,
    bm25: BM25Retriever,
    config: dict[str, Any],
) -> tuple[dict[str, RoundRecord], dict[str, float]]:
    import torch

    base = config["base_pipeline"]
    active = [item for item in items if not routes[item.query_id].should_refuse]
    if not active:
        return {}, {}

    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    embedding_load_started = time.perf_counter()
    encoder = SentenceTransformerEncoder(
        str(project_path(base["embedding_model"])),
        device=base["device"],
        batch_size=int(base["embedding_batch_size"]),
        max_length=int(base["embedding_max_length"]),
        query_instruction=base["embedding_query_instruction"],
        query_template=base["embedding_query_template"],
    )
    dense = DenseRetriever(chunks, encoder, document_embeddings=document_embeddings)
    embedding_load_ms = (time.perf_counter() - embedding_load_started) * 1000
    candidate_rows: dict[str, tuple[list[SearchResult], float]] = {}
    for item in active:
        started = time.perf_counter()
        search_query = search_queries[item.query_id]
        allowed = infer_allowed_document_ids(item.query, chunks)
        dense_results = dense.search(
            search_query,
            top_k=int(base["candidate_k"]),
            allowed_document_ids=allowed,
        )
        bm25_results = bm25.search(
            expand_query_with_section_anchors(search_query),
            top_k=int(base["candidate_k"]),
            allowed_document_ids=allowed,
        )
        candidates = reciprocal_rank_fusion(
            [dense_results, bm25_results],
            top_k=int(base["candidate_k"]),
            weights=[1.0, float(base["bm25_weight"])],
        )
        candidate_rows[item.query_id] = (
            candidates,
            (time.perf_counter() - started) * 1000,
        )
    embedding_peak_memory_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
    del dense, encoder
    gc.collect()
    torch.cuda.empty_cache()

    torch.cuda.reset_peak_memory_stats()
    reranker_load_started = time.perf_counter()
    reranker = Qwen3Reranker(
        str(project_path(base["reranker_model"])),
        device=base["device"],
        batch_size=int(base["reranker_batch_size"]),
        max_length=int(base["reranker_max_length"]),
        instruction=(
            "Given a Chinese annual report question, judge whether the passage "
            "directly contains evidence that answers the question."
        ),
    )
    reranker_load_ms = (time.perf_counter() - reranker_load_started) * 1000
    records: dict[str, RoundRecord] = {}
    per_item_embedding_load = embedding_load_ms / len(active)
    per_item_reranker_load = reranker_load_ms / len(active)
    for item in active:
        candidates, retrieval_ms = candidate_rows[item.query_id]
        started = time.perf_counter()
        reranked = rerank_results(
            item.query,
            candidates[: int(base["rerank_candidates"])],
            reranker,
            top_k=int(base["top_k"]),
        )
        rerank_ms = (time.perf_counter() - started) * 1000
        cost = RoundCost(
            retrieval_ms=retrieval_ms,
            rerank_ms=rerank_ms,
            amortized_embedding_load_ms=per_item_embedding_load,
            amortized_reranker_load_ms=per_item_reranker_load,
            dense_queries=1,
            bm25_queries=1,
            reranker_calls=1,
            candidates_retrieved=len(candidates),
            passages_reranked=min(len(candidates), int(base["rerank_candidates"])),
        )
        records[item.query_id] = RoundRecord(
            search_query=search_queries[item.query_id],
            results=tuple(reranked),
            cost=cost,
        )
    reranker_peak_memory_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
    del reranker
    gc.collect()
    torch.cuda.empty_cache()
    return records, {
        "round_index": float(round_index),
        "queries": float(len(active)),
        "embedding_load_ms": embedding_load_ms,
        "reranker_load_ms": reranker_load_ms,
        "embedding_peak_memory_mb": embedding_peak_memory_mb,
        "reranker_peak_memory_mb": reranker_peak_memory_mb,
    }


def evaluate_item(
    item: EvalItem,
    route: Any,
    scope: CorpusScope,
    policy: PolicyConfig,
    round_records: dict[int, dict[str, RoundRecord]],
    verifier: EvidenceVerifier,
    minimum_reranker_score: float,
) -> dict[str, Any]:
    if route.should_refuse:
        answer = answer_from_evidence(item.query, route, [], scope)
        return {
            "query_id": item.query_id,
            "query": item.query,
            "group": item.group,
            "question_type": item.question_type,
            "status": answer.status,
            "reason": answer.reason,
            "predicted_answer": answer.answer,
            "expected_answer": item.expected_answer,
            "answer_exact_match": answer.answer == item.expected_answer,
            "refusal_correct": item.group == "hard_no_answer_dev" and answer.status == "refused",
            "all_gold_evidence_retrieved": not item.relevant_chunk_ids,
            "citation_chunk_ids": [],
            "citation_pages": [],
            "citation_count": 0,
            "correct_citation_count": 0,
            "citation_pages_exact_match": not item.expected_pages,
            "final_evidence_chunk_ids": [],
            "trajectory": [],
            "usage": summarize_usage([]).to_dict(),
        }

    merged: list[SearchResult] = []
    steps: list[RetrievalStep] = []
    answer = None
    reject_quarter_evidence = False
    for round_index in range(1, policy.max_rounds + 1):
        record = round_records[round_index][item.query_id]
        merged = merge_ranked_results(merged, list(record.results))
        if reject_quarter_evidence:
            merged = verifier.prepare_escalated_evidence(
                item.query,
                route,
                merged,
            )
        answer = answer_from_evidence(
            item.query,
            route,
            merged,
            scope,
            minimum_reranker_score=minimum_reranker_score,
        )
        assessment = verifier.assess(
            item.query,
            route,
            merged,
            answer,
            round_index=round_index,
        )
        reject_quarter_evidence = reject_quarter_evidence or (
            "annual_quarter_mismatch" in assessment.reasons
        )
        decision = decide_next_action(policy, assessment, round_index=round_index)
        steps.append(
            RetrievalStep(
                round_index=round_index,
                search_query=record.search_query,
                evidence_chunk_ids=tuple(result.chunk.chunk_id for result in merged),
                evidence_scores=tuple(float(result.score) for result in merged),
                answer_status=answer.status,
                answer_reason=answer.reason,
                assessment_status=assessment.status,
                assessment_reasons=assessment.reasons,
                action=decision.action,
                action_reason=decision.reason,
                cost=record.cost,
            )
        )
        if decision.action == "stop":
            break
    if answer is None:
        raise RuntimeError(f"no answer produced for {item.query_id}")

    citation_ids = [citation.chunk_id for citation in answer.citations]
    citation_pages = [citation.page for citation in answer.citations]
    final_ids = [result.chunk.chunk_id for result in merged]
    return {
        "query_id": item.query_id,
        "query": item.query,
        "group": item.group,
        "question_type": item.question_type,
        "status": answer.status,
        "reason": answer.reason,
        "predicted_answer": answer.answer,
        "expected_answer": item.expected_answer,
        "answer_exact_match": answer.answer == item.expected_answer,
        "refusal_correct": item.group == "hard_no_answer_dev" and answer.status == "refused",
        "all_gold_evidence_retrieved": all(
            chunk_id in final_ids for chunk_id in item.relevant_chunk_ids
        ),
        "citation_chunk_ids": citation_ids,
        "citation_pages": citation_pages,
        "citation_count": len(citation_ids),
        "correct_citation_count": sum(
            chunk_id in item.relevant_chunk_ids for chunk_id in citation_ids
        ),
        "citation_pages_exact_match": sorted(page for page in citation_pages if page is not None)
        == sorted(item.expected_pages),
        "final_evidence_chunk_ids": final_ids,
        "trajectory": [step.to_dict() for step in steps],
        "usage": summarize_usage(steps).to_dict(),
    }


def quality_metrics(details: list[dict[str, Any]]) -> dict[str, Any]:
    answerable = [row for row in details if row["group"] == "answerable_dev"]
    no_answer = [row for row in details if row["group"] == "hard_no_answer_dev"]
    correct_citations = sum(row["correct_citation_count"] for row in answerable)
    all_citations = sum(row["citation_count"] for row in answerable)
    return {
        "answerable_questions": len(answerable),
        "exact_answer_accuracy": ratio(
            sum(row["answer_exact_match"] for row in answerable), len(answerable)
        ),
        "all_gold_evidence_recall_at_10": ratio(
            sum(row["all_gold_evidence_retrieved"] for row in answerable), len(answerable)
        ),
        "citation_chunk_precision": ratio(correct_citations, all_citations),
        "citation_page_set_accuracy": ratio(
            sum(row["citation_pages_exact_match"] for row in answerable), len(answerable)
        ),
        "hard_no_answer_questions": len(no_answer),
        "hard_no_answer_refusal_accuracy": ratio(
            sum(row["refusal_correct"] for row in no_answer), len(no_answer)
        ),
    }


def budget_metrics(details: list[dict[str, Any]]) -> dict[str, Any]:
    development = [row for row in details if row["group"] != "known_failure_canary"]
    usages = [row["usage"] for row in development]
    latencies = [float(usage["amortized_wall_clock_ms"]) for usage in usages]
    correct = sum(
        row["answer_exact_match"]
        if row["group"] == "answerable_dev"
        else row["refusal_correct"]
        for row in development
    )
    totals = {
        key: sum(float(usage[key]) for usage in usages)
        for key in (
            "dense_queries",
            "bm25_queries",
            "reranker_calls",
            "candidates_retrieved",
            "passages_reranked",
            "planner_input_tokens",
            "planner_output_tokens",
            "estimated_api_cost_usd",
            "amortized_wall_clock_ms",
        )
    }
    return {
        "tasks": len(development),
        "correct_tasks": correct,
        "average_retrieval_rounds": float(
            np.mean([usage["retrieval_rounds"] for usage in usages])
        ),
        "escalation_rate": ratio(
            sum(usage["retrieval_rounds"] > 1 for usage in usages), len(usages)
        ),
        "amortized_mean_latency_ms": float(np.mean(latencies)) if latencies else 0.0,
        "amortized_p50_latency_ms": percentile(latencies, 50),
        "amortized_p95_latency_ms": percentile(latencies, 95),
        "totals": totals,
        "cost_per_correct_answer": {
            "estimated_api_cost_usd": ratio(totals["estimated_api_cost_usd"], correct),
            "amortized_latency_ms": ratio(totals["amortized_wall_clock_ms"], correct),
            "passages_reranked": ratio(totals["passages_reranked"], correct),
        },
    }


def add_comparative_routing_metrics(
    results: dict[str, list[dict[str, Any]]],
    metrics: dict[str, dict[str, Any]],
) -> None:
    static = {row["query_id"]: row for row in results["A"]}
    static_dev = {
        query_id: row
        for query_id, row in static.items()
        if row["group"] != "known_failure_canary"
    }
    for key, rows in results.items():
        development = [row for row in rows if row["group"] != "known_failure_canary"]
        unnecessary_population = [
            row
            for row in development
            if (
                static_dev[row["query_id"]]["answer_exact_match"]
                if row["group"] == "answerable_dev"
                else static_dev[row["query_id"]]["refusal_correct"]
            )
        ]
        unnecessary = sum(
            row["usage"]["retrieval_rounds"] > 1 for row in unnecessary_population
        )
        static_failures = [
            row
            for row in development
            if not (
                static_dev[row["query_id"]]["answer_exact_match"]
                if row["group"] == "answerable_dev"
                else static_dev[row["query_id"]]["refusal_correct"]
            )
        ]
        missed = sum(row["usage"]["retrieval_rounds"] == 1 for row in static_failures)
        canaries = [row for row in rows if row["group"] == "known_failure_canary"]
        recoverable = [
            row
            for row in canaries
            if not static[row["query_id"]]["answer_exact_match"]
        ]
        recovered = sum(row["answer_exact_match"] for row in recoverable)
        metrics[key]["routing"] = {
            "unnecessary_escalation_rate": ratio(
                unnecessary, len(unnecessary_population)
            ),
            "missed_escalation_rate": ratio(missed, len(static_failures)),
            "known_failure_canary_recovery_rate": ratio(recovered, len(recoverable)),
            "known_failure_canary_recovered": f"{recovered}/{len(recoverable)}",
        }


def validate_development_gates(
    config: dict[str, Any],
    metrics: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    gates = config["development_gates"]
    checks = {
        "static_answerable_exact_accuracy": metrics["A"]["quality"][
            "exact_answer_accuracy"
        ]
        >= float(gates["static_answerable_exact_accuracy_min"]),
        "static_no_answer_refusal_accuracy": metrics["A"]["quality"][
            "hard_no_answer_refusal_accuracy"
        ]
        >= float(gates["static_no_answer_refusal_accuracy_min"]),
        "always_agentic_answerable_exact_accuracy": metrics["B"]["quality"][
            "exact_answer_accuracy"
        ]
        >= float(gates["always_agentic_answerable_exact_accuracy_min"]),
        "adaptive_answerable_exact_accuracy": metrics["C"]["quality"][
            "exact_answer_accuracy"
        ]
        >= float(gates["adaptive_answerable_exact_accuracy_min"]),
        "adaptive_no_answer_refusal_accuracy": metrics["C"]["quality"][
            "hard_no_answer_refusal_accuracy"
        ]
        >= float(gates["adaptive_no_answer_refusal_accuracy_min"]),
        "adaptive_average_rounds_below_always_agentic": metrics["C"]["budget"][
            "average_retrieval_rounds"
        ]
        < metrics["B"]["budget"]["average_retrieval_rounds"],
    }
    return {"passed": all(checks.values()), "checks": checks}


def main() -> None:
    args = parse_args()
    config_path = project_path(args.config)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    items = load_items(config, args.limit)
    base = config["base_pipeline"]
    chunks = read_chunks(project_path(base["corpus"]))
    scope = CorpusScope.from_chunks(chunks)
    routes = {item.query_id: route_query(item.query, scope) for item in items}
    policies = build_policies(config)
    max_rounds = max(policy.max_rounds for policy in policies.values())
    search_queries = build_search_queries(items, routes, max_rounds)
    document_embeddings = load_embedding_cache(config, chunks)
    bm25 = BM25Retriever(chunks)

    round_records: dict[int, dict[str, RoundRecord]] = {}
    round_runtime: list[dict[str, float]] = []
    experiment_started = time.perf_counter()
    for round_index in range(1, max_rounds + 1):
        records, runtime = execute_round(
            round_index=round_index,
            items=items,
            routes=routes,
            search_queries=search_queries[round_index],
            chunks=chunks,
            document_embeddings=document_embeddings,
            bm25=bm25,
            config=config,
        )
        round_records[round_index] = records
        round_runtime.append(runtime)

    verifier = EvidenceVerifier(
        refusal_confirmation_rounds=int(
            config["orchestration"]["refusal_confirmation_rounds"]
        )
    )
    results: dict[str, list[dict[str, Any]]] = {}
    metrics: dict[str, dict[str, Any]] = {}
    for key, policy in policies.items():
        details = [
            evaluate_item(
                item,
                routes[item.query_id],
                scope,
                policy,
                round_records,
                verifier,
                float(base["minimum_reranker_score"]),
            )
            for item in items
        ]
        results[key] = details
        metrics[key] = {
            "system": asdict(policy),
            "quality": quality_metrics(details),
            "budget": budget_metrics(details),
        }
    add_comparative_routing_metrics(results, metrics)
    gates = validate_development_gates(config, metrics)

    import torch

    output_path = project_path(
        args.output or config["development_evaluation"]["output"]
    )
    payload = {
        "configuration_id": config["configuration_id"],
        "evaluation_split": "development_and_contaminated_canary",
        "frozen_test_evaluated": False,
        "known_failure_canary_in_aggregate_metrics": False,
        "planner": {
            "type": config["orchestration"]["query_rewriter"],
            "model": config["orchestration"]["planner_model"],
            "input_tokens": 0,
            "output_tokens": 0,
            "estimated_api_cost_usd": 0.0,
        },
        "dataset": {
            "answerable_dev": sum(item.group == "answerable_dev" for item in items),
            "hard_no_answer_dev": sum(
                item.group == "hard_no_answer_dev" for item in items
            ),
            "known_failure_canary": sum(
                item.group == "known_failure_canary" for item in items
            ),
            "corpus_chunks": len(chunks),
        },
        "runtime": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "compiled_cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0)
            if torch.cuda.is_available()
            else None,
            "experiment_wall_clock_seconds": time.perf_counter() - experiment_started,
            "rounds": round_runtime,
            "execution_reused_across_policies": True,
            "latency_definition": (
                "Per-policy effective latency sums only rounds the policy would use; "
                "model load time is amortized over the batch for each round."
            ),
        },
        "metrics": metrics,
        "development_gates": gates,
        "details": results,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    summary = {
        key: {
            "quality": value["quality"],
            "average_retrieval_rounds": value["budget"]["average_retrieval_rounds"],
            "mean_latency_ms": value["budget"]["amortized_mean_latency_ms"],
            "routing": value["routing"],
        }
        for key, value in metrics.items()
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(json.dumps(gates, ensure_ascii=False, indent=2))
    print(f"frozen_test_evaluated={payload['frozen_test_evaluated']}")
    print(f"output={output_path}")


if __name__ == "__main__":
    main()
