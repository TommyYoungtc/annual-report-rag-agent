from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("HF_HOME", str(ROOT / "cache" / "huggingface"))
sys.path.insert(0, str(ROOT / "src"))

from annual_report_agent.agent import (
    CorpusScope,
    answer_from_evidence,
    route_query,
)
from annual_report_agent.generation import (
    ContextPolicy,
    PromptVariant,
    build_context,
    build_prompt,
    choose_model,
)
from annual_report_agent.io_utils import read_chunks
from annual_report_agent.schemas import SearchResult
from annual_report_agent.telemetry import (
    HuggingFaceTokenCounter,
    ModelPrice,
    TokenUsage,
    UsageSource,
    calculate_cost_usd,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replay V4 token-efficiency profiles on old Dev without touching V3"
    )
    parser.add_argument("--config", default="configs/token_efficiency_v4.yaml")
    parser.add_argument("--output", default=None)
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def load_labels(config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    labels: dict[str, dict[str, Any]] = {}
    for group, key in (
        ("answerable_dev", "answerable_dev"),
        ("hard_no_answer_dev", "hard_no_answer_dev"),
    ):
        for row in read_jsonl(project_path(config["data"][key])):
            labels[row["query_id"]] = {
                "group": group,
                "expected_answer": row.get("answer"),
                "relevant_chunk_ids": tuple(row.get("relevant_chunk_ids", ())),
                "expected_pages": tuple(row.get("source_pages", ())),
            }
    return labels


def reconstruct_results(
    row: dict[str, Any],
    chunks_by_id: dict[str, Any],
) -> list[SearchResult]:
    score_by_id: dict[str, float] = {}
    for step in row.get("trajectory", ()):
        score_by_id.update(zip(step["evidence_chunk_ids"], step["evidence_scores"], strict=True))
    return [
        SearchResult(
            chunk=chunks_by_id[chunk_id],
            score=float(score_by_id.get(chunk_id, 0.0)),
            rank=rank,
            source="reranker",
        )
        for rank, chunk_id in enumerate(row["final_evidence_chunk_ids"], start=1)
        if chunk_id in chunks_by_id
    ]


def model_price(model_config: dict[str, Any]) -> ModelPrice:
    return ModelPrice(
        input_per_million_usd=model_config.get("input_per_million_usd"),
        output_per_million_usd=model_config.get("output_per_million_usd"),
        cached_input_per_million_usd=model_config.get("cached_input_per_million_usd"),
    )


def evaluate_profile(
    *,
    profile_name: str,
    profile: dict[str, Any],
    source_rows: list[dict[str, Any]],
    labels: dict[str, dict[str, Any]],
    chunks_by_id: dict[str, Any],
    scope: CorpusScope,
    counter: HuggingFaceTokenCounter,
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    details: list[dict[str, Any]] = []
    context_policy = ContextPolicy(profile["context_policy"])
    prompt_variant = PromptVariant(profile["prompt"])
    weights = config["normalized_cost"]
    for source in source_rows:
        query_id = source["query_id"]
        if query_id not in labels:
            continue
        label = labels[query_id]
        query = source["query"]
        full_results = reconstruct_results(source, chunks_by_id)
        context = build_context(
            query,
            full_results,
            counter=counter,
            policy=context_policy,
            top_k=int(profile["context_top_k"]),
            max_tokens=(
                int(profile["context_max_tokens"])
                if profile.get("context_max_tokens") is not None
                else None
            ),
            window_chars=int(profile.get("context_window_chars", 420)),
        )
        route = route_query(query, scope)
        answer = answer_from_evidence(query, route, list(context.results), scope)
        prompt = build_prompt(query, context.text, variant=prompt_variant)
        rounds = int(source["usage"]["retrieval_rounds"])
        decision = choose_model(
            query,
            policy=profile["model_policy"],
            retrieval_rounds=rounds,
            evidence_count=len(context.results),
        )
        model_config = config["models"][decision.model_tier]
        input_tokens = counter.count_messages(prompt.messages)
        static_tokens = counter.count_text(prompt.system_prompt)
        query_tokens = counter.count_text(query)
        output_proxy = json.dumps(
            {
                "status": answer.status,
                "answer": answer.answer,
                "reason": answer.reason,
                "citations": [
                    {
                        "chunk_id": citation.chunk_id,
                        "page": citation.page,
                        "quote": citation.quote,
                    }
                    for citation in answer.citations
                ],
                "calculation": answer.calculation,
            },
            ensure_ascii=False,
            separators=(",", ":"),
        )
        output_tokens = counter.count_text(output_proxy)
        full_context = build_context(
            query,
            full_results,
            counter=counter,
            policy=ContextPolicy.FULL,
            top_k=10,
        )
        pre_prompt = build_prompt(query, full_context.text, variant=prompt_variant)
        pre_truncation_tokens = counter.count_messages(pre_prompt.messages)
        price = model_price(model_config)
        usage = TokenUsage(
            stage="generator_replay",
            model=model_config["model_id"],
            source=UsageSource.TOKENIZER,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            prompt_static_tokens=static_tokens,
            query_tokens=query_tokens,
            evidence_tokens_before=context.evidence_tokens_before,
            evidence_tokens_after=context.evidence_tokens_after,
            pre_truncation_tokens=pre_truncation_tokens,
            post_truncation_tokens=input_tokens,
            cost_usd=calculate_cost_usd(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                price=price,
            ),
            metadata={
                "output_is_deterministic_answer_proxy": True,
                "provider_was_called": False,
            },
        )
        normalized_cost = float(model_config["relative_cost_multiplier"]) * (
            input_tokens * float(weights["input_token_weight"])
            + output_tokens * float(weights["output_token_weight"])
        )
        citations = [citation.chunk_id for citation in answer.citations]
        pages = [citation.page for citation in answer.citations]
        relevant = set(label["relevant_chunk_ids"])
        correct_citations = sum(chunk_id in relevant for chunk_id in citations)
        details.append(
            {
                "profile": profile_name,
                "query_id": query_id,
                "query": query,
                "group": label["group"],
                "model_tier": decision.model_tier,
                "model_reason": decision.reason,
                "model_complexity_score": decision.complexity_score,
                "prompt_variant": prompt.variant.value,
                "prompt_sha256": prompt.prompt_sha256,
                "context_policy": context.policy.value,
                "context_chunks_before": context.chunks_before,
                "context_chunks_after": context.chunks_after,
                "predicted_answer": answer.answer,
                "expected_answer": label["expected_answer"],
                "status": answer.status,
                "answer_exact_match": answer.answer == label["expected_answer"],
                "refusal_correct": (
                    label["group"] == "hard_no_answer_dev" and answer.status == "refused"
                ),
                "all_gold_evidence_retrieved": all(
                    chunk_id in {result.chunk.chunk_id for result in context.results}
                    for chunk_id in relevant
                ),
                "citation_chunk_ids": citations,
                "citation_pages": pages,
                "citation_count": len(citations),
                "correct_citation_count": correct_citations,
                "citation_pages_exact_match": sorted(page for page in pages if page is not None)
                == sorted(label["expected_pages"]),
                "retrieval_rounds": rounds,
                "normalized_cost_units": normalized_cost,
                "token_usage": usage.to_dict(),
            }
        )
    return details


def aggregate(details: list[dict[str, Any]]) -> dict[str, Any]:
    answerable = [row for row in details if row["group"] == "answerable_dev"]
    no_answer = [row for row in details if row["group"] == "hard_no_answer_dev"]
    correct = sum(row["answer_exact_match"] for row in answerable) + sum(
        row["refusal_correct"] for row in no_answer
    )
    usage = [row["token_usage"] for row in details]
    input_tokens = sum(row["input_tokens"] for row in usage)
    output_tokens = sum(row["output_tokens"] for row in usage)
    before = sum(row["evidence_tokens_before"] for row in usage)
    after = sum(row["evidence_tokens_after"] for row in usage)
    costs = [row["cost_usd"] for row in usage]
    known_cost = all(value is not None for value in costs)
    total_usd = sum(float(value) for value in costs) if known_cost else None
    normalized = sum(float(row["normalized_cost_units"]) for row in details)
    citation_total = sum(row["citation_count"] for row in answerable)
    correct_citation_total = sum(row["correct_citation_count"] for row in answerable)
    quality = {
        "exact_answer_accuracy": ratio(
            sum(row["answer_exact_match"] for row in answerable), len(answerable)
        ),
        "hard_no_answer_refusal_accuracy": ratio(
            sum(row["refusal_correct"] for row in no_answer), len(no_answer)
        ),
        "all_gold_evidence_recall": ratio(
            sum(row["all_gold_evidence_retrieved"] for row in answerable),
            len(answerable),
        ),
        "citation_chunk_precision": ratio(correct_citation_total, citation_total),
        "citation_page_set_accuracy": ratio(
            sum(row["citation_pages_exact_match"] for row in answerable),
            len(answerable),
        ),
    }
    return {
        "tasks": len(details),
        "correct_tasks": correct,
        "quality": quality,
        "routing": {
            "small": sum(row["model_tier"] == "small" for row in details),
            "large": sum(row["model_tier"] == "large" for row in details),
        },
        "tokens": {
            "input": input_tokens,
            "output_proxy": output_tokens,
            "total_proxy": input_tokens + output_tokens,
            "mean_input": ratio(input_tokens, len(details)),
            "mean_output_proxy": ratio(output_tokens, len(details)),
            "evidence_before": before,
            "evidence_after": after,
            "context_compression_ratio": ratio(after, before),
            "per_correct_answer": ratio(input_tokens + output_tokens, correct),
        },
        "cost": {
            "normalized_total_units": normalized,
            "normalized_per_correct_answer": ratio(normalized, correct),
            "estimated_usd": total_usd,
            "estimated_usd_per_correct_answer": (
                ratio(total_usd, correct) if total_usd is not None else None
            ),
        },
    }


def validate_gates(
    config: dict[str, Any],
    metrics: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    baseline = metrics["baseline"]["quality"]
    gates = config["quality_gates"]
    checks: dict[str, bool] = {}
    mapping = {
        "exact_answer_accuracy": "exact_answer_accuracy_drop_max",
        "hard_no_answer_refusal_accuracy": ("hard_no_answer_refusal_accuracy_drop_max"),
        "all_gold_evidence_recall": "all_gold_evidence_recall_drop_max",
        "citation_chunk_precision": "citation_chunk_precision_drop_max",
        "citation_page_set_accuracy": "citation_page_set_accuracy_drop_max",
    }
    for profile_name, values in metrics.items():
        if profile_name == "baseline":
            continue
        for metric_name, gate_name in mapping.items():
            checks[f"{profile_name}.{metric_name}"] = values["quality"][metric_name] >= baseline[
                metric_name
            ] - float(gates[gate_name])
    return {"passed": all(checks.values()), "checks": checks}


def main() -> None:
    args = parse_args()
    config_path = project_path(args.config)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    contract = config["experiment_contract"]
    if contract["frozen_test_evaluated"]:
        raise RuntimeError("V4 replay must never evaluate a frozen test")
    if contract["evaluation_split"] != "old_dev_only":
        raise RuntimeError("V4 replay is restricted to old Dev")

    source_path = project_path(contract["source_trajectory"])
    source_payload = json.loads(source_path.read_text(encoding="utf-8"))
    if source_payload.get("frozen_test_evaluated") is not False:
        raise RuntimeError("source trajectory is not a development-only artifact")
    source_rows = source_payload["details"][contract["source_system"]]
    if args.limit is not None:
        source_rows = [row for row in source_rows if row["group"] != "known_failure_canary"][
            : args.limit
        ]
    else:
        source_rows = [row for row in source_rows if row["group"] != "known_failure_canary"]

    labels = load_labels(config)
    chunks = read_chunks(project_path(config["data"]["corpus"]))
    chunks_by_id = {chunk.chunk_id: chunk for chunk in chunks}
    scope = CorpusScope.from_chunks(chunks)
    counter = HuggingFaceTokenCounter(str(project_path(config["tokenizer"]["model"])))

    results: dict[str, list[dict[str, Any]]] = {}
    metrics: dict[str, dict[str, Any]] = {}
    for profile_name, profile in config["profiles"].items():
        details = evaluate_profile(
            profile_name=profile_name,
            profile=profile,
            source_rows=source_rows,
            labels=labels,
            chunks_by_id=chunks_by_id,
            scope=scope,
            counter=counter,
            config=config,
        )
        results[profile_name] = details
        metrics[profile_name] = aggregate(details)

    gates = validate_gates(config, metrics)
    baseline_cost = metrics["baseline"]["cost"]["normalized_total_units"]
    baseline_tokens = metrics["baseline"]["tokens"]["total_proxy"]
    comparisons = {
        profile: {
            "token_reduction_vs_baseline": 1
            - ratio(values["tokens"]["total_proxy"], baseline_tokens),
            "normalized_cost_reduction_vs_baseline": 1
            - ratio(values["cost"]["normalized_total_units"], baseline_cost),
        }
        for profile, values in metrics.items()
    }
    paths_to_hash = {
        "config": config_path,
        "source_trajectory": source_path,
        "corpus": project_path(config["data"]["corpus"]),
        "answerable_dev": project_path(config["data"]["answerable_dev"]),
        "hard_no_answer_dev": project_path(config["data"]["hard_no_answer_dev"]),
    }
    payload = {
        "configuration_id": config["configuration_id"],
        "status": "development_replay",
        "evaluation_split": contract["evaluation_split"],
        "frozen_test_evaluated": False,
        "llm_generation_validated": False,
        "provider_called": False,
        "claim_scope": contract["claim_scope"],
        "runtime": {
            "python": platform.python_version(),
            "tokenizer": config["tokenizer"]["model"],
            "counting_method": config["tokenizer"]["counting_method"],
        },
        "artifact_sha256": {key: sha256(path) for key, path in paths_to_hash.items()},
        "metrics": metrics,
        "comparisons": comparisons,
        "quality_gates": gates,
        "details": results,
    }
    output_path = project_path(args.output or config["output"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "metrics": metrics,
                "comparisons": comparisons,
                "quality_gates": gates,
                "llm_generation_validated": False,
                "output": str(output_path),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
