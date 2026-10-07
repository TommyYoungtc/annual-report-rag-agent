from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import platform
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from annual_report_agent.agent import CorpusScope, EvidenceVerifier, route_query
from annual_report_agent.io_utils import read_chunks
from annual_report_agent.retrieval import BM25Retriever

CANDIDATE_LOCK = ROOT / "configs" / "adaptive_retrieval_v1.lock.json"
CONFIG = ROOT / "configs" / "adaptive_retrieval_v1.yaml"
FROZEN_MANIFEST = ROOT / "data" / "eval" / "annual_report_v3_frozen_manifest.json"
FROZEN_TEST = ROOT / "data" / "eval" / "annual_report_v3_frozen_test.jsonl"
OUTPUT = ROOT / "outputs" / "adaptive_retrieval_v3_frozen_test.json"
RESULT_SEAL = ROOT / "configs" / "adaptive_retrieval_v3_result.json"
LOCKED_EVALUATOR = ROOT / "scripts" / "run_adaptive_retrieval_evaluation.py"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def validate_candidate_lock() -> dict[str, Any]:
    lock = json.loads(CANDIDATE_LOCK.read_text(encoding="utf-8"))
    if lock["status"] != "candidate_locked_before_new_holdout_selection":
        raise ValueError("adaptive candidate is not in the expected locked state")
    for name, artifact in lock["artifacts"].items():
        path = ROOT / artifact["path"]
        if not path.is_file() or sha256(path) != artifact["sha256"]:
            raise ValueError(f"candidate artifact changed after lock: {name}")
    return lock


def validate_artifact(value: dict[str, Any], label: str) -> None:
    path = ROOT / value["path"]
    if not path.is_file():
        raise FileNotFoundError(f"frozen artifact missing: {label}: {path}")
    if path.stat().st_size != int(value["bytes"]) or sha256(path) != value["sha256"]:
        raise ValueError(f"frozen artifact changed after freeze: {label}")


def validate_frozen_manifest() -> dict[str, Any]:
    manifest = json.loads(FROZEN_MANIFEST.read_text(encoding="utf-8"))
    if manifest["status"] != "frozen_before_first_model_run":
        raise ValueError("V3 manifest is not in the expected pre-run frozen state")
    if manifest["successful_model_runs_at_freeze"] != 0:
        raise ValueError("V3 manifest records a prior model run")
    if manifest["maximum_successful_frozen_test_runs"] != 1:
        raise ValueError("V3 manifest does not enforce a one-run limit")
    for name, value in manifest["artifacts"].items():
        if name == "source_pdfs":
            for document_id, pdf_artifact in value.items():
                validate_artifact(pdf_artifact, f"source_pdfs.{document_id}")
        else:
            validate_artifact(value, name)
    return manifest


def load_locked_evaluator() -> Any:
    spec = importlib.util.spec_from_file_location("locked_adaptive_evaluator", LOCKED_EVALUATOR)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load the locked adaptive evaluator")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_items(module: Any) -> tuple[list[Any], dict[str, dict[str, Any]]]:
    rows = read_jsonl(FROZEN_TEST)
    items = []
    by_id = {}
    for row in rows:
        group = "answerable_dev" if row["expected_action"] == "answer" else "hard_no_answer_dev"
        items.append(
            module.EvalItem(
                query_id=row["query_id"],
                query=row["query"],
                group=group,
                question_type=row["category"],
                expected_answer=row["answer"],
                relevant_chunk_ids=tuple(row["relevant_chunk_ids"]),
                expected_pages=tuple(row["source_pages"]),
            )
        )
        by_id[row["query_id"]] = row
    return items, by_id


def category_metrics(
    details: list[dict[str, Any]],
    rows: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    categories = sorted({row["category"] for row in rows.values()})
    for category in categories:
        selected = [row for row in details if rows[row["query_id"]]["category"] == category]
        correct = sum(
            row["answer_exact_match"]
            if row["group"] == "answerable_dev"
            else row["refusal_correct"]
            for row in selected
        )
        metrics[category] = {
            "questions": len(selected),
            "correct": correct,
            "accuracy": correct / len(selected) if selected else 0.0,
        }
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run and immediately seal the one-shot Adaptive V3 frozen evaluation"
    )
    parser.add_argument("--output", default=str(OUTPUT.relative_to(ROOT)))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = Path(args.output)
    output_path = output_path if output_path.is_absolute() else ROOT / output_path
    if output_path.resolve() != OUTPUT.resolve():
        raise ValueError("the frozen runner only permits the declared output path")
    if OUTPUT.exists() or RESULT_SEAL.exists():
        raise FileExistsError(
            "V3 frozen result or result seal already exists; a second run is forbidden"
        )

    candidate_before = validate_candidate_lock()
    manifest = validate_frozen_manifest()
    manifest_sha_before = sha256(FROZEN_MANIFEST)
    module = load_locked_evaluator()
    items, rows_by_id = load_items(module)

    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    config["base_pipeline"] = dict(config["base_pipeline"])
    config["base_pipeline"]["corpus"] = manifest["artifacts"]["corpus"]["path"]
    config["base_pipeline"]["embedding_cache"] = manifest["artifacts"]["embedding_cache"]["path"]
    base = config["base_pipeline"]
    chunks = read_chunks(ROOT / base["corpus"])
    scope = CorpusScope.from_chunks(chunks)
    routes = {item.query_id: route_query(item.query, scope) for item in items}
    policies = module.build_policies(config)
    max_rounds = max(policy.max_rounds for policy in policies.values())
    search_queries = module.build_search_queries(items, routes, max_rounds)
    document_embeddings = module.load_embedding_cache(config, chunks)
    bm25 = BM25Retriever(chunks)

    round_records: dict[int, dict[str, Any]] = {}
    round_runtime: list[dict[str, float]] = []
    experiment_started = time.perf_counter()
    for round_index in range(1, max_rounds + 1):
        records, runtime = module.execute_round(
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
        refusal_confirmation_rounds=int(config["orchestration"]["refusal_confirmation_rounds"])
    )
    results: dict[str, list[dict[str, Any]]] = {}
    metrics: dict[str, dict[str, Any]] = {}
    for key, policy in policies.items():
        details = [
            module.evaluate_item(
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
        quality = module.quality_metrics(details)
        budget = module.budget_metrics(details)
        correct_tasks = int(budget["correct_tasks"])
        metrics[key] = {
            "system": asdict(policy),
            "quality": quality,
            "overall_task_accuracy": correct_tasks / len(items),
            "category_accuracy": category_metrics(details, rows_by_id),
            "budget": budget,
            "cost_per_correct_answer": {
                "estimated_api_cost_usd": budget["cost_per_correct_answer"][
                    "estimated_api_cost_usd"
                ],
                "amortized_latency_ms": budget["cost_per_correct_answer"]["amortized_latency_ms"],
                "passages_reranked": budget["cost_per_correct_answer"]["passages_reranked"],
            },
        }
    module.add_comparative_routing_metrics(results, metrics)

    import torch

    candidate_after = validate_candidate_lock()
    validate_frozen_manifest()
    if candidate_before != candidate_after or sha256(FROZEN_MANIFEST) != manifest_sha_before:
        raise ValueError("candidate lock or frozen manifest changed during evaluation")

    payload = {
        "configuration_id": config["configuration_id"],
        "evaluation_id": manifest["dataset_id"],
        "evaluation_split": "v3_independent_frozen_test",
        "successful_run_number": 1,
        "maximum_successful_runs": 1,
        "candidate_modified_after_lock": False,
        "result_policy": "sealed immediately; no post-result tuning permitted",
        "dataset": {
            "questions": len(items),
            "answerable": sum(item.group == "answerable_dev" for item in items),
            "no_answer": sum(item.group == "hard_no_answer_dev" for item in items),
            "companies": list(scope.companies),
            "years": list(scope.years),
            "corpus_chunks": len(chunks),
            "category_counts": manifest["category_counts"],
        },
        "runtime": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "compiled_cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "experiment_wall_clock_seconds": time.perf_counter() - experiment_started,
            "rounds": round_runtime,
            "execution_reused_across_policies": True,
            "latency_definition": (
                "Per-policy effective latency sums only rounds that policy would use; "
                "model load time is amortized over the batch for each round."
            ),
            "monetary_cost_note": (
                "All inference is local, so estimated API cost is USD 0. "
                "Latency and reranked passages are the reported compute-cost proxies."
            ),
        },
        "integrity": {
            "candidate_lock_sha256": sha256(CANDIDATE_LOCK),
            "frozen_manifest_sha256": manifest_sha_before,
            "frozen_test_sha256": sha256(FROZEN_TEST),
            "locked_evaluator_sha256": sha256(LOCKED_EVALUATOR),
        },
        "metrics": metrics,
        "details": results,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    seal = {
        "result_id": "annual-report-agent-adaptive-v3-single-run-result",
        "status": "sealed_after_single_successful_frozen_test_run",
        "successful_frozen_test_runs": 1,
        "maximum_successful_frozen_test_runs": 1,
        "candidate_modified_after_lock": False,
        "post_result_tuning_permitted": False,
        "candidate_lock": {
            "path": str(CANDIDATE_LOCK.relative_to(ROOT)).replace("\\", "/"),
            "sha256": sha256(CANDIDATE_LOCK),
        },
        "frozen_manifest": {
            "path": str(FROZEN_MANIFEST.relative_to(ROOT)).replace("\\", "/"),
            "sha256": manifest_sha_before,
        },
        "frozen_test": {
            "path": str(FROZEN_TEST.relative_to(ROOT)).replace("\\", "/"),
            "sha256": sha256(FROZEN_TEST),
        },
        "output": {
            "path": str(OUTPUT.relative_to(ROOT)).replace("\\", "/"),
            "bytes": OUTPUT.stat().st_size,
            "sha256": sha256(OUTPUT),
        },
        "metrics_summary": {
            key: {
                "overall_task_accuracy": value["overall_task_accuracy"],
                "answerable_exact_answer_accuracy": value["quality"]["exact_answer_accuracy"],
                "no_answer_refusal_accuracy": value["quality"]["hard_no_answer_refusal_accuracy"],
                "average_retrieval_rounds": value["budget"]["average_retrieval_rounds"],
                "amortized_mean_latency_ms": value["budget"]["amortized_mean_latency_ms"],
                "cost_per_correct_answer": value["cost_per_correct_answer"],
            }
            for key, value in metrics.items()
        },
        "seal_note": (
            "This file was created immediately after the only permitted successful run. "
            "The verifier, rewriter, models, parameters, corpus and labels must not be "
            "changed in response to these results."
        ),
    }
    RESULT_SEAL.write_text(
        json.dumps(seal, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(seal["metrics_summary"], ensure_ascii=False, indent=2))
    print(f"output={OUTPUT}")
    print(f"sealed_result={RESULT_SEAL}")


if __name__ == "__main__":
    main()
