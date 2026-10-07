from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "adaptive_retrieval_v1.yaml"
RESULT = ROOT / "outputs" / "adaptive_retrieval_v1_dev.json"
BASE_LOCK = ROOT / "configs" / "final_v2_blind.lock.json"
OUTPUT = ROOT / "configs" / "adaptive_retrieval_v1.lock.json"

ARTIFACTS = {
    "configuration": "configs/adaptive_retrieval_v1.yaml",
    "experiment_protocol": "docs/ADAPTIVE_RETRIEVAL_EXPERIMENT.md",
    "development_report": "docs/ADAPTIVE_RETRIEVAL_DEV_REPORT.md",
    "development_result": "outputs/adaptive_retrieval_v1_dev.json",
    "trajectory_source": "src/annual_report_agent/agent/trajectory.py",
    "evidence_verifier_source": (
        "src/annual_report_agent/agent/evidence_verifier.py"
    ),
    "query_rewriter_source": "src/annual_report_agent/agent/query_rewriter.py",
    "retrieval_controller_source": (
        "src/annual_report_agent/agent/retrieval_controller.py"
    ),
    "agent_exports_source": "src/annual_report_agent/agent/__init__.py",
    "evaluation_source": "scripts/run_adaptive_retrieval_evaluation.py",
    "candidate_lock_source": "scripts/lock_adaptive_retrieval_candidate.py",
    "trajectory_test": "tests/test_trajectory.py",
    "verifier_test": "tests/test_evidence_verifier.py",
    "rewriter_test": "tests/test_query_rewriter.py",
    "controller_test": "tests/test_retrieval_controller.py",
    "corpus": "data/processed/pypdf_corpus_v2_blind.jsonl",
    "embedding_cache": (
        "cache/embeddings/pypdf_bge_small_annual_report_v2_blind_384.npz"
    ),
    "embedding_model": (
        "cache/models/bge-small-zh-v1.5-annual-report-v1/model.safetensors"
    ),
    "reranker_model": "cache/models/Qwen3-Reranker-0.6B-modelscope/model.safetensors",
    "answerable_development_set": "data/eval/annual_report_v2_dev.jsonl",
    "hard_no_answer_development_set": (
        "data/eval/annual_report_v2_hard_no_answer_dev.jsonl"
    ),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def verify_base_lock() -> dict[str, Any]:
    lock = json.loads(BASE_LOCK.read_text(encoding="utf-8"))
    require(lock["status"] == "locked", "base V2 lock is not locked")
    for name, artifact in lock["artifacts"].items():
        path = ROOT / artifact["path"]
        require(path.is_file(), f"base artifact missing: {name}: {path}")
        require(
            sha256(path) == artifact["sha256"],
            f"base artifact changed after V2 lock: {name}",
        )
    return lock


def validate_result(
    config: dict[str, Any], result: dict[str, Any]
) -> dict[str, Any]:
    require(
        result["configuration_id"] == config["configuration_id"],
        "result does not match configuration",
    )
    require(
        result["evaluation_split"] == "development_and_contaminated_canary",
        "unexpected evaluation split",
    )
    require(
        result["frozen_test_evaluated"] is False,
        "frozen test must not be evaluated before candidate lock",
    )
    require(
        result["known_failure_canary_in_aggregate_metrics"] is False,
        "known V2 failures must be excluded from aggregate metrics",
    )
    require(
        result["development_gates"]["passed"] is True,
        "development gates did not pass",
    )
    require(
        result["dataset"]["answerable_dev"] == 122,
        "lock requires the complete 122-question answerable Dev set",
    )
    require(
        result["dataset"]["hard_no_answer_dev"] == 30,
        "lock requires the complete 30-question hard no-answer Dev set",
    )
    require(
        result["dataset"]["known_failure_canary"] == 2,
        "lock requires exactly two contaminated development canaries",
    )
    return {
        key: {
            "quality": value["quality"],
            "budget": value["budget"],
            "routing": value["routing"],
        }
        for key, value in result["metrics"].items()
    }


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(
            "adaptive candidate lock already exists; refusing to replace it"
        )
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    require(
        config["status"] == "ready_for_development_run",
        "configuration is not ready for development evaluation",
    )
    base_lock = verify_base_lock()
    result = json.loads(RESULT.read_text(encoding="utf-8"))
    metrics = validate_result(config, result)

    artifacts = {}
    for name, relative_path in ARTIFACTS.items():
        path = ROOT / relative_path
        if not path.is_file():
            raise FileNotFoundError(f"missing adaptive artifact: {name}: {path}")
        artifacts[name] = {
            "path": relative_path,
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        }

    payload = {
        "configuration_id": config["configuration_id"],
        "status": "candidate_locked_before_new_holdout_selection",
        "locked_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "base_v2_lock": {
            "path": str(BASE_LOCK.relative_to(ROOT)).replace("\\", "/"),
            "sha256": sha256(BASE_LOCK),
            "configuration_id": base_lock["configuration_id"],
        },
        "development_only": True,
        "frozen_test_evaluated": False,
        "known_failure_canaries": {
            "ids": config["development_evaluation"]["known_failure_canary_ids"],
            "included_in_aggregate_metrics": False,
            "eligible_for_new_test_claims": False,
        },
        "development_metrics": metrics,
        "future_holdout_policy": config["future_holdout_policy"],
        "artifacts": artifacts,
    }
    OUTPUT.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"candidate_lock={OUTPUT}")
    print(f"sha256={sha256(OUTPUT)}")


if __name__ == "__main__":
    main()
