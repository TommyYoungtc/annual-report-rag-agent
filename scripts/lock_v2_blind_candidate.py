from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "configs" / "v2_blind_candidate.lock.json"

ARTIFACTS = {
    "configuration": "configs/v2_blind_candidate.yaml",
    "embedding_model": (
        "cache/models/bge-small-zh-v1.5-annual-report-v1/model.safetensors"
    ),
    "reranker_model": "cache/models/Qwen3-Reranker-0.6B-modelscope/model.safetensors",
    "development_corpus": "data/processed/pypdf_corpus_v2.jsonl",
    "development_embedding_cache": (
        "cache/embeddings/pypdf_bge_small_annual_report_v2_384.npz"
    ),
    "development_set": "data/eval/annual_report_v2_dev.jsonl",
    "hard_no_answer_development_set": (
        "data/eval/annual_report_v2_hard_no_answer_dev.jsonl"
    ),
    "development_result": "outputs/v2_blind_dev_metrics_locked_candidate.json",
    "runtime_source": "src/annual_report_agent/runtime.py",
    "answerer_source": "src/annual_report_agent/agent/answerer.py",
    "query_filter_source": "src/annual_report_agent/retrieval/query_filters.py",
    "dense_source": "src/annual_report_agent/retrieval/dense.py",
    "reranker_source": "src/annual_report_agent/retrieval/reranker.py",
    "evaluation_source": "scripts/run_agent_evaluation.py",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(
            "candidate lock already exists; refusing to replace pre-holdout evidence"
        )
    artifacts = {}
    for name, relative_path in ARTIFACTS.items():
        path = ROOT / relative_path
        if not path.is_file():
            raise FileNotFoundError(f"missing candidate artifact: {name}: {path}")
        artifacts[name] = {
            "path": relative_path,
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
    payload = {
        "configuration_id": "annual-report-agent-v2-blind-candidate",
        "status": "candidate_locked_before_holdout_ingestion",
        "locked_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "development_metrics": {
            "questions": 122,
            "end_to_end_exact_answer_accuracy": 1.0,
            "all_gold_evidence_recall_at_10": 1.0,
            "hard_no_answer_questions": 30,
            "hard_no_answer_refusal_accuracy": 1.0,
        },
        "invalidated_prior_claim": {
            "configuration_id": "annual-report-agent-final-v2",
            "reported_accuracy": 1.0,
            "reason": (
                "海康威视和美的集团的报告内容已在字段兼容与错误分析阶段被查看，"
                "因此原 32 题结果只能作为开发结果，不能作为无泄露测试成绩。"
            ),
        },
        "holdout_policy": {
            "holdout_companies_selected_after_this_lock": True,
            "source_and_hyperparameters_must_remain_unchanged": True,
            "test_labels_created_after_lock": True,
            "test_frozen_before_first_end_to_end_run": True,
            "maximum_successful_frozen_test_runs": 1,
        },
        "future_artifacts": {
            "manifest": "data/raw/manifest_v2_blind.jsonl",
            "corpus": "data/processed/pypdf_corpus_v2_blind.jsonl",
            "embedding_cache": (
                "cache/embeddings/pypdf_bge_small_annual_report_v2_blind_384.npz"
            ),
            "frozen_test": "data/eval/annual_report_v2_blind_test.jsonl",
        },
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
