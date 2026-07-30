from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
CANDIDATE_LOCK = ROOT / "configs" / "v2_blind_candidate.lock.json"
FINAL_LOCK = ROOT / "configs" / "final_v2_blind.lock.json"
TEST_RESULT = ROOT / "outputs" / "v2_blind_frozen_test_result.json"

ARTIFACTS = {
    "candidate_lock": "configs/v2_blind_candidate.lock.json",
    "configuration": "configs/v2_blind_candidate.yaml",
    "raw_manifest": "data/raw/manifest_v2_blind.jsonl",
    "corpus": "data/processed/pypdf_corpus_v2_blind.jsonl",
    "extraction_report": "data/parsed/pypdf_v2_blind/extraction_report.json",
    "embedding_model": (
        "cache/models/bge-small-zh-v1.5-annual-report-v1/model.safetensors"
    ),
    "embedding_cache": (
        "cache/embeddings/pypdf_bge_small_annual_report_v2_blind_384.npz"
    ),
    "reranker_model": "cache/models/Qwen3-Reranker-0.6B-modelscope/model.safetensors",
    "development_set": "data/eval/annual_report_v2_dev.jsonl",
    "hard_no_answer_development_set": (
        "data/eval/annual_report_v2_hard_no_answer_dev.jsonl"
    ),
    "development_result": "outputs/v2_blind_dev_metrics_locked_candidate.json",
    "frozen_test": "data/eval/annual_report_v2_blind_test.jsonl",
    "frozen_test_manifest": "data/eval/annual_report_v2_blind_manifest.json",
    "runtime_source": "src/annual_report_agent/runtime.py",
    "answerer_source": "src/annual_report_agent/agent/answerer.py",
    "query_filter_source": "src/annual_report_agent/retrieval/query_filters.py",
    "dense_source": "src/annual_report_agent/retrieval/dense.py",
    "reranker_source": "src/annual_report_agent/retrieval/reranker.py",
    "evaluation_source": "scripts/run_agent_evaluation.py",
    "test_builder_source": "scripts/build_and_freeze_v2_blind_test.py",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_candidate_lock() -> dict:
    lock = json.loads(CANDIDATE_LOCK.read_text(encoding="utf-8"))
    if lock["status"] != "candidate_locked_before_holdout_ingestion":
        raise ValueError("candidate configuration is not locked")
    for name, artifact in lock["artifacts"].items():
        path = ROOT / artifact["path"]
        if not path.is_file() or sha256(path) != artifact["sha256"]:
            raise ValueError(f"candidate artifact changed after lock: {name}")
    return lock


def main() -> None:
    if FINAL_LOCK.exists():
        raise FileExistsError("final blind lock already exists; refusing to overwrite")
    if TEST_RESULT.exists():
        raise FileExistsError("frozen test result exists before final lock")
    candidate = validate_candidate_lock()
    manifest_path = ROOT / "data" / "eval" / "annual_report_v2_blind_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["status"] != "frozen_before_first_model_run":
        raise ValueError("blind test manifest is not frozen")
    if manifest["successful_model_runs_at_freeze"] != 0:
        raise ValueError("blind test has already been run")
    artifacts = {}
    for name, relative_path in ARTIFACTS.items():
        path = ROOT / relative_path
        if not path.is_file():
            raise FileNotFoundError(f"missing final artifact: {name}: {path}")
        artifacts[name] = {
            "path": relative_path,
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
    payload = {
        "configuration_id": "annual-report-agent-final-v2-blind",
        "status": "locked",
        "locked_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "candidate_locked_at": candidate["locked_at"],
        "quality_gates": {
            "development_122": "passed_100_percent",
            "hard_no_answer_30": "passed_100_percent",
            "frozen_test_32": "pending_single_run",
        },
        "holdout_policy": candidate["holdout_policy"],
        "test_result_output": str(TEST_RESULT.relative_to(ROOT)),
        "artifacts": artifacts,
    }
    FINAL_LOCK.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"final_lock={FINAL_LOCK}")
    print(f"sha256={sha256(FINAL_LOCK)}")


if __name__ == "__main__":
    main()
