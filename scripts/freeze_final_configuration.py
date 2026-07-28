from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import yaml

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Lock the final configuration before Test")
    parser.add_argument("--config", default="configs/final_v1.yaml")
    parser.add_argument("--output", default="configs/final_v1.lock.json")
    return parser.parse_args()


def project_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_perfect_dev(path: Path) -> None:
    result = json.loads(path.read_text(encoding="utf-8"))
    metrics = result["metrics"]
    gates = (
        "dev_exact_answer_accuracy",
        "dev_all_gold_evidence_recall_at_10",
        "citation_chunk_precision",
        "citation_page_set_accuracy",
        "no_answer_refusal_accuracy",
    )
    failed = {name: metrics[name] for name in gates if metrics[name] != 1.0}
    if failed:
        raise ValueError(f"Dev quality gate failed: {failed}")
    if result.get("frozen_test_evaluated"):
        raise ValueError("Dev result unexpectedly reports a frozen-test run")


def require_perfect_hard_no_answer(path: Path) -> None:
    result = json.loads(path.read_text(encoding="utf-8"))
    accuracy = result["metrics"]["constraint_aware_hard_refusal_accuracy"]
    if accuracy != 1.0:
        raise ValueError(f"hard no-answer quality gate failed: {accuracy}")
    if result.get("frozen_test_evaluated"):
        raise ValueError("hard no-answer result unexpectedly reports a frozen-test run")


def main() -> None:
    args = parse_args()
    config_path = project_path(args.config)
    output_path = project_path(args.output)
    if output_path.exists():
        raise FileExistsError(f"configuration lock already exists: {output_path}")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    artifact_paths = {
        name: project_path(value) for name, value in config["artifacts"].items()
    }
    missing = {name: str(path) for name, path in artifact_paths.items() if not path.is_file()}
    if missing:
        raise FileNotFoundError(f"cannot lock missing artifacts: {missing}")

    require_perfect_dev(artifact_paths["dev_result"])
    require_perfect_hard_no_answer(artifact_paths["hard_no_answer_result"])
    artifact_paths["configuration"] = config_path
    lock = {
        "configuration_id": config["configuration_id"],
        "status": "locked",
        "locked_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "quality_gates": {
            "dev": "passed",
            "hard_no_answer": "passed",
            "frozen_test": "pending_one_time_run",
        },
        "artifacts": {
            name: {
                "path": str(path.relative_to(ROOT)),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
            for name, path in artifact_paths.items()
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(lock, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": lock["status"],
                "configuration_id": lock["configuration_id"],
                "artifacts": len(lock["artifacts"]),
                "frozen_test": lock["quality_gates"]["frozen_test"],
                "output": str(output_path),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
