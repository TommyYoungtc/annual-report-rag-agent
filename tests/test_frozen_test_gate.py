from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


def load_evaluation_script():
    script = Path(__file__).resolve().parents[1] / "scripts" / "run_agent_evaluation.py"
    spec = importlib.util.spec_from_file_location("run_agent_evaluation", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_lock(module, root: Path, test_path: Path) -> Path:
    lock_path = root / "configs" / "final_v1.lock.json"
    lock_path.parent.mkdir(parents=True)
    lock_path.write_text(
        json.dumps(
            {
                "status": "locked",
                "artifacts": {
                    "frozen_test": {
                        "path": str(test_path.relative_to(root)),
                        "sha256": module.sha256(test_path),
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    return lock_path


def test_frozen_gate_validates_hash_and_refuses_second_output(tmp_path: Path) -> None:
    module = load_evaluation_script()
    module.ROOT = tmp_path
    test_path = tmp_path / "data" / "test.jsonl"
    test_path.parent.mkdir(parents=True)
    test_path.write_text('{"query_id":"test-001"}\n', encoding="utf-8")
    lock_path = write_lock(module, tmp_path, test_path)
    output_path = tmp_path / "outputs" / "frozen.json"

    module.validate_frozen_configuration(lock_path, test_path, output_path)
    output_path.parent.mkdir()
    output_path.write_text("{}\n", encoding="utf-8")
    with pytest.raises(FileExistsError, match="refusing a second run"):
        module.validate_frozen_configuration(lock_path, test_path, output_path)


def test_frozen_gate_rejects_changed_artifact(tmp_path: Path) -> None:
    module = load_evaluation_script()
    module.ROOT = tmp_path
    test_path = tmp_path / "data" / "test.jsonl"
    test_path.parent.mkdir(parents=True)
    test_path.write_text('{"query_id":"test-001"}\n', encoding="utf-8")
    lock_path = write_lock(module, tmp_path, test_path)
    test_path.write_text('{"query_id":"changed"}\n', encoding="utf-8")

    with pytest.raises(ValueError, match="locked artifact changed"):
        module.validate_frozen_configuration(
            lock_path,
            test_path,
            tmp_path / "outputs" / "frozen.json",
        )
