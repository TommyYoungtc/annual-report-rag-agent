from __future__ import annotations

import importlib.util
import json
import platform
import shutil
import subprocess
import sys
from typing import Any


def _gpu_info() -> dict[str, Any]:
    executable = shutil.which("nvidia-smi")
    if not executable:
        return {"available": False, "reason": "nvidia-smi not found"}
    command = [
        executable,
        "--query-gpu=name,memory.total,memory.free,driver_version",
        "--format=csv,noheader,nounits",
    ]
    try:
        completed = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (OSError, subprocess.SubprocessError) as error:
        return {"available": False, "reason": str(error)}

    first_line = completed.stdout.strip().splitlines()[0]
    name, total, free, driver = [part.strip() for part in first_line.split(",", 3)]
    return {
        "available": True,
        "name": name,
        "memory_total_mb": int(total),
        "memory_free_mb": int(free),
        "driver": driver,
    }


def collect_environment() -> dict[str, Any]:
    result: dict[str, Any] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "executable": sys.executable,
        "gpu": _gpu_info(),
        "packages": {},
    }

    for package in ("torch", "transformers", "sentence_transformers", "numpy"):
        result["packages"][package] = importlib.util.find_spec(package) is not None

    if result["packages"]["torch"]:
        import torch

        result["torch"] = {
            "version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "compiled_cuda": torch.version.cuda,
        }
        if torch.cuda.is_available():
            result["torch"]["device_name"] = torch.cuda.get_device_name(0)
            result["torch"]["allocated_mb"] = round(torch.cuda.memory_allocated(0) / 1024**2, 1)
    return result


def main() -> None:
    print(json.dumps(collect_environment(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
