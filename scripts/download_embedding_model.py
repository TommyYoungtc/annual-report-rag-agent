from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("HF_HOME", str(ROOT / "cache" / "huggingface"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download the compact BGE embedding baseline")
    parser.add_argument("--repo-id", default="BAAI/bge-small-zh-v1.5")
    parser.add_argument("--output", default="cache/models/bge-small-zh-v1.5")
    return parser.parse_args()


def project_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def main() -> None:
    from huggingface_hub import snapshot_download

    args = parse_args()
    output = project_path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    resolved = snapshot_download(
        repo_id=args.repo_id,
        local_dir=output,
        ignore_patterns=["pytorch_model.bin", "*.h5", "*.msgpack"],
    )
    safetensors = list(output.glob("*.safetensors"))
    if not safetensors:
        raise RuntimeError("download completed without a safetensors weight file")
    print(
        json.dumps(
            {
                "repo_id": args.repo_id,
                "local_dir": str(Path(resolved).resolve()),
                "weight_files": [path.name for path in safetensors],
                "bytes": sum(path.stat().st_size for path in output.rglob("*") if path.is_file()),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
