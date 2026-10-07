from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("HF_HOME", str(ROOT / "cache" / "huggingface"))
os.environ.setdefault("TORCH_HOME", str(ROOT / "cache" / "torch"))
sys.path.insert(0, str(ROOT / "src"))

from annual_report_agent.io_utils import read_chunks
from annual_report_agent.retrieval.dense import SentenceTransformerEncoder

CANDIDATE_LOCK = ROOT / "configs" / "adaptive_retrieval_v1.lock.json"
CONFIG = ROOT / "configs" / "adaptive_retrieval_v1.yaml"
CORPUS = ROOT / "data" / "processed" / "pypdf_corpus_v3_frozen.jsonl"
OUTPUT = ROOT / "cache" / "embeddings" / "pypdf_bge_small_annual_report_v3_frozen_384.npz"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_candidate_lock() -> dict:
    lock = json.loads(CANDIDATE_LOCK.read_text(encoding="utf-8"))
    if lock["status"] != "candidate_locked_before_new_holdout_selection":
        raise ValueError("adaptive candidate is not in the expected locked state")
    for name, artifact in lock["artifacts"].items():
        path = ROOT / artifact["path"]
        if not path.is_file() or sha256(path) != artifact["sha256"]:
            raise ValueError(f"candidate artifact changed after lock: {name}")
    return lock


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the independent V3 holdout document embedding cache"
    )
    parser.add_argument("--output", default=str(OUTPUT.relative_to(ROOT)))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = Path(args.output)
    output = output if output.is_absolute() else ROOT / output
    if output.exists():
        raise FileExistsError("V3 embedding cache already exists; refusing to overwrite it")

    validate_candidate_lock()
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    base = config["base_pipeline"]
    chunks = read_chunks(CORPUS)
    if not chunks:
        raise ValueError("V3 corpus is empty")

    encoder = SentenceTransformerEncoder(
        str(ROOT / base["embedding_model"]),
        device=base["device"],
        batch_size=int(base["embedding_batch_size"]),
        max_length=int(base["embedding_max_length"]),
        query_instruction=base["embedding_query_instruction"],
        query_template=base["embedding_query_template"],
    )
    embeddings = encoder.encode_documents([chunk.text for chunk in chunks])
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        embeddings=np.asarray(embeddings, dtype=np.float32),
        chunk_ids=np.asarray([chunk.chunk_id for chunk in chunks]),
        model=np.asarray(base["embedding_model"]),
        max_length=np.asarray(int(base["embedding_max_length"])),
        corpus_sha256=np.asarray(sha256(CORPUS)),
        candidate_lock_sha256=np.asarray(sha256(CANDIDATE_LOCK)),
    )

    del encoder, embeddings
    gc.collect()
    summary = {
        "output": str(output.relative_to(ROOT)).replace("\\", "/"),
        "chunks": len(chunks),
        "bytes": output.stat().st_size,
        "sha256": sha256(output),
        "corpus_sha256": sha256(CORPUS),
        "candidate_lock_sha256": sha256(CANDIDATE_LOCK),
        "model": base["embedding_model"],
        "max_length": int(base["embedding_max_length"]),
        "queries_encoded": 0,
        "abc_systems_run": False,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
