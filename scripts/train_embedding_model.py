from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("HF_HOME", str(ROOT / "cache" / "huggingface"))
os.environ.setdefault("TORCH_HOME", str(ROOT / "cache" / "torch"))
sys.path.insert(0, str(ROOT / "src"))

from annual_report_agent.io_utils import read_chunks

BGE_QUERY_INSTRUCTION = "为这个句子生成表示以用于检索相关文章："


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune BGE-small with mined hard negatives")
    parser.add_argument("--model", default="cache/models/bge-small-zh-v1.5")
    parser.add_argument(
        "--triples",
        default="data/training/embedding_hard_negatives_v1.jsonl",
    )
    parser.add_argument("--corpus", default="data/processed/pypdf_corpus.jsonl")
    parser.add_argument(
        "--output-model",
        default="cache/models/bge-small-zh-v1.5-annual-report-v1",
    )
    parser.add_argument(
        "--checkpoints",
        default="outputs/bge-small-training-checkpoints",
    )
    parser.add_argument(
        "--metrics",
        default="outputs/bge_small_training_metrics_v1.json",
    )
    parser.add_argument("--epochs", type=float, default=2.0)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--max-length", type=int, default=384)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def project_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    import torch
    from datasets import Dataset
    from sentence_transformers import (
        SentenceTransformer,
        SentenceTransformerTrainer,
        SentenceTransformerTrainingArguments,
    )
    from sentence_transformers.sentence_transformer import losses
    from sentence_transformers.sentence_transformer.training_args import BatchSamplers

    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this training run")

    triples_path = project_path(args.triples)
    triples = read_jsonl(triples_path)
    chunks = read_chunks(project_path(args.corpus))
    chunk_by_id = {chunk.chunk_id: chunk for chunk in chunks}
    dataset = Dataset.from_dict(
        {
            "anchor": [BGE_QUERY_INSTRUCTION + row["query"] for row in triples],
            "positive": [chunk_by_id[row["positive_chunk_id"]].text for row in triples],
            "negative": [chunk_by_id[row["negative_chunk_id"]].text for row in triples],
        }
    )

    model = SentenceTransformer(str(project_path(args.model)), device="cuda")
    model.max_seq_length = args.max_length
    loss = losses.MultipleNegativesRankingLoss(model)
    training_args = SentenceTransformerTrainingArguments(
        output_dir=str(project_path(args.checkpoints)),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        warmup_steps=0.1,
        fp16=True,
        bf16=False,
        batch_sampler=BatchSamplers.NO_DUPLICATES,
        logging_steps=10,
        logging_first_step=True,
        save_strategy="epoch",
        save_total_limit=1,
        report_to="none",
        seed=args.seed,
        data_seed=args.seed,
        dataloader_num_workers=0,
    )
    trainer = SentenceTransformerTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        loss=loss,
    )

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    result = trainer.train()
    elapsed = time.perf_counter() - started
    output_model = project_path(args.output_model)
    model.save_pretrained(str(output_model))

    metrics = {
        "status": "completed",
        "base_model": str(project_path(args.model)),
        "output_model": str(output_model),
        "loss": "MultipleNegativesRankingLoss",
        "query_instruction": BGE_QUERY_INSTRUCTION,
        "training_triples": len(triples),
        "training_data_sha256": sha256(triples_path),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "max_length": args.max_length,
        "seed": args.seed,
        "elapsed_seconds": round(elapsed, 3),
        "cuda_device": torch.cuda.get_device_name(0),
        "cuda_peak_allocated_mib": round(torch.cuda.max_memory_allocated() / 1024**2, 1),
        "cuda_peak_reserved_mib": round(torch.cuda.max_memory_reserved() / 1024**2, 1),
        "train_result": result.metrics,
    }
    metrics_path = project_path(args.metrics)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
