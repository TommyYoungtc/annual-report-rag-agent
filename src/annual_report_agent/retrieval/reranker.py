from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

import numpy as np

from ..schemas import SearchResult


class PairScorer(Protocol):
    def score(self, query: str, documents: Sequence[str]) -> np.ndarray: ...


class SentenceTransformerReranker:
    """CrossEncoder adapter sized for sequential use on an 8GB RTX 4060."""

    def __init__(
        self,
        model_name: str,
        *,
        device: str = "cuda",
        batch_size: int = 1,
        max_length: int = 1024,
        instruction: str | None = None,
    ) -> None:
        try:
            import torch
            from sentence_transformers import CrossEncoder
        except ImportError as error:
            raise RuntimeError(
                "Reranking requires the 'models' dependencies. "
                'Install with: python -m pip install -e ".[models]"'
            ) from error

        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA device requested but PyTorch cannot access CUDA")

        model_kwargs = {"torch_dtype": torch.float16} if device.startswith("cuda") else {}
        prompts = {"annual_report": instruction} if instruction else None
        self.model = CrossEncoder(
            model_name,
            device=device,
            model_kwargs=model_kwargs,
            max_length=max_length,
            prompts=prompts,
            default_prompt_name="annual_report" if instruction else None,
        )
        self.batch_size = batch_size

    def score(self, query: str, documents: Sequence[str]) -> np.ndarray:
        if not documents:
            return np.empty(0, dtype=np.float32)
        pairs = [(query, document) for document in documents]
        scores = self.model.predict(
            pairs,
            batch_size=self.batch_size,
            convert_to_numpy=True,
            show_progress_bar=len(pairs) > self.batch_size,
        )
        return np.asarray(scores, dtype=np.float32).reshape(-1)


def rerank_results(
    query: str,
    results: Sequence[SearchResult],
    scorer: PairScorer,
    *,
    top_k: int = 5,
) -> list[SearchResult]:
    if top_k <= 0 or not results:
        return []
    scores = np.asarray(
        scorer.score(query, [result.chunk.text for result in results]),
        dtype=np.float32,
    ).reshape(-1)
    if scores.shape[0] != len(results):
        raise ValueError("reranker score count must match candidate count")

    ordered = np.argsort(-scores, kind="stable")[:top_k]
    return [
        SearchResult(
            chunk=results[int(index)].chunk,
            score=float(scores[index]),
            rank=rank,
            source="reranker",
        )
        for rank, index in enumerate(ordered, start=1)
    ]
