from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence

from ..schemas import SearchResult


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[SearchResult]],
    *,
    rank_constant: int = 60,
    top_k: int = 5,
    weights: Sequence[float] | None = None,
) -> list[SearchResult]:
    if rank_constant <= 0:
        raise ValueError("rank_constant must be positive")
    if weights is None:
        weights = [1.0] * len(rankings)
    if len(weights) != len(rankings):
        raise ValueError("weights must have the same length as rankings")
    if any(weight < 0 for weight in weights):
        raise ValueError("weights cannot be negative")

    scores: defaultdict[str, float] = defaultdict(float)
    chunks = {}
    sources: defaultdict[str, set[str]] = defaultdict(set)

    for ranking, weight in zip(rankings, weights):
        for fallback_rank, result in enumerate(ranking, start=1):
            rank = result.rank if result.rank > 0 else fallback_rank
            chunk_id = result.chunk.chunk_id
            scores[chunk_id] += weight / (rank_constant + rank)
            chunks[chunk_id] = result.chunk
            sources[chunk_id].add(result.source)

    ordered = sorted(scores, key=lambda key: (-scores[key], key))[:top_k]
    return [
        SearchResult(
            chunk=chunks[chunk_id],
            score=scores[chunk_id],
            rank=rank,
            source="+".join(sorted(sources[chunk_id])),
        )
        for rank, chunk_id in enumerate(ordered, start=1)
    ]
