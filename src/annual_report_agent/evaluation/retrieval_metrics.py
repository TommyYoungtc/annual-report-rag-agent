from __future__ import annotations

import math
from collections.abc import Iterable, Sequence


def recall_at_k(retrieved_ids: Sequence[str], relevant_ids: Iterable[str], k: int) -> float:
    relevant = set(relevant_ids)
    if not relevant:
        return 0.0
    retrieved = set(retrieved_ids[:k])
    return len(retrieved & relevant) / len(relevant)


def reciprocal_rank(retrieved_ids: Sequence[str], relevant_ids: Iterable[str]) -> float:
    relevant = set(relevant_ids)
    for rank, chunk_id in enumerate(retrieved_ids, start=1):
        if chunk_id in relevant:
            return 1.0 / rank
    return 0.0


def mrr(results: Sequence[tuple[Sequence[str], Iterable[str]]]) -> float:
    if not results:
        return 0.0
    return sum(reciprocal_rank(retrieved, relevant) for retrieved, relevant in results) / len(
        results
    )


def ndcg_at_k(retrieved_ids: Sequence[str], relevant_ids: Iterable[str], k: int) -> float:
    relevant = set(relevant_ids)
    if not relevant:
        return 0.0
    dcg = sum(
        1.0 / math.log2(rank + 1)
        for rank, chunk_id in enumerate(retrieved_ids[:k], start=1)
        if chunk_id in relevant
    )
    ideal_hits = min(len(relevant), k)
    ideal_dcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    return dcg / ideal_dcg if ideal_dcg else 0.0


def evaluate_retrieval(
    rows: Sequence[tuple[Sequence[str], Iterable[str]]],
    *,
    recall_ks: Sequence[int] = (1, 3, 5),
    ndcg_k: int = 5,
) -> dict[str, float]:
    if not rows:
        return {**{f"recall@{k}": 0.0 for k in recall_ks}, "mrr": 0.0, f"ndcg@{ndcg_k}": 0.0}

    metrics = {
        f"recall@{k}": sum(recall_at_k(retrieved, relevant, k) for retrieved, relevant in rows)
        / len(rows)
        for k in recall_ks
    }
    metrics["mrr"] = mrr(rows)
    metrics[f"ndcg@{ndcg_k}"] = sum(
        ndcg_at_k(retrieved, relevant, ndcg_k) for retrieved, relevant in rows
    ) / len(rows)
    return metrics
