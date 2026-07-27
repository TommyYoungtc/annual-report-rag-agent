from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

import numpy as np

from ..schemas import SearchResult


class PairScorer(Protocol):
    def score(self, query: str, documents: Sequence[str]) -> np.ndarray: ...


class Qwen3Reranker:
    """Official yes/no-logit Qwen3 reranker adapted for an 8GB RTX 4060."""

    PREFIX = (
        "<|im_start|>system\nJudge whether the Document meets the requirements "
        "based on the Query and the Instruct provided. Note that the answer can only "
        'be "yes" or "no".<|im_end|>\n<|im_start|>user\n'
    )
    SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"

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
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as error:
            raise RuntimeError(
                "Reranking requires the 'models' dependencies. "
                'Install with: python -m pip install -e ".[models]"'
            ) from error

        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA device requested but PyTorch cannot access CUDA")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")

        self.torch = torch
        self.device = device
        self.batch_size = batch_size
        self.max_length = max_length
        self.instruction = instruction or (
            "Given a web search query, retrieve relevant passages that answer the query"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            padding_side="left",
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=torch.float16 if device.startswith("cuda") else torch.float32,
        ).to(device)
        self.model.eval()
        self.false_token_id = self.tokenizer.convert_tokens_to_ids("no")
        self.true_token_id = self.tokenizer.convert_tokens_to_ids("yes")
        self.prefix_tokens = self.tokenizer.encode(self.PREFIX, add_special_tokens=False)
        self.suffix_tokens = self.tokenizer.encode(self.SUFFIX, add_special_tokens=False)
        if max_length <= len(self.prefix_tokens) + len(self.suffix_tokens):
            raise ValueError("max_length is too small for the reranker prompt")

    def _format_pair(self, query: str, document: str) -> str:
        return f"<Instruct>: {self.instruction}\n<Query>: {query}\n<Document>: {document}"

    def _prepare_inputs(self, pairs: Sequence[str]):
        content_length = self.max_length - len(self.prefix_tokens) - len(self.suffix_tokens)
        encoded = self.tokenizer(
            list(pairs),
            padding=False,
            truncation="longest_first",
            return_attention_mask=False,
            max_length=content_length,
        )
        input_ids = [
            self.prefix_tokens + item + self.suffix_tokens for item in encoded["input_ids"]
        ]
        inputs = self.tokenizer.pad(
            {"input_ids": input_ids},
            padding=True,
            return_tensors="pt",
        )
        return {key: value.to(self.device) for key, value in inputs.items()}

    def score(self, query: str, documents: Sequence[str]) -> np.ndarray:
        if not documents:
            return np.empty(0, dtype=np.float32)

        formatted = [self._format_pair(query, document) for document in documents]
        scores: list[float] = []
        with self.torch.inference_mode():
            for start in range(0, len(formatted), self.batch_size):
                inputs = self._prepare_inputs(formatted[start : start + self.batch_size])
                final_logits = self.model(**inputs).logits[:, -1, :]
                true_logits = final_logits[:, self.true_token_id]
                false_logits = final_logits[:, self.false_token_id]
                binary_logits = self.torch.stack([false_logits, true_logits], dim=1)
                probabilities = self.torch.nn.functional.log_softmax(binary_logits, dim=1)[
                    :, 1
                ].exp()
                scores.extend(probabilities.float().cpu().tolist())
        return np.asarray(scores, dtype=np.float32)


# Backward-compatible name retained for the first engineering milestone.
SentenceTransformerReranker = Qwen3Reranker


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
