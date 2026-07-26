from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

import numpy as np

from ..schemas import Chunk, SearchResult


class TextEncoder(Protocol):
    def encode_documents(self, texts: Sequence[str]) -> np.ndarray: ...

    def encode_queries(self, texts: Sequence[str]) -> np.ndarray: ...


class SentenceTransformerEncoder:
    """Lazy Sentence Transformers adapter tuned for an 8GB RTX 4060."""

    def __init__(
        self,
        model_name: str,
        *,
        device: str = "cuda",
        batch_size: int = 4,
        max_length: int = 768,
        query_instruction: str | None = None,
    ) -> None:
        try:
            import torch
            from sentence_transformers import SentenceTransformer
        except ImportError as error:
            raise RuntimeError(
                "Dense retrieval requires the 'models' dependencies. "
                'Install with: python -m pip install -e ".[models]"'
            ) from error

        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA device requested but PyTorch cannot access CUDA")

        model_kwargs = {"torch_dtype": torch.float16} if device.startswith("cuda") else {}
        self.model = SentenceTransformer(
            model_name,
            device=device,
            model_kwargs=model_kwargs,
        )
        self.model.max_seq_length = max_length
        self.batch_size = batch_size
        self.query_instruction = query_instruction

    def _encode(self, texts: Sequence[str]) -> np.ndarray:
        return np.asarray(
            self.model.encode(
                list(texts),
                batch_size=self.batch_size,
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=len(texts) > self.batch_size,
            ),
            dtype=np.float32,
        )

    def encode_documents(self, texts: Sequence[str]) -> np.ndarray:
        return self._encode(texts)

    def encode_queries(self, texts: Sequence[str]) -> np.ndarray:
        if self.query_instruction:
            texts = [f"Instruct: {self.query_instruction}\nQuery: {text}" for text in texts]
        return self._encode(texts)


class DenseRetriever:
    def __init__(
        self,
        chunks: Sequence[Chunk],
        encoder: TextEncoder,
        *,
        document_embeddings: np.ndarray | None = None,
    ) -> None:
        if not chunks:
            raise ValueError("chunks cannot be empty")
        self.chunks = list(chunks)
        self.encoder = encoder
        if document_embeddings is None:
            document_embeddings = encoder.encode_documents([chunk.text for chunk in self.chunks])
        self.document_embeddings = np.asarray(document_embeddings, dtype=np.float32)
        if self.document_embeddings.ndim != 2:
            raise ValueError("encoder must return a 2D embedding matrix")
        if self.document_embeddings.shape[0] != len(self.chunks):
            raise ValueError("embedding row count must match chunk count")

    def search(
        self,
        query: str,
        *,
        top_k: int = 5,
        allowed_document_ids: set[str] | None = None,
    ) -> list[SearchResult]:
        if top_k <= 0:
            return []
        query_embedding = np.asarray(self.encoder.encode_queries([query]), dtype=np.float32)
        if query_embedding.shape != (1, self.document_embeddings.shape[1]):
            raise ValueError("query embedding dimension does not match document embeddings")

        scores = self.document_embeddings @ query_embedding[0]
        if allowed_document_ids is None:
            candidate_indices = np.arange(len(self.chunks))
        else:
            candidate_indices = np.asarray(
                [
                    index
                    for index, chunk in enumerate(self.chunks)
                    if chunk.document_id in allowed_document_ids
                ],
                dtype=np.int64,
            )
        if candidate_indices.size == 0:
            return []
        candidate_scores = scores[candidate_indices]
        ordered = candidate_indices[np.argsort(-candidate_scores, kind="stable")[:top_k]]
        return [
            SearchResult(
                chunk=self.chunks[int(index)],
                score=float(scores[index]),
                rank=rank,
                source="dense",
            )
            for rank, index in enumerate(ordered, start=1)
        ]
