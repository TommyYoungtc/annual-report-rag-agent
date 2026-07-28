from __future__ import annotations

import gc
import threading
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .agent import CorpusScope, answer_from_evidence, route_query
from .io_utils import read_chunks
from .retrieval import (
    BM25Retriever,
    Qwen3Reranker,
    expand_query_with_section_anchors,
    infer_allowed_document_ids,
    reciprocal_rank_fusion,
    rerank_results,
)
from .retrieval.dense import DenseRetriever, SentenceTransformerEncoder


@dataclass(frozen=True, slots=True)
class RuntimeSettings:
    project_root: Path
    corpus_path: Path
    embedding_cache_path: Path
    embedding_model_path: Path
    reranker_model_path: Path
    device: str = "cuda"
    candidate_k: int = 30
    rerank_candidates: int = 10
    top_k: int = 10
    bm25_weight: float = 10.0
    embedding_batch_size: int = 32
    embedding_max_length: int = 384
    embedding_query_instruction: str = "为这个句子生成表示以用于检索相关文章："
    embedding_query_template: str = "{instruction}{query}"
    reranker_batch_size: int = 1
    reranker_max_length: int = 1024
    minimum_reranker_score: float = 0.0

    @classmethod
    def from_project(cls, project_root: Path) -> RuntimeSettings:
        root = project_root.resolve()
        return cls(
            project_root=root,
            corpus_path=root / "data" / "processed" / "pypdf_corpus.jsonl",
            embedding_cache_path=root
            / "cache"
            / "embeddings"
            / "pypdf_bge_small_annual_report_v1_384.npz",
            embedding_model_path=root
            / "cache"
            / "models"
            / "bge-small-zh-v1.5-annual-report-v1",
            reranker_model_path=root / "cache" / "models" / "Qwen3-Reranker-0.6B-modelscope",
        )


@dataclass(frozen=True, slots=True)
class RuntimeResponse:
    query: str
    status: str
    answer: str | None
    reason: str | None
    route: dict[str, Any]
    citations: tuple[dict[str, Any], ...]
    calculation: str | None
    timing_ms: dict[str, float]
    gpu_used: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class AnnualReportAgentRuntime:
    def __init__(
        self,
        settings: RuntimeSettings,
        *,
        encoder_factory: Callable[..., Any] = SentenceTransformerEncoder,
        reranker_factory: Callable[..., Any] = Qwen3Reranker,
    ) -> None:
        self.settings = settings
        self.chunks = read_chunks(settings.corpus_path)
        self.scope = CorpusScope.from_chunks(self.chunks)
        self.bm25 = BM25Retriever(self.chunks)
        self._encoder_factory = encoder_factory
        self._reranker_factory = reranker_factory
        self._encoder = None
        self._dense = None
        self._reranker = None
        self._lock = threading.Lock()
        self._document_embeddings = self._load_embedding_cache()

    def _load_embedding_cache(self) -> np.ndarray:
        with np.load(self.settings.embedding_cache_path, allow_pickle=False) as cache:
            cached_ids = cache["chunk_ids"].astype(str).tolist()
            if cached_ids != [chunk.chunk_id for chunk in self.chunks]:
                raise ValueError("embedding cache does not match corpus chunk IDs")
            cached_length = int(cache["max_length"].item())
            if cached_length != self.settings.embedding_max_length:
                raise ValueError("embedding cache max_length does not match runtime")
            cached_model = Path(str(cache["model"].item()))
            if not cached_model.is_absolute():
                cached_model = self.settings.project_root / cached_model
            if cached_model.resolve() != self.settings.embedding_model_path.resolve():
                raise ValueError("embedding cache model does not match runtime")
            return np.asarray(cache["embeddings"], dtype=np.float32)

    @staticmethod
    def _clear_cuda() -> None:
        gc.collect()
        try:
            import torch
        except ImportError:
            return
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _load_dense(self) -> tuple[DenseRetriever, float]:
        if self._dense is not None:
            return self._dense, 0.0
        if self._reranker is not None:
            del self._reranker
            self._reranker = None
            self._clear_cuda()
        started = time.perf_counter()
        self._encoder = self._encoder_factory(
            str(self.settings.embedding_model_path),
            device=self.settings.device,
            batch_size=self.settings.embedding_batch_size,
            max_length=self.settings.embedding_max_length,
            query_instruction=self.settings.embedding_query_instruction,
            query_template=self.settings.embedding_query_template,
        )
        self._dense = DenseRetriever(
            self.chunks,
            self._encoder,
            document_embeddings=self._document_embeddings,
        )
        return self._dense, (time.perf_counter() - started) * 1000

    def _load_reranker(self) -> tuple[Any, float]:
        if self._reranker is not None:
            return self._reranker, 0.0
        if self._dense is not None:
            del self._dense, self._encoder
            self._dense = None
            self._encoder = None
            self._clear_cuda()
        started = time.perf_counter()
        self._reranker = self._reranker_factory(
            str(self.settings.reranker_model_path),
            device=self.settings.device,
            batch_size=self.settings.reranker_batch_size,
            max_length=self.settings.reranker_max_length,
            instruction=(
                "Given a Chinese annual report question, judge whether the passage "
                "directly contains evidence that answers the question."
            ),
        )
        return self._reranker, (time.perf_counter() - started) * 1000

    @staticmethod
    def _route_dict(route: Any) -> dict[str, Any]:
        return {
            "companies": list(route.companies),
            "years": list(route.years),
            "task_type": route.task_type,
            "requires_calculation": route.requires_calculation,
            "should_refuse": route.should_refuse,
            "refusal_reason": route.refusal_reason,
        }

    def ask(self, query: str) -> RuntimeResponse:
        total_started = time.perf_counter()
        route = route_query(query, self.scope)
        if route.should_refuse:
            answer = answer_from_evidence(query, route, [], self.scope)
            return RuntimeResponse(
                query=query,
                status=answer.status,
                answer=answer.answer,
                reason=answer.reason,
                route=self._route_dict(route),
                citations=(),
                calculation=None,
                timing_ms={"total": (time.perf_counter() - total_started) * 1000},
                gpu_used=False,
            )

        with self._lock:
            dense, embedding_load_ms = self._load_dense()
            retrieval_started = time.perf_counter()
            allowed = infer_allowed_document_ids(query, self.chunks)
            dense_results = dense.search(
                query,
                top_k=self.settings.candidate_k,
                allowed_document_ids=allowed,
            )
            bm25_results = self.bm25.search(
                expand_query_with_section_anchors(query),
                top_k=self.settings.candidate_k,
                allowed_document_ids=allowed,
            )
            candidates = reciprocal_rank_fusion(
                [dense_results, bm25_results],
                top_k=self.settings.candidate_k,
                weights=[1.0, self.settings.bm25_weight],
            )
            retrieval_ms = (time.perf_counter() - retrieval_started) * 1000
            reranker, reranker_load_ms = self._load_reranker()
            rerank_started = time.perf_counter()
            reranked = rerank_results(
                query,
                candidates[: self.settings.rerank_candidates],
                reranker,
                top_k=self.settings.top_k,
            )
            rerank_ms = (time.perf_counter() - rerank_started) * 1000
            answer = answer_from_evidence(
                query,
                route,
                reranked,
                self.scope,
                minimum_reranker_score=self.settings.minimum_reranker_score,
            )

        return RuntimeResponse(
            query=query,
            status=answer.status,
            answer=answer.answer,
            reason=answer.reason,
            route=self._route_dict(route),
            citations=tuple(asdict(citation) for citation in answer.citations),
            calculation=answer.calculation,
            timing_ms={
                "embedding_load": embedding_load_ms,
                "retrieval": retrieval_ms,
                "reranker_load": reranker_load_ms,
                "rerank": rerank_ms,
                "total": (time.perf_counter() - total_started) * 1000,
            },
            gpu_used=self.settings.device.startswith("cuda"),
        )

    def close(self) -> None:
        self._dense = None
        self._encoder = None
        self._reranker = None
        self._clear_cuda()
