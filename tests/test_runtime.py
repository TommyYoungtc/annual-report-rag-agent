from __future__ import annotations

from pathlib import Path

import numpy as np

from annual_report_agent.io_utils import write_chunks
from annual_report_agent.runtime import AnnualReportAgentRuntime, RuntimeSettings
from annual_report_agent.schemas import Chunk


class FakeEncoder:
    def encode_documents(self, texts):
        return np.asarray([[1.0, 0.0] for _ in texts], dtype=np.float32)

    def encode_queries(self, texts):
        return np.asarray([[1.0, 0.0] for _ in texts], dtype=np.float32)


class FakeReranker:
    def score(self, query, documents):
        return np.asarray([0.9 - index * 0.1 for index in range(len(documents))])


def build_runtime(tmp_path: Path):
    corpus_path = tmp_path / "corpus.jsonl"
    chunks = [
        Chunk(
            chunk_id="catl-2024:0001",
            document_id="catl-2024",
            company="宁德时代",
            year=2024,
            section="主要会计数据和财务指标",
            page=9,
            text="主要会计数据和财务指标 单位：千元 营业收入 362,012,554",
        ),
        Chunk(
            chunk_id="catl-2024:0002",
            document_id="catl-2024",
            company="宁德时代",
            year=2024,
            section="业务讨论",
            page=20,
            text="公司业务保持稳定发展。",
        ),
    ]
    write_chunks(corpus_path, chunks)
    embedding_model = tmp_path / "embedding-model"
    reranker_model = tmp_path / "reranker-model"
    embedding_model.mkdir()
    reranker_model.mkdir()
    cache_path = tmp_path / "embeddings.npz"
    np.savez(
        cache_path,
        chunk_ids=np.asarray([chunk.chunk_id for chunk in chunks]),
        embeddings=np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
        model=np.asarray(str(embedding_model)),
        max_length=np.asarray(384),
    )
    settings = RuntimeSettings(
        project_root=tmp_path,
        corpus_path=corpus_path,
        embedding_cache_path=cache_path,
        embedding_model_path=embedding_model,
        reranker_model_path=reranker_model,
        device="cpu",
        candidate_k=2,
        rerank_candidates=2,
        top_k=2,
    )
    calls = {"encoder": 0, "reranker": 0}

    def encoder_factory(*args, **kwargs):
        calls["encoder"] += 1
        return FakeEncoder()

    def reranker_factory(*args, **kwargs):
        calls["reranker"] += 1
        return FakeReranker()

    runtime = AnnualReportAgentRuntime(
        settings,
        encoder_factory=encoder_factory,
        reranker_factory=reranker_factory,
    )
    return runtime, calls


def test_project_settings_use_v2_blind_corpus_and_finetuned_bge(
    tmp_path: Path,
) -> None:
    settings = RuntimeSettings.from_project(tmp_path)
    assert settings.embedding_model_path.name == "bge-small-zh-v1.5-annual-report-v1"
    assert settings.corpus_path.name == "pypdf_corpus_v2_blind.jsonl"
    assert (
        settings.embedding_cache_path.name
        == "pypdf_bge_small_annual_report_v2_blind_384.npz"
    )
    assert settings.embedding_max_length == 384
    assert settings.embedding_query_template == "{instruction}{query}"
    assert settings.bm25_weight == 3.0


def test_runtime_answers_with_page_citation(tmp_path: Path) -> None:
    runtime, calls = build_runtime(tmp_path)
    response = runtime.ask("宁德时代2024年的营业收入是多少？")
    assert response.status == "answered"
    assert response.answer == "362,012,554千元"
    assert response.citations[0]["page"] == 9
    assert response.route["task_type"] == "single_fact"
    assert not response.gpu_used
    assert calls == {"encoder": 1, "reranker": 1}
    runtime.close()


def test_runtime_refuses_out_of_scope_before_model_load(tmp_path: Path) -> None:
    runtime, calls = build_runtime(tmp_path)
    response = runtime.ask("宁德时代2025年的营业收入是多少？")
    assert response.status == "refused"
    assert response.reason == "period_not_covered"
    assert not response.gpu_used
    assert calls == {"encoder": 0, "reranker": 0}
    runtime.close()


def test_runtime_switches_back_to_embedding_for_next_query(tmp_path: Path) -> None:
    runtime, calls = build_runtime(tmp_path)
    runtime.ask("宁德时代2024年的营业收入是多少？")
    runtime.ask("宁德时代2024年的营业收入是多少？")
    assert calls == {"encoder": 2, "reranker": 2}
    runtime.close()
