from collections.abc import Mapping, Sequence

from annual_report_agent.generation import (
    ContextPolicy,
    PromptVariant,
    build_context,
    build_prompt,
    choose_model,
)
from annual_report_agent.schemas import Chunk, SearchResult


class CharacterCounter:
    name = "character-test-counter"

    def count_text(self, text: str) -> int:
        return len(text)

    def truncate_text(self, text: str, max_tokens: int) -> str:
        return text[:max_tokens]

    def count_messages(
        self,
        messages: Sequence[Mapping[str, str]],
        *,
        add_generation_prompt: bool = True,
    ) -> int:
        size = sum(len(message["role"]) + len(message["content"]) for message in messages)
        return size + (9 if add_generation_prompt else 0)


def result(chunk_id: str, text: str, rank: int = 1) -> SearchResult:
    return SearchResult(
        chunk=Chunk(
            chunk_id=chunk_id,
            document_id="doc",
            company="测试公司",
            year=2024,
            section="主要会计数据",
            page=8,
            text=text,
        ),
        score=1.0,
        rank=rank,
        source="reranker",
    )


def test_compressed_prompt_is_smaller_and_hashed() -> None:
    full = build_prompt("营业收入？", "证据", variant=PromptVariant.FULL)
    compressed = build_prompt(
        "营业收入？",
        "证据",
        variant=PromptVariant.COMPRESSED,
    )
    assert len(compressed.system_prompt) < len(full.system_prompt)
    assert compressed.prompt_sha256 != full.prompt_sha256


def test_model_router_escalates_multi_period_questions() -> None:
    simple = choose_model(
        "测试公司2024年营业收入是多少？",
        policy="adaptive",
    )
    complex_decision = choose_model(
        "测试公司2023年和2024年营业收入变化率是多少？",
        policy="adaptive",
    )
    assert simple.model_tier == "small"
    assert complex_decision.model_tier == "large"


def test_context_span_keeps_metadata_and_respects_budget() -> None:
    counter = CharacterCounter()
    long_text = "无关内容" * 80 + "营业收入 100亿元" + "尾部" * 80
    built = build_context(
        "测试公司2024年营业收入是多少？",
        [result("doc:1", long_text)],
        counter=counter,
        policy=ContextPolicy.EVIDENCE_SPAN,
        top_k=1,
        max_tokens=180,
        window_chars=50,
    )
    assert built.evidence_tokens_after <= 180
    assert "chunk_id=doc:1" in built.text
    assert "营业收入" in built.text
    assert built.results[0].chunk.text in built.text
    assert built.truncated
