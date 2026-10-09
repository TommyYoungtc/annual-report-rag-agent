from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, replace
from enum import Enum

from ..agent.answerer import extract_value, infer_fact_spec
from ..schemas import SearchResult
from ..telemetry.tokenizer_counter import TokenCounter


class ContextPolicy(str, Enum):
    FULL = "full"
    EVIDENCE_SPAN = "evidence_span"


@dataclass(frozen=True, slots=True)
class ContextBuild:
    policy: ContextPolicy
    text: str
    results: tuple[SearchResult, ...]
    evidence_tokens_before: int
    evidence_tokens_after: int
    chunks_before: int
    chunks_after: int
    truncated: bool

    @property
    def compression_ratio(self) -> float:
        if self.evidence_tokens_before == 0:
            return 1.0
        return self.evidence_tokens_after / self.evidence_tokens_before


ANCHORS = (
    "营业收入",
    "基本每股收益",
    "净资产收益率",
    "资产总额",
    "总资产",
    "研发人员",
    "研发投入",
    "净利润",
    "现金流量净额",
    "净资产",
    "员工",
    "技术人员",
    "每10股",
    "海外",
    "境外",
    "季度",
    "年度",
)

STRUCTURAL_MARKERS = (
    "主要会计数据和财务指标",
    "公司研发人员情况",
    "近三年公司研发投入金额",
    "公司研发投入情况",
    "公司员工情况",
    "员工数量、专业构成及教育程度",
    "董事会审议的报告期利润分配预案",
    "年度报告 第一节 重要提示",
)


def _format_result(result: SearchResult, body: str | None = None) -> str:
    chunk = result.chunk
    header = (
        f"[chunk_id={chunk.chunk_id};company={chunk.company};year={chunk.year};"
        f"page={chunk.page};section={chunk.section}]"
    )
    return f"{header}\n{body if body is not None else chunk.text}"


def _query_anchors(query: str) -> tuple[str, ...]:
    anchors = [anchor for anchor in ANCHORS if anchor in query]
    anchors.extend(re.findall(r"20\d{2}", query))
    return tuple(dict.fromkeys(anchors))


def _evidence_span(result: SearchResult, query: str, *, window_chars: int) -> str:
    text = result.chunk.text
    spec = infer_fact_spec(query)
    extracted = extract_value(result, spec) if spec is not None else None
    if extracted is not None:
        unit_matches = list(re.finditer(r"单位[：:]?\s*(?:千元|万元|亿元|元)", text))
        unit_context = unit_matches[-1].group(0) if unit_matches else ""
        structural_context = next(
            (marker for marker in STRUCTURAL_MARKERS if marker in text),
            "",
        )
        parts = [
            part
            for part in (structural_context, unit_context, extracted.citation.quote)
            if part
        ]
        return "\n".join(dict.fromkeys(parts))
    anchors = _query_anchors(query)
    center = next((text.find(anchor) for anchor in anchors if text.find(anchor) >= 0), -1)
    if center < 0:
        return text[: window_chars * 2]
    line_start = text.rfind("\n", 0, center)
    line_end = text.find("\n", center)
    if line_start >= 0 and line_end > center and line_end - line_start <= window_chars * 2:
        start = max(0, line_start - window_chars)
        end = min(len(text), line_end + window_chars)
    else:
        start = max(0, center - window_chars)
        end = min(len(text), center + window_chars)
    return text[start:end].strip()


def _fit_body(
    result: SearchResult,
    body: str,
    *,
    remaining_tokens: int,
    counter: TokenCounter,
) -> str | None:
    empty_cost = counter.count_text(_format_result(result, ""))
    body_budget = remaining_tokens - empty_cost
    if body_budget <= 0:
        return None
    return counter.truncate_text(body, body_budget)


def build_context(
    query: str,
    results: Sequence[SearchResult],
    *,
    counter: TokenCounter,
    policy: ContextPolicy,
    top_k: int,
    max_tokens: int | None = None,
    window_chars: int = 420,
) -> ContextBuild:
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    if max_tokens is not None and max_tokens <= 0:
        raise ValueError("max_tokens must be positive")

    baseline_results = list(results[:top_k])
    baseline_text = "\n\n".join(_format_result(result) for result in baseline_results)
    before_tokens = counter.count_text(baseline_text)
    selected: list[SearchResult] = []
    blocks: list[str] = []
    used_tokens = 0

    for result in baseline_results:
        body = result.chunk.text
        if policy == ContextPolicy.EVIDENCE_SPAN:
            body = _evidence_span(result, query, window_chars=window_chars)
        block = _format_result(result, body)
        block_tokens = counter.count_text(block)
        separator_tokens = counter.count_text("\n\n") if blocks else 0
        if max_tokens is not None and used_tokens + separator_tokens + block_tokens > max_tokens:
            remaining = max_tokens - used_tokens - separator_tokens
            body = _fit_body(
                result,
                body,
                remaining_tokens=remaining,
                counter=counter,
            )
            if not body:
                continue
            block = _format_result(result, body)
            block_tokens = counter.count_text(block)
        clipped_chunk = replace(result.chunk, text=body)
        selected.append(replace(result, chunk=clipped_chunk))
        blocks.append(block)
        used_tokens += separator_tokens + block_tokens
        if max_tokens is not None and used_tokens >= max_tokens:
            break

    text = "\n\n".join(blocks)
    after_tokens = counter.count_text(text)
    return ContextBuild(
        policy=policy,
        text=text,
        results=tuple(selected),
        evidence_tokens_before=before_tokens,
        evidence_tokens_after=after_tokens,
        chunks_before=len(baseline_results),
        chunks_after=len(selected),
        truncated=after_tokens < before_tokens,
    )
