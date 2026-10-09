from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


class UsageSource(str, Enum):
    """How token usage was obtained.

    Provider usage is suitable for billing claims. Tokenizer counts are exact for
    the configured tokenizer but are not evidence of an API bill. Estimates must
    never be mixed into either category silently.
    """

    PROVIDER = "provider_usage"
    TOKENIZER = "tokenizer_count"
    ESTIMATE = "estimate"


@dataclass(frozen=True, slots=True)
class ModelPrice:
    input_per_million_usd: float | None = None
    output_per_million_usd: float | None = None
    cached_input_per_million_usd: float | None = None

    @property
    def configured(self) -> bool:
        return self.input_per_million_usd is not None and self.output_per_million_usd is not None


def calculate_cost_usd(
    *,
    input_tokens: int,
    output_tokens: int,
    cached_input_tokens: int = 0,
    price: ModelPrice,
) -> float | None:
    """Calculate cost only when an explicit price table is configured."""

    if min(input_tokens, output_tokens, cached_input_tokens) < 0:
        raise ValueError("token counts cannot be negative")
    if not price.configured:
        return None
    cached = min(cached_input_tokens, input_tokens)
    uncached = input_tokens - cached
    cached_rate = (
        price.cached_input_per_million_usd
        if price.cached_input_per_million_usd is not None
        else price.input_per_million_usd
    )
    assert price.input_per_million_usd is not None
    assert price.output_per_million_usd is not None
    assert cached_rate is not None
    return (
        uncached * price.input_per_million_usd
        + cached * cached_rate
        + output_tokens * price.output_per_million_usd
    ) / 1_000_000


@dataclass(frozen=True, slots=True)
class TokenUsage:
    stage: str
    model: str
    source: UsageSource
    input_tokens: int = 0
    output_tokens: int = 0
    cached_input_tokens: int = 0
    prompt_static_tokens: int = 0
    query_tokens: int = 0
    evidence_tokens_before: int = 0
    evidence_tokens_after: int = 0
    pre_truncation_tokens: int = 0
    post_truncation_tokens: int = 0
    calls: int = 1
    cost_usd: float | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        numeric = (
            self.input_tokens,
            self.output_tokens,
            self.cached_input_tokens,
            self.prompt_static_tokens,
            self.query_tokens,
            self.evidence_tokens_before,
            self.evidence_tokens_after,
            self.pre_truncation_tokens,
            self.post_truncation_tokens,
            self.calls,
        )
        if any(value < 0 for value in numeric):
            raise ValueError("token usage values cannot be negative")
        if self.cached_input_tokens > self.input_tokens:
            raise ValueError("cached_input_tokens cannot exceed input_tokens")
        if self.cost_usd is not None and self.cost_usd < 0:
            raise ValueError("cost_usd cannot be negative")

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    @property
    def context_compression_ratio(self) -> float | None:
        if self.evidence_tokens_before == 0:
            return None
        return self.evidence_tokens_after / self.evidence_tokens_before

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["source"] = self.source.value
        value["total_tokens"] = self.total_tokens
        value["context_compression_ratio"] = self.context_compression_ratio
        return value

    @classmethod
    def from_openai_usage(
        cls,
        *,
        stage: str,
        model: str,
        usage: Mapping[str, Any],
        price: ModelPrice | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> TokenUsage:
        input_tokens = int(usage.get("prompt_tokens", usage.get("input_tokens", 0)))
        output_tokens = int(usage.get("completion_tokens", usage.get("output_tokens", 0)))
        details = usage.get("prompt_tokens_details") or usage.get("input_tokens_details") or {}
        cached_input_tokens = int(details.get("cached_tokens", 0))
        resolved_price = price or ModelPrice()
        return cls(
            stage=stage,
            model=model,
            source=UsageSource.PROVIDER,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cached_input_tokens=cached_input_tokens,
            cost_usd=calculate_cost_usd(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cached_input_tokens=cached_input_tokens,
                price=resolved_price,
            ),
            metadata=metadata or {},
        )


class TokenLedger:
    """Append-only per-task token ledger with stage/model rollups."""

    def __init__(self) -> None:
        self._entries: list[TokenUsage] = []

    @property
    def entries(self) -> tuple[TokenUsage, ...]:
        return tuple(self._entries)

    def add(self, usage: TokenUsage) -> None:
        self._entries.append(usage)

    def totals(self) -> dict[str, Any]:
        known_costs = [entry.cost_usd for entry in self._entries if entry.cost_usd is not None]
        return {
            "input_tokens": sum(entry.input_tokens for entry in self._entries),
            "output_tokens": sum(entry.output_tokens for entry in self._entries),
            "total_tokens": sum(entry.total_tokens for entry in self._entries),
            "cached_input_tokens": sum(entry.cached_input_tokens for entry in self._entries),
            "calls": sum(entry.calls for entry in self._entries),
            "cost_usd": sum(known_costs) if len(known_costs) == len(self._entries) else None,
            "cost_coverage": (len(known_costs) / len(self._entries) if self._entries else 0.0),
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "entries": [entry.to_dict() for entry in self._entries],
            "totals": self.totals(),
        }
