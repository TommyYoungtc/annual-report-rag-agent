from annual_report_agent.telemetry import (
    ModelPrice,
    TokenLedger,
    TokenUsage,
    UsageSource,
    calculate_cost_usd,
)


def test_calculate_cost_separates_cached_input() -> None:
    price = ModelPrice(
        input_per_million_usd=2.0,
        output_per_million_usd=8.0,
        cached_input_per_million_usd=0.5,
    )
    cost = calculate_cost_usd(
        input_tokens=1_000_000,
        output_tokens=100_000,
        cached_input_tokens=200_000,
        price=price,
    )
    assert cost == 2.5


def test_unconfigured_price_returns_none() -> None:
    assert (
        calculate_cost_usd(
            input_tokens=100,
            output_tokens=10,
            price=ModelPrice(),
        )
        is None
    )


def test_provider_usage_is_preserved_and_aggregated() -> None:
    usage = TokenUsage.from_openai_usage(
        stage="generator",
        model="test-model",
        usage={
            "prompt_tokens": 120,
            "completion_tokens": 30,
            "prompt_tokens_details": {"cached_tokens": 20},
        },
    )
    assert usage.source == UsageSource.PROVIDER
    assert usage.total_tokens == 150
    assert usage.cached_input_tokens == 20

    ledger = TokenLedger()
    ledger.add(usage)
    ledger.add(
        TokenUsage(
            stage="router",
            model="router",
            source=UsageSource.TOKENIZER,
            input_tokens=10,
        )
    )
    assert ledger.totals()["total_tokens"] == 160
    assert ledger.totals()["cost_usd"] is None
    assert ledger.totals()["cost_coverage"] == 0.0
