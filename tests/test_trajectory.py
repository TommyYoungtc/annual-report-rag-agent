from annual_report_agent.agent import RetrievalStep, RoundCost, summarize_usage


def test_summarize_usage_adds_round_costs() -> None:
    steps = [
        RetrievalStep(
            round_index=index,
            search_query="query",
            evidence_chunk_ids=(),
            evidence_scores=(),
            answer_status="answered",
            answer_reason=None,
            assessment_status="sufficient",
            assessment_reasons=("ok",),
            action="stop" if index == 2 else "retrieve_more",
            action_reason="test",
            cost=RoundCost(
                retrieval_ms=10.0,
                rerank_ms=20.0,
                dense_queries=1,
                bm25_queries=1,
                reranker_calls=1,
                candidates_retrieved=30,
                passages_reranked=10,
            ),
        )
        for index in (1, 2)
    ]
    usage = summarize_usage(steps)
    assert usage.retrieval_rounds == 2
    assert usage.reranker_calls == 2
    assert usage.passages_reranked == 20
    assert usage.amortized_wall_clock_ms == 60.0
