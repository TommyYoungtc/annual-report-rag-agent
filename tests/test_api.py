from __future__ import annotations

import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from annual_report_agent.agent import CorpusScope
from annual_report_agent.api import create_app


class FakeService:
    scope = CorpusScope(("宁德时代", "比亚迪", "科大讯飞"), (2024, 2025))

    def ask(self, query: str):
        return {
            "query": query,
            "status": "answered",
            "answer": "362,012,554千元",
            "reason": None,
            "route": {
                "companies": ["宁德时代"],
                "years": [2024],
                "task_type": "single_fact",
                "requires_calculation": False,
                "should_refuse": False,
                "refusal_reason": None,
            },
            "citations": [
                {
                    "chunk_id": "catl-2024:0009",
                    "document_id": "catl-2024",
                    "company": "宁德时代",
                    "year": 2024,
                    "page": 9,
                    "quote": "营业收入（千元）362,012,554",
                }
            ],
            "calculation": None,
            "timing_ms": {"total": 12.0},
            "gpu_used": False,
        }

    def close(self) -> None:
        return None


@pytest.fixture
def client():
    with TestClient(create_app(FakeService())) as test_client:
        yield test_client


def test_serves_demo_page(client: TestClient) -> None:
    response = client.get("/")
    assert response.status_code == 200
    assert "每一个数字" in response.text


def test_health_and_scope(client: TestClient) -> None:
    health = client.get("/api/health")
    assert health.status_code == 200
    assert health.json()["companies"] == ["宁德时代", "比亚迪", "科大讯飞"]
    scope = client.get("/api/scope")
    assert "营业收入" in scope.json()["supported_fields"]


def test_query_returns_structured_citation(client: TestClient) -> None:
    response = client.post("/api/query", json={"query": "宁德时代2024年营业收入是多少？"})
    assert response.status_code == 200
    assert response.json()["answer"] == "362,012,554千元"
    assert response.json()["citations"][0]["page"] == 9


def test_query_validates_empty_text(client: TestClient) -> None:
    response = client.post("/api/query", json={"query": " "})
    assert response.status_code == 422
