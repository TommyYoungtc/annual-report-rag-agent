from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Protocol

try:
    from fastapi import FastAPI, HTTPException, Request
    from fastapi.responses import FileResponse
    from pydantic import BaseModel, Field
except ImportError as error:
    raise RuntimeError(
        'API dependencies are missing. Install with: python -m pip install -e ".[api]"'
    ) from error

from .runtime import AnnualReportAgentRuntime, RuntimeSettings

WEB_ROOT = Path(__file__).resolve().parent / "web"


class AgentService(Protocol):
    scope: Any

    def ask(self, query: str) -> Any: ...

    def close(self) -> None: ...


class QueryRequest(BaseModel):
    query: str = Field(min_length=2, max_length=500)


def _response_dict(response: Any) -> dict[str, Any]:
    return response.to_dict() if hasattr(response, "to_dict") else dict(response)


def create_app(
    service: AgentService | None = None,
    *,
    project_root: Path | None = None,
) -> FastAPI:
    supplied_service = service
    root = project_root or Path(
        os.environ.get("ANNUAL_REPORT_PROJECT_ROOT", Path.cwd())
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        runtime = supplied_service or AnnualReportAgentRuntime(
            RuntimeSettings.from_project(root)
        )
        app.state.agent = runtime
        try:
            yield
        finally:
            if supplied_service is None:
                runtime.close()

    app = FastAPI(
        title="年报证据研究 Agent",
        version="0.1.0",
        description="本地 Hybrid + Reranker 年报问答与引用服务",
        lifespan=lifespan,
    )

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(WEB_ROOT / "index.html")

    @app.get("/api/health")
    def health(request: Request) -> dict[str, Any]:
        agent = request.app.state.agent
        return {
            "status": "ok",
            "model_mode": "local_sequential_gpu",
            "companies": list(agent.scope.companies),
            "years": list(agent.scope.years),
        }

    @app.get("/api/scope")
    def scope(request: Request) -> dict[str, Any]:
        agent = request.app.state.agent
        return {
            "companies": list(agent.scope.companies),
            "years": list(agent.scope.years),
            "supported_fields": [
                "营业收入",
                "归母净利润",
                "经营现金流",
                "归母净资产",
                "研发投入",
                "员工与技术人员",
                "每10股现金红利",
            ],
        }

    @app.post("/api/query")
    def query(payload: QueryRequest, request: Request) -> dict[str, Any]:
        text = payload.query.strip()
        if len(text) < 2:
            raise HTTPException(status_code=422, detail="问题不能为空")
        try:
            return _response_dict(request.app.state.agent.ask(text))
        except (ValueError, RuntimeError) as error:
            raise HTTPException(status_code=500, detail=str(error)) from error

    return app


app = create_app()
