from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class Chunk:
    chunk_id: str
    document_id: str
    company: str
    year: int
    section: str
    page: int | None
    text: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> Chunk:
        return cls(
            chunk_id=str(value["chunk_id"]),
            document_id=str(value["document_id"]),
            company=str(value["company"]),
            year=int(value["year"]),
            section=str(value.get("section", "")),
            page=int(value["page"]) if value.get("page") is not None else None,
            text=str(value["text"]),
        )


@dataclass(frozen=True, slots=True)
class SearchResult:
    chunk: Chunk
    score: float
    rank: int
    source: str


@dataclass(frozen=True, slots=True)
class EvaluationQuery:
    query_id: str
    query: str
    relevant_chunk_ids: tuple[str, ...]
    question_type: str

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> EvaluationQuery:
        return cls(
            query_id=str(value["query_id"]),
            query=str(value["query"]),
            relevant_chunk_ids=tuple(map(str, value["relevant_chunk_ids"])),
            question_type=str(value.get("question_type", "unknown")),
        )
