from __future__ import annotations

import json
import urllib.error
import urllib.request
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from ..telemetry import ModelPrice, TokenUsage


@dataclass(frozen=True, slots=True)
class GenerationResult:
    text: str
    usage: TokenUsage
    raw_finish_reason: str | None = None


class OpenAICompatibleGenerator:
    """Minimal OpenAI-compatible chat client that preserves provider usage."""

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str,
        timeout_seconds: float = 120.0,
        price: ModelPrice | None = None,
    ) -> None:
        if not base_url.startswith(("http://", "https://")):
            raise ValueError("base_url must be HTTP(S)")
        if not api_key:
            raise ValueError("api_key cannot be empty")
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.api_key = api_key
        self.model = model
        self.timeout_seconds = timeout_seconds
        self.price = price or ModelPrice()

    def generate(
        self,
        messages: Sequence[Mapping[str, str]],
        *,
        temperature: float = 0.0,
        max_output_tokens: int = 512,
    ) -> GenerationResult:
        payload = {
            "model": self.model,
            "messages": list(messages),
            "temperature": temperature,
            "max_tokens": max_output_tokens,
        }
        request = urllib.request.Request(
            self.url,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(
                request,
                timeout=self.timeout_seconds,
            ) as response:
                body: dict[str, Any] = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as error:
            detail = error.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"generation API returned HTTP {error.code}: {detail}") from error
        choices = body.get("choices") or []
        if not choices:
            raise RuntimeError("generation API returned no choices")
        choice = choices[0]
        text = str(choice.get("message", {}).get("content", ""))
        usage = TokenUsage.from_openai_usage(
            stage="generator",
            model=self.model,
            usage=body.get("usage") or {},
            price=self.price,
        )
        return GenerationResult(
            text=text,
            usage=usage,
            raw_finish_reason=choice.get("finish_reason"),
        )
