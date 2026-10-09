from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Protocol


class TokenCounter(Protocol):
    name: str

    def count_text(self, text: str) -> int: ...

    def truncate_text(self, text: str, max_tokens: int) -> str: ...

    def count_messages(
        self,
        messages: Sequence[Mapping[str, str]],
        *,
        add_generation_prompt: bool = True,
    ) -> int: ...


class HuggingFaceTokenCounter:
    """Count post-template tokens with the exact configured local tokenizer."""

    def __init__(self, model_name_or_path: str) -> None:
        try:
            from transformers import AutoTokenizer
        except ImportError as error:
            raise RuntimeError(
                "Exact token counting requires transformers; install the models extra."
            ) from error
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.name = str(model_name_or_path)

    def count_text(self, text: str) -> int:
        return len(self.tokenizer.encode(text, add_special_tokens=False))

    def truncate_text(self, text: str, max_tokens: int) -> str:
        if max_tokens < 0:
            raise ValueError("max_tokens cannot be negative")
        token_ids = self.tokenizer.encode(
            text,
            add_special_tokens=False,
            truncation=True,
            max_length=max_tokens,
        )
        return self.tokenizer.decode(token_ids, skip_special_tokens=True)

    def count_messages(
        self,
        messages: Sequence[Mapping[str, str]],
        *,
        add_generation_prompt: bool = True,
    ) -> int:
        apply_template = getattr(self.tokenizer, "apply_chat_template", None)
        chat_template = getattr(self.tokenizer, "chat_template", None)
        if callable(apply_template) and chat_template:
            encoded: Any = apply_template(
                list(messages),
                tokenize=True,
                add_generation_prompt=add_generation_prompt,
            )
            if hasattr(encoded, "input_ids"):
                input_ids = encoded.input_ids
                if hasattr(input_ids, "shape"):
                    return int(input_ids.shape[-1])
                return len(input_ids)
            if hasattr(encoded, "shape"):
                return int(encoded.shape[-1])
            return len(encoded)
        flattened = "\n".join(f"<{message['role']}>\n{message['content']}" for message in messages)
        if add_generation_prompt:
            flattened += "\n<assistant>\n"
        return self.count_text(flattened)
