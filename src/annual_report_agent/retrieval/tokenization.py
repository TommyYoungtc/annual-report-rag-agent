from __future__ import annotations

import re

TOKEN_PATTERN = re.compile(r"[A-Za-z]+(?:[-_.][A-Za-z0-9]+)*|\d+(?:\.\d+)?|[\u4e00-\u9fff]+")
CHINESE_PATTERN = re.compile(r"^[\u4e00-\u9fff]+$")


def tokenize(text: str) -> list[str]:
    """Tokenize mixed Chinese/English text without external dependencies.

    Chinese spans produce unigram and bigram features. This remains deterministic
    while distinguishing phrases such as ``营业收入`` and ``研发投入`` better than
    a character-only baseline.
    """

    output: list[str] = []
    for raw_token in TOKEN_PATTERN.findall(text):
        token = raw_token.lower()
        if not CHINESE_PATTERN.fullmatch(token):
            output.append(token)
            continue
        characters = list(token)
        output.extend(characters)
        output.extend(
            characters[index] + characters[index + 1] for index in range(len(characters) - 1)
        )
        if len(characters) <= 8:
            output.append(token)
    return output
