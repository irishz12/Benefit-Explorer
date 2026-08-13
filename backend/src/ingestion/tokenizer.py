"""Token counting helpers with a deterministic fallback."""

from __future__ import annotations

import re
from typing import Protocol


class TokenCounter(Protocol):
    """The tokenizer behavior needed by the chunker."""

    def count(self, text: str) -> int:
        """Count tokens in text."""

    def tail(self, text: str, token_count: int) -> str:
        """Return at most the last ``token_count`` tokens of text."""

    def split(self, text: str, token_count: int) -> list[str]:
        """Split text into pieces no larger than ``token_count`` tokens."""


class TiktokenCounter:
    """Token counter based on OpenAI's ``cl100k_base`` encoding."""

    def __init__(self) -> None:
        import tiktoken

        self._encoding = tiktoken.get_encoding("cl100k_base")

    def count(self, text: str) -> int:
        return len(self._encoding.encode(text))

    def tail(self, text: str, token_count: int) -> str:
        tokens = self._encoding.encode(text)
        return self._encoding.decode(tokens[-token_count:]).strip()

    def split(self, text: str, token_count: int) -> list[str]:
        tokens = self._encoding.encode(text)
        return [
            self._encoding.decode(tokens[index : index + token_count]).strip()
            for index in range(0, len(tokens), token_count)
        ]


class RegexTokenCounter:
    """Dependency-free approximation used only if tiktoken is unavailable."""

    _TOKEN_PATTERN = re.compile(r"\w+|[^\w\s]", re.UNICODE)

    def count(self, text: str) -> int:
        return len(self._TOKEN_PATTERN.findall(text))

    def tail(self, text: str, token_count: int) -> str:
        matches = list(self._TOKEN_PATTERN.finditer(text))
        if len(matches) <= token_count:
            return text.strip()
        return text[matches[-token_count].start() :].strip()

    def split(self, text: str, token_count: int) -> list[str]:
        matches = list(self._TOKEN_PATTERN.finditer(text))
        if not matches:
            return []
        pieces: list[str] = []
        for index in range(0, len(matches), token_count):
            start = matches[index].start()
            end_index = min(index + token_count, len(matches)) - 1
            end = matches[end_index].end()
            pieces.append(text[start:end].strip())
        return pieces


def build_token_counter() -> TokenCounter:
    """Build the preferred tokenizer, falling back for offline environments."""

    try:
        return TiktokenCounter()
    except (ImportError, OSError):
        return RegexTokenCounter()
    except Exception:
        # Some tiktoken releases download their encoding table on first use.
        # Ingestion must remain runnable in offline and air-gapped deployments.
        return RegexTokenCounter()
