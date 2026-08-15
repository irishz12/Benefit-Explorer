from __future__ import annotations

from types import SimpleNamespace

import pytest

from src.generation.citation_utils import CitationValidationError
from src.generation.generator import BedrockGenerator
from src.retrieval.models import ChunkRecord, HybridResult


def _context() -> HybridResult:
    record = ChunkRecord(
        chunk_id="chunk_real",
        product_name="Kotak EDGE",
        page_number=2,
        page_numbers=(2, 3),
        section_type="Benefits",
        section_types=("Benefits",),
        source_file="edge.pdf",
        text="The maturity benefit shall be payable at the end of the policy term.",
    )
    return HybridResult(record, 1.0, None, None, None, None)


class _FakeCompletions:
    """Returns the next queued response; raises if called more times than expected."""

    def __init__(self, responses: list[str]) -> None:
        self._responses = list(responses)
        self.calls = 0

    def create(self, **_: object) -> SimpleNamespace:
        self.calls += 1
        content = self._responses.pop(0)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
        )


class _FakeClient:
    def __init__(self, responses: list[str]) -> None:
        self.chat = SimpleNamespace(completions=_FakeCompletions(responses))


def test_exhausted_invalid_generations_fail_without_plain_text_fallback() -> None:
    """Once structured JSON retries are exhausted, the generator must raise
    CitationValidationError immediately instead of issuing an extra plain-text
    completion request."""

    # One initial call plus one JSON-correction retry (max_json_retries=1);
    # both responses are unparseable plain text. Only two responses are queued,
    # so a third completion call (the removed plain-text fallback) would raise
    # IndexError instead of silently succeeding.
    client = _FakeClient(
        [
            "This is not JSON at all [1].",
            "Still not JSON [1].",
        ]
    )
    generator = BedrockGenerator(
        model="test-model",
        client=client,
        max_retries=1,
        max_json_retries=1,
    )

    with pytest.raises(CitationValidationError):
        generator.generate("What is the maturity benefit?", [_context()])

    assert client.chat.completions.calls == 2
