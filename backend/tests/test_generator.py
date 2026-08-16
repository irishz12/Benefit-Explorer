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


class _RecordingCompletions:
    """Like _FakeCompletions, but also captures the messages sent to the model."""

    def __init__(self, responses: list[str]) -> None:
        self._responses = list(responses)
        self.last_messages: list[dict[str, str]] | None = None

    def create(self, **kwargs: object) -> SimpleNamespace:
        self.last_messages = kwargs["messages"]
        content = self._responses.pop(0)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
        )


class _RecordingClient:
    def __init__(self, responses: list[str]) -> None:
        self.chat = SimpleNamespace(completions=_RecordingCompletions(responses))


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


def test_format_context_exposes_a_canonical_citation_page() -> None:
    """The model must be given one explicit page to copy per context, instead
    of picking a page out of the informational `pages` list (or the context
    index) itself — this is the recurring `page is not in context` failure."""

    context = _context()  # page_number=2, page_numbers=(2, 3)

    formatted = BedrockGenerator._format_context([context])

    assert "citation_page: 2" in formatted
    assert "pages: 2, 3" in formatted
    # citation_page must be stated before the informational pages list.
    assert formatted.index("citation_page: 2") < formatted.index("pages: 2, 3")


def test_system_prompt_states_the_citation_page_rule() -> None:
    """The generation prompt must direct the model to copy citation_page and
    never confuse it with the citation index, since that confusion is exactly
    what triggers the extra citation-correction retry being eliminated here."""

    valid_response = (
        '{"answer": "A maturity benefit is payable [1].", '
        '"citations": [{"index": 1, "chunk_id": "chunk_real", "product": "Kotak EDGE", '
        '"page": 2, "supporting_text": '
        '"The maturity benefit shall be payable at the end of the policy term."}]}'
    )
    client = _RecordingClient([valid_response])
    generator = BedrockGenerator(model="test-model", client=client)

    generator.generate("What is the maturity benefit?", [_context()])

    system_message = client.chat.completions.last_messages[0]["content"]
    assert "citation_page" in system_message
    assert "never confuse it with the citation" in system_message


def test_system_prompt_forbids_trailing_prose_after_the_json() -> None:
    """Q015 emitted a valid JSON object followed by explanatory prose. The prompt
    must explicitly forbid anything after the closing brace so the extra text
    does not appear in the first place."""

    valid_response = (
        '{"answer": "A maturity benefit is payable [1].", '
        '"citations": [{"index": 1, "chunk_id": "chunk_real", "product": "Kotak EDGE", '
        '"page": 2, "supporting_text": '
        '"The maturity benefit shall be payable at the end of the policy term."}]}'
    )
    client = _RecordingClient([valid_response])
    generator = BedrockGenerator(model="test-model", client=client)

    generator.generate("What is the maturity benefit?", [_context()])

    system_message = client.chat.completions.last_messages[0]["content"]
    assert "nothing after the" in system_message
    assert "closing brace" in system_message
