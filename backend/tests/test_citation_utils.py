from __future__ import annotations

import json

import pytest

from src.generation.citation_utils import (
    CitationValidationError,
    parse_and_verify_answer,
    recover_answer_payload,
)
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


def _payload(**overrides: object) -> dict[str, object]:
    citation: dict[str, object] = {
        "index": 1,
        "chunk_id": "chunk_real",
        "product": "Kotak EDGE",
        "page": 2,
        "supporting_text": "The maturity benefit shall be payable at the end of the policy term.",
    }
    citation.update(overrides)
    return {"answer": "A maturity benefit is payable [1].", "citations": [citation]}


def test_accepts_identity_matched_citation() -> None:
    answer = parse_and_verify_answer(_payload(), [_context()])
    assert answer.citations[0].chunk_id == "chunk_real"
    assert answer.citations[0].page == 2


@pytest.mark.parametrize(
    ("field", "value", "reason"),
    [
        ("chunk_id", "chunk_invented", "does not match context"),
        ("product", "Kotak TULIP", "does not match context"),
        ("page", 99, "is not in context"),
        ("index", 2, "outside the supplied context range"),
    ],
)
def test_rejects_mismatched_citation_identity(
    field: str,
    value: object,
    reason: str,
) -> None:
    with pytest.raises(CitationValidationError, match=reason):
        parse_and_verify_answer(_payload(**{field: value}), [_context()])


def test_does_not_resolve_supporting_text_against_another_context() -> None:
    second = _context()
    first_record = ChunkRecord(
        "chunk_other",
        "Kotak EDGE",
        1,
        (1,),
        "Benefits",
        ("Benefits",),
        "edge.pdf",
        "This context discusses eligibility only.",
    )
    first = HybridResult(first_record, 2.0, None, None, None, None)
    payload = _payload(chunk_id="chunk_other", page=1)
    with pytest.raises(CitationValidationError, match="declared context"):
        parse_and_verify_answer(payload, [first, second])


def test_reconstructs_complete_sentence_from_pdf_wrapped_lines() -> None:
    wrapped_text = """Policy Benefits

2. Maturity Benefit

On survival of Life Insured till the end of the policy term provided all

the premiums are paid up to date and the policy is in force, Fund

Value (Main Account + Top up Account, if any) inclusive of Loyalty

Additions shall be payable.

14

Fund Value is payable for a reduced paid-up policy."""
    record = ChunkRecord(
        chunk_id="chunk_tulip_maturity",
        product_name="Kotak TULIP",
        page_number=14,
        page_numbers=(14,),
        section_type="Benefits",
        section_types=("Benefits",),
        source_file="tulip.pdf",
        text=wrapped_text,
    )
    context = HybridResult(record, 1.0, None, None, None, None)
    full_sentence = (
        "On survival of Life Insured till the end of the policy term provided all "
        "the premiums are paid up to date and the policy is in force, Fund Value "
        "(Main Account + Top up Account, if any) inclusive of Loyalty Additions "
        "shall be payable."
    )
    payload = {
        "answer": "Kotak TULIP pays its fund value at maturity [1].",
        "citations": [
            {
                "index": 1,
                "chunk_id": "chunk_tulip_maturity",
                "product": "Kotak TULIP",
                "page": 14,
                "supporting_text": full_sentence,
            }
        ],
    }

    answer = parse_and_verify_answer(payload, [context])

    assert answer.citations[0].supporting_text == full_sentence
    assert answer.citations[0].supporting_text.startswith("On survival")
    assert answer.citations[0].supporting_text.endswith("shall be payable.")


def test_rejects_citation_object_without_inline_marker() -> None:
    """A declared citation object with no matching [N] marker must fail, not
    be silently fixed by appending a marker to the answer."""

    payload = _payload()
    payload["answer"] = "A maturity benefit is payable."
    with pytest.raises(CitationValidationError, match="inline citation markers"):
        parse_and_verify_answer(payload, [_context()])


def test_rejects_inline_marker_without_citation_object() -> None:
    payload = {"answer": "A maturity benefit is payable [1].", "citations": []}
    with pytest.raises(CitationValidationError, match="inline citation markers"):
        parse_and_verify_answer(payload, [_context()])


def test_recovery_does_not_synthesize_citation_from_plain_text() -> None:
    """A plain-text fallback answer with an inline [1] must not cause Python
    to manufacture a citation object from the retrieved context."""

    raw_output = (
        "ANSWER:\nA maturity benefit is payable [1].\n"
        "CITATIONS:\n"
        "[1] | chunk_id: chunk_real | product: Kotak EDGE | page: 2 | "
        "supporting_text: The maturity benefit shall be payable at the end of the policy term."
    )
    recovered = recover_answer_payload(raw_output, [_context()])
    assert recovered["citations"] == []
    with pytest.raises(CitationValidationError, match="inline citation markers"):
        parse_and_verify_answer(recovered, [_context()])


def test_recovers_fenced_json_with_model_produced_citation() -> None:
    """Fenced JSON that already contains model-produced citation data is safe
    to recover and must still pass full verification."""

    raw_output = f"Sure, here is the answer:\n```json\n{json.dumps(_payload())}\n```"
    recovered = recover_answer_payload(raw_output, [_context()])
    answer = parse_and_verify_answer(recovered, [_context()])
    assert answer.citations[0].chunk_id == "chunk_real"
    assert answer.citations[0].page == 2
