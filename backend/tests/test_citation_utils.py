from __future__ import annotations

import pytest

from src.generation.citation_utils import CitationValidationError, parse_and_verify_answer
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
