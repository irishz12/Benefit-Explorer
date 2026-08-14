"""Strict validation for model-generated answers and citations."""

from __future__ import annotations

import ast
import json
import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Any, Mapping, Sequence

from src.retrieval.product_detection import ProductDetector
from src.retrieval.models import HybridResult

_INLINE_CITATION = re.compile(r"\[(\d+)]")
_FUZZY_MATCH_THRESHOLD = 0.82
_BULLET_MARKER = re.compile(r"^[•●▪◦–—-]\s*")
_PAGE_NUMBER = re.compile(r"^\d{1,3}$")
_SECTION_HEADING = re.compile(r"^(?:\d+(?:\.\d+)*\.?\s+)?[A-Z][A-Za-z0-9 &'()@/+\-]{2,60}:?$")
_QUOTE_TRANSLATION = str.maketrans(
    {
        "\u00a0": " ",
        "‘": "'",
        "’": "'",
        "‚": "'",
        "‛": "'",
        "`": "'",
        "´": "'",
        "“": '"',
        "”": '"',
        "„": '"',
        "‟": '"',
    }
)


class CitationValidationError(ValueError):
    """Raised when generated citations are not grounded in final context."""

    def __init__(
        self,
        message: str,
        *,
        payload: str | Mapping[str, Any] | None = None,
        reasons: Sequence[str] = (),
        contexts: Sequence[HybridResult] = (),
    ) -> None:
        super().__init__(message)
        self.payload = payload
        self.reasons = tuple(reasons) or (message,)
        self.contexts = tuple(contexts)
        self.detected_products: tuple[str, ...] = ()
        self.product_retrieval_mode = "none"


@dataclass(frozen=True, slots=True)
class VerifiedCitation:
    index: int
    chunk_id: str
    product: str
    page: int
    supporting_text: str


@dataclass(frozen=True, slots=True)
class VerifiedAnswer:
    answer: str
    citations: tuple[VerifiedCitation, ...]


@dataclass(frozen=True, slots=True)
class ProductCitationCoverage:
    """Verified citation indices for one requested or discussed product."""

    product: str
    citation_indices: tuple[int, ...]

    @property
    def covered(self) -> bool:
        return bool(self.citation_indices)


def _normalize(text: str) -> str:
    normalized = text.translate(_QUOTE_TRANSLATION)
    normalized = re.sub(r"\s+", " ", normalized)
    normalized = re.sub(r"\s+([,.;:!?])", r"\1", normalized)
    return normalized.strip().casefold()


def _candidate_passages(chunk_text: str, support_length: int) -> list[str]:
    """Build sentence and token windows for fuzzy quotation matching."""

    sentences = [
        _normalize(sentence)
        for sentence in re.split(r"(?<=[.!?])\s+|\n+", chunk_text)
        if sentence.strip()
    ]
    passages: list[str] = []
    for start in range(len(sentences)):
        for width in range(1, 4):
            passage = " ".join(sentences[start : start + width]).strip()
            if passage:
                passages.append(passage)

    tokens = _normalize(chunk_text).split()
    if support_length and tokens:
        window_sizes = {max(4, round(support_length * ratio)) for ratio in (0.85, 1.0, 1.15)}
        step = max(1, support_length // 6)
        for window_size in window_sizes:
            for start in range(0, max(1, len(tokens) - window_size + 1), step):
                passages.append(" ".join(tokens[start : start + window_size]))
            if len(tokens) > window_size:
                passages.append(" ".join(tokens[-window_size:]))
    return passages


def _support_match(supporting_text: str, chunk_text: str) -> tuple[bool, str, float]:
    """Return exact/close match details for a supporting quotation."""

    support = _normalize(supporting_text)
    chunk = _normalize(chunk_text)
    if support and support in chunk:
        return True, "EXACT", 1.0
    if not support:
        return False, "NONE", 0.0
    passages = _candidate_passages(chunk_text, len(support.split()))
    similarity = max(
        (SequenceMatcher(None, support, passage).ratio() for passage in passages),
        default=0.0,
    )
    if similarity >= _FUZZY_MATCH_THRESHOLD:
        return True, "CLOSE", similarity
    return False, "NONE", similarity


def _product_citation_checks(
    answer_text: str,
    citations: Sequence[VerifiedCitation],
    contexts: Sequence[HybridResult],
    required_products: Sequence[str] = (),
) -> tuple[ProductCitationCoverage, ...]:
    """Map requested/mentioned products to citations backed by their chunks."""

    context_products = tuple(dict.fromkeys(result.record.product_name for result in contexts))
    detector_names = tuple(dict.fromkeys((*context_products, *required_products)))
    mentioned_products = (
        ProductDetector(detector_names).detect(answer_text) if detector_names else ()
    )
    products = tuple(dict.fromkeys((*required_products, *mentioned_products)))
    context_by_chunk = {result.record.chunk_id: result.record for result in contexts}
    checks: list[ProductCitationCoverage] = []
    for product in products:
        indices = tuple(
            dict.fromkeys(
                citation.index
                for citation in citations
                if (record := context_by_chunk.get(citation.chunk_id)) is not None
                and record.product_name == product
            )
        )
        checks.append(
            ProductCitationCoverage(
                product=product,
                citation_indices=indices,
            )
        )
    return tuple(checks)


def parse_and_verify_answer(
    payload: str | Mapping[str, Any],
    contexts: Sequence[HybridResult],
    required_products: Sequence[str] = (),
) -> VerifiedAnswer:
    """Validate and canonicalize citations using exact or fuzzy support matches."""

    if isinstance(payload, str):
        try:
            data = json.loads(payload)
        except json.JSONDecodeError as exc:
            reason = f"Response is not valid JSON: {exc}"
            raise CitationValidationError(
                reason, payload=payload, reasons=(reason,), contexts=contexts
            ) from exc
    else:
        data = dict(payload)

    answer = data.get("answer")
    raw_citations = data.get("citations")
    if not isinstance(answer, str) or not answer.strip():
        reason = "'answer' must be a non-empty string"
        raise CitationValidationError(reason, payload=data, reasons=(reason,), contexts=contexts)
    if not isinstance(raw_citations, list):
        reason = "'citations' must be an array"
        raise CitationValidationError(reason, payload=data, reasons=(reason,), contexts=contexts)

    by_index = {index: result.record for index, result in enumerate(contexts, start=1)}
    verified: list[VerifiedCitation] = []
    errors: list[str] = []
    seen_declared_indices: set[int] = set()
    for position, citation in enumerate(raw_citations, start=1):
        if not isinstance(citation, Mapping):
            errors.append(f"citation {position} is not an object")
            continue
        try:
            index = int(citation.get("index"))
            page = int(citation.get("page"))
        except (TypeError, ValueError):
            errors.append(f"citation {position} has an invalid index or page")
            continue
        if index in seen_declared_indices:
            # Models sometimes repeat the same context object for two claims.
            # One inline marker already identifies that source, so retain the
            # first object instead of paying for a formatting-only retry.
            continue
        seen_declared_indices.add(index)

        record = by_index.get(index)
        if record is None:
            errors.append(
                f"citation {index} failed: index is outside the supplied context range "
                f"1-{len(by_index)}"
            )
            continue

        declared_chunk_id = str(citation.get("chunk_id", "")).strip()
        declared_product = str(citation.get("product", "")).strip()
        if declared_chunk_id != record.chunk_id:
            errors.append(
                f"citation {index} failed: chunk_id {declared_chunk_id!r} does not match "
                f"context {index} ({record.chunk_id})"
            )
            continue
        if declared_product != record.product_name:
            errors.append(
                f"citation {index} failed: product {declared_product!r} does not match "
                f"context {index} ({record.product_name!r})"
            )
            continue
        if page not in record.page_numbers:
            errors.append(
                f"citation {index} failed: page {page} is not in context {index} pages "
                f"{list(record.page_numbers)}"
            )
            continue

        supporting_text = str(citation.get("supporting_text", "")).strip()
        if len(supporting_text) < 5:
            errors.append(f"citation {index} supporting_text is too short")
            continue

        matched, _, similarity = _support_match(supporting_text, record.text)
        if not matched:
            errors.append(
                f"citation {index} failed: supporting_text did not reach the fuzzy threshold "
                f"{_FUZZY_MATCH_THRESHOLD:.0%} in its declared context "
                f"({record.chunk_id}); similarity was {similarity:.1%}"
            )
            continue

        canonical_support = _best_supporting_sentence(
            supporting_text,
            record.text,
        )
        verified.append(
            VerifiedCitation(
                index,
                record.chunk_id,
                record.product_name,
                page,
                canonical_support,
            )
        )

    inline_indices = {int(value) for value in _INLINE_CITATION.findall(answer)}
    if inline_indices and inline_indices != seen_declared_indices:
        errors.append(
            f"inline citation indices {sorted(inline_indices)} do not match citation objects "
            f"{sorted(seen_declared_indices)}"
        )
    deduplicated: dict[int, VerifiedCitation] = {}
    for citation in verified:
        deduplicated.setdefault(citation.index, citation)
    verified = list(deduplicated.values())
    verified_indices = {citation.index for citation in verified}
    if not inline_indices and verified_indices:
        markers = "".join(f"[{index}]" for index in sorted(verified_indices))
        answer = f"{answer.rstrip()} {markers}"
    rewritten_inline_indices = {int(value) for value in _INLINE_CITATION.findall(answer)}
    if rewritten_inline_indices != verified_indices:
        errors.append(
            f"canonical inline indices {sorted(rewritten_inline_indices)} do not match verified "
            f"citations {sorted(verified_indices)}"
        )
    insufficiency_signals = (
        "cannot determine",
        "not enough information",
        "insufficient context",
        "context does not",
        "not provided in the context",
    )
    if not verified and not any(signal in answer.casefold() for signal in insufficiency_signals):
        errors.append("a substantive answer must include at least one verified citation")
    product_checks = _product_citation_checks(
        answer,
        verified,
        contexts,
        required_products,
    )
    missing_product_citations = [check.product for check in product_checks if not check.covered]
    if missing_product_citations:
        errors.append(
            "missing a verified citation from the product's own chunks for: "
            + ", ".join(missing_product_citations)
        )
    if errors:
        raise CitationValidationError(
            "; ".join(errors),
            payload=data,
            reasons=errors,
            contexts=contexts,
        )
    return VerifiedAnswer(answer=answer.strip(), citations=tuple(verified))


def _extract_json_mapping(raw_output: str) -> Mapping[str, Any] | None:
    """Recover a JSON/Python-style object embedded in otherwise plain output."""

    stripped = raw_output.strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)```", stripped, flags=re.DOTALL | re.IGNORECASE)
    candidates = [fenced.group(1).strip()] if fenced else []
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start >= 0 and end > start:
        candidates.append(stripped[start : end + 1])
    candidates.append(stripped)
    for candidate in dict.fromkeys(candidates):
        for parser in (json.loads, ast.literal_eval):
            try:
                decoded = parser(candidate)
            except (json.JSONDecodeError, SyntaxError, ValueError):
                continue
            if isinstance(decoded, Mapping):
                return decoded
    return None


def _split_plain_answer(raw_output: str) -> str:
    """Remove a trailing SOURCES/CITATIONS block from fallback output."""

    cleaned = re.sub(r"```(?:text|json)?", "", raw_output, flags=re.IGNORECASE)
    cleaned = cleaned.replace("```", "").strip()
    answer_match = re.search(
        r"(?:^|\n)\s*ANSWER\s*:\s*(.*?)(?=\n\s*(?:CITATIONS?|SOURCES?)\s*:|\Z)",
        cleaned,
        flags=re.DOTALL | re.IGNORECASE,
    )
    if answer_match:
        return answer_match.group(1).strip()
    return re.split(
        r"\n\s*(?:CITATIONS?|SOURCES?)\s*:\s*",
        cleaned,
        maxsplit=1,
        flags=re.IGNORECASE,
    )[0].strip()


def _claim_for_index(answer: str, index: int) -> str:
    """Return the answer sentence nearest an inline citation marker."""

    marker = f"[{index}]"
    sentences = re.split(r"(?<=[.!?])\s+|\n+", answer)
    for sentence in sentences:
        if marker in sentence:
            return _INLINE_CITATION.sub("", sentence).strip()
    return _INLINE_CITATION.sub("", answer).strip()


def _citation_passages(chunk_text: str) -> list[str]:
    """Reconstruct readable source clauses from PDF-wrapped chunk text.

    PDF extraction commonly inserts a newline (and sometimes a blank line)
    after every visual line. Treating those newlines as sentence boundaries
    produces source-card fragments such as ``"till the end ..."``. This
    routine removes standalone page numbers and short section headings, joins
    wrapped lines, and then splits only at punctuation or real bullet markers.
    """

    lines: list[str] = []
    for raw_line in chunk_text.splitlines():
        line = re.sub(r"\s+", " ", raw_line).strip()
        if not line or _PAGE_NUMBER.fullmatch(line):
            continue
        word_count = len(line.split())
        if (
            word_count <= 8
            and _SECTION_HEADING.fullmatch(line)
            and not re.search(r"[.!?]$", line)
            and not _BULLET_MARKER.match(line)
        ):
            continue
        lines.append(line)

    collapsed = " ".join(lines).strip()
    if not collapsed:
        return []
    passages = re.split(
        r"(?<=[.!?])\s+|\s+(?=[•●▪◦–—-]\s*)",
        collapsed,
    )
    return [
        _BULLET_MARKER.sub("", passage).strip()
        for passage in passages
        if len(_BULLET_MARKER.sub("", passage).strip()) >= 5
    ]


def _best_supporting_sentence(claim: str, chunk_text: str) -> str:
    """Select an exact chunk sentence most lexically related to a cited claim."""

    sentences = _citation_passages(chunk_text)
    if not sentences:
        return re.sub(r"\s+", " ", chunk_text).strip()
    claim_tokens = set(re.findall(r"[a-z0-9]+", _normalize(claim)))

    def score(sentence: str) -> tuple[float, float]:
        sentence_tokens = set(re.findall(r"[a-z0-9]+", _normalize(sentence)))
        overlap = len(claim_tokens & sentence_tokens) / max(1, len(claim_tokens))
        similarity = SequenceMatcher(None, _normalize(claim), _normalize(sentence)).ratio()
        return overlap, similarity

    return max(sentences, key=score)


def recover_answer_payload(
    raw_output: str,
    contexts: Sequence[HybridResult],
) -> Mapping[str, Any]:
    """Recover schema-like data from rejected JSON or a plain-text fallback.

    Inline ``[N]`` markers are mapped to the actual numbered contexts. Supporting
    text is selected verbatim from each referenced chunk so the result still has
    to pass the normal strict citation and product-coverage validator.
    """

    recovered = _extract_json_mapping(raw_output)
    if recovered is not None:
        return recovered

    answer = _split_plain_answer(raw_output)
    inline_values = _INLINE_CITATION.findall(answer)
    marker_source = answer if inline_values else raw_output
    indices = tuple(dict.fromkeys(int(value) for value in _INLINE_CITATION.findall(marker_source)))
    citations: list[dict[str, Any]] = []
    for index in indices:
        if index < 1 or index > len(contexts):
            continue
        record = contexts[index - 1].record
        supporting_text = _best_supporting_sentence(
            _claim_for_index(answer, index),
            record.text,
        )
        citations.append(
            {
                "index": index,
                "chunk_id": record.chunk_id,
                "product": record.product_name,
                "page": record.page_number,
                "supporting_text": supporting_text,
            }
        )
    return {"answer": answer, "citations": citations}
