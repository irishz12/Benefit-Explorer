"""Golden dataset loading and active-index validation."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


@dataclass(frozen=True, slots=True)
class EvidenceGroup:
    """Alternative chunks that support the same material answer evidence."""

    evidence_id: str
    description: str
    chunk_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class GoldenQuestion:
    """One customer question with a reference answer and labeled evidence."""

    question_id: str
    question: str
    reference_answer: str
    relevant_chunk_ids: tuple[str, ...]
    relevant_evidence_groups: tuple[EvidenceGroup, ...] = ()
    product: str | None = None
    category: str | None = None

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "GoldenQuestion":
        def required_text(field: str) -> str:
            value = payload.get(field)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{field!r} must be a non-empty string")
            return value.strip()

        relevant = _string_list(payload.get("relevant_chunk_ids"), "relevant_chunk_ids")
        if not relevant:
            raise ValueError("'relevant_chunk_ids' must contain at least one chunk ID")
        product = payload.get("product")
        category = payload.get("category") or payload.get("question_type")
        return cls(
            question_id=required_text("question_id"),
            question=required_text("question"),
            reference_answer=required_text("reference_answer"),
            relevant_chunk_ids=tuple(dict.fromkeys(relevant)),
            product=product.strip() if isinstance(product, str) and product.strip() else None,
            category=(category.strip() if isinstance(category, str) and category.strip() else None),
        )


@dataclass(frozen=True, slots=True)
class EvaluationSplits:
    """Frozen development and holdout question identifiers."""

    dev_question_ids: tuple[str, ...]
    holdout_question_ids: tuple[str, ...]


def _string_list(value: object, field: str) -> list[str]:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError(f"{field!r} must be an array of strings")
    return [item.strip() for item in value if item.strip()]


def load_golden_dataset(path: Path) -> list[GoldenQuestion]:
    """Load golden questions and their evidence-group relevance labels."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise FileNotFoundError(f"Golden dataset not found: {path}") from error
    if not isinstance(payload, list):
        raise ValueError(f"Expected a JSON array in {path}")
    questions = [GoldenQuestion.from_dict(item) for item in payload]
    ids = [question.question_id for question in questions]
    duplicates = sorted(question_id for question_id in set(ids) if ids.count(question_id) > 1)
    if duplicates:
        raise ValueError(f"Duplicate question IDs: {duplicates}")
    groups_path = path.with_name("evidence_groups.json")
    return _attach_evidence_groups(questions, groups_path)


def load_evaluation_splits(
    path: Path,
    questions: Sequence[GoldenQuestion],
) -> EvaluationSplits:
    """Load and validate the immutable dev/holdout partition."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise FileNotFoundError(f"Evaluation split manifest not found: {path}") from error
    if not isinstance(payload, Mapping):
        raise ValueError(f"Expected a JSON object in {path}")
    dev_ids = tuple(_string_list(payload.get("dev_question_ids"), "dev_question_ids"))
    holdout_ids = tuple(_string_list(payload.get("holdout_question_ids"), "holdout_question_ids"))
    if len(dev_ids) != 20 or len(holdout_ids) != 10:
        raise ValueError("Frozen evaluation split must contain 20 dev and 10 holdout IDs")
    if set(dev_ids).intersection(holdout_ids):
        raise ValueError("Development and holdout splits must be disjoint")
    golden_ids = {question.question_id for question in questions}
    if set(dev_ids).union(holdout_ids) != golden_ids:
        raise ValueError("Frozen splits must cover the golden dataset exactly")
    return EvaluationSplits(dev_ids, holdout_ids)


def _attach_evidence_groups(
    questions: Sequence[GoldenQuestion],
    path: Path,
) -> list[GoldenQuestion]:
    """Attach sidecar evidence groups and ensure they cover exactly the labels."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise FileNotFoundError(f"Evidence-group labels not found: {path}") from error
    raw_questions = payload.get("questions") if isinstance(payload, Mapping) else None
    if not isinstance(raw_questions, Mapping):
        raise ValueError(f"Expected a 'questions' object in {path}")

    known_ids = {question.question_id for question in questions}
    unknown_ids = sorted(set(raw_questions) - known_ids)
    if unknown_ids:
        raise ValueError(f"Evidence groups reference unknown questions: {unknown_ids}")

    attached: list[GoldenQuestion] = []
    for question in questions:
        raw_groups = raw_questions.get(question.question_id)
        if not isinstance(raw_groups, list) or not raw_groups:
            raise ValueError(f"{question.question_id} requires at least one evidence group")
        groups: list[EvidenceGroup] = []
        for position, raw_group in enumerate(raw_groups, start=1):
            if not isinstance(raw_group, Mapping):
                raise ValueError(f"{question.question_id} evidence group {position} is invalid")
            chunk_ids = tuple(dict.fromkeys(_string_list(raw_group.get("chunk_ids"), "chunk_ids")))
            if not chunk_ids:
                raise ValueError(
                    f"{question.question_id} evidence group {position} has no chunk IDs"
                )
            evidence_id = str(raw_group.get("evidence_id", f"E{position}")).strip()
            description = str(raw_group.get("description", "")).strip()
            groups.append(EvidenceGroup(evidence_id, description, chunk_ids))

        flattened = {chunk_id for group in groups for chunk_id in group.chunk_ids}
        expected = set(question.relevant_chunk_ids)
        if flattened != expected:
            raise ValueError(
                f"{question.question_id} evidence groups must cover exactly its "
                "relevant_chunk_ids"
            )
        attached.append(
            GoldenQuestion(
                question_id=question.question_id,
                question=question.question,
                reference_answer=question.reference_answer,
                relevant_chunk_ids=question.relevant_chunk_ids,
                relevant_evidence_groups=tuple(groups),
                product=question.product,
                category=question.category,
            )
        )
    return attached


def validate_relevant_chunk_ids(
    questions: Sequence[GoldenQuestion],
    available_chunk_ids: set[str],
) -> None:
    """Reject golden labels that do not exist in the active chunk collection."""

    missing = {
        question.question_id: sorted(set(question.relevant_chunk_ids) - available_chunk_ids)
        for question in questions
        if set(question.relevant_chunk_ids) - available_chunk_ids
    }
    if missing:
        details = "; ".join(f"{key}: {value}" for key, value in missing.items())
        raise ValueError(f"Golden dataset references unknown chunk IDs: {details}")
