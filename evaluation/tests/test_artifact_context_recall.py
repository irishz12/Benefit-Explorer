from evaluation.src.report import aggregate_dual_judges
from evaluation.src.runner import artifact_context_recall


def _artifact(selected: list[str], error: str | None = None) -> dict:
    return {
        "generation_error": error,
        "relevant_evidence_groups": [
            {"evidence_id": "E1", "description": "a", "chunk_ids": ["chunk_a", "chunk_b"]},
            {"evidence_id": "E2", "description": "b", "chunk_ids": ["chunk_c"]},
        ],
        "selected_contexts": [{"chunk_id": chunk_id} for chunk_id in selected],
    }


def test_overlapping_chunks_count_once_as_one_evidence_group() -> None:
    assert artifact_context_recall(_artifact(["chunk_a", "chunk_b"])) == 0.5
    assert artifact_context_recall(_artifact(["chunk_b", "chunk_c"])) == 1.0


def test_only_the_first_four_contexts_count() -> None:
    selected = ["x1", "x2", "x3", "x4", "chunk_a", "chunk_c"]
    assert artifact_context_recall(_artifact(selected)) == 0.0


def test_failed_generation_has_no_measurement_rather_than_zero() -> None:
    artifact = _artifact([], error="CitationValidationError: boom")
    assert artifact_context_recall(artifact) is None


def test_missing_measurements_are_counted_not_averaged() -> None:
    rows: list[dict] = [
        {
            "question_id": "Q001",
            "split": "dev",
            "generation_error": None,
            "judges": {},
            "context_recall_at_4": 1.0,
        },
        {
            "question_id": "Q002",
            "split": "dev",
            "generation_error": None,
            "judges": {},
            "context_recall_at_4": 0.5,
        },
        {
            "question_id": "Q003",
            "split": "dev",
            "generation_error": "boom",
            "judges": {},
            "context_recall_at_4": None,
        },
    ]
    recall = aggregate_dual_judges(rows)["dev"]["context_recall_at_4"]
    assert recall["mean"] == 0.75
    assert recall["effective_n"] == 2
    assert recall["eligible_n"] == 2
    assert recall["fully_covered_n"] == 1
