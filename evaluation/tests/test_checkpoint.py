import json
from pathlib import Path

from evaluation.src.dataset import EvidenceGroup, GoldenQuestion
from evaluation.src.runner import _load_checkpoint


def _question() -> GoldenQuestion:
    return GoldenQuestion(
        question_id="Q001",
        question="Question?",
        reference_answer="Reference.",
        relevant_chunk_ids=("chunk_a", "chunk_b"),
        relevant_evidence_groups=(
            EvidenceGroup("E1", "answer", ("chunk_a", "chunk_b")),
        ),
    )


def _write(path: Path, row: dict[str, object]) -> None:
    path.write_text(json.dumps({"questions": [row]}), encoding="utf-8")


def test_legacy_condensed_result_is_ignored(tmp_path: Path) -> None:
    path = tmp_path / "results.json"
    _write(path, {"question_id": "Q001", "metrics": {}})
    assert _load_checkpoint(path, [_question()]) == {}


def test_failed_row_is_retried(tmp_path: Path) -> None:
    path = tmp_path / "results.json"
    _write(
        path,
        {
            "question_id": "Q001",
            "question": "Question?",
            "reference_answer": "Reference.",
            "relevant_chunk_ids": ["chunk_a", "chunk_b"],
            "generated_answer": None,
            "selected_contexts": [],
            "metrics": {},
            "error": "temporary provider failure",
        },
    )
    assert _load_checkpoint(path, [_question()]) == {}


def test_successful_row_is_reused_and_recall_is_recomputed(tmp_path: Path) -> None:
    path = tmp_path / "results.json"
    _write(
        path,
        {
            "question_id": "Q001",
            "question": "Question?",
            "reference_answer": "Reference.",
            "relevant_chunk_ids": ["chunk_a", "chunk_b"],
            "generated_answer": "Generated.",
            "selected_contexts": [{"chunk_id": "chunk_b"}],
            "metrics": {
                "faithfulness": 0.9,
                "context_recall_at_4": 0.5,
                "answer_correctness": 0.8,
            },
            "ragas_errors": [],
            "error": None,
        },
    )
    checkpoint = _load_checkpoint(path, [_question()])
    assert checkpoint["Q001"]["metrics"]["context_recall_at_4"] == 1.0
