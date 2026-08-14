import json
from pathlib import Path

import pytest

from evaluation.src.artifacts import load_generation_checkpoint, question_fingerprint
from evaluation.src.dataset import EvidenceGroup, GoldenQuestion
from evaluation.src.runner import assert_independent_judge


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


def _write(path: Path, row: dict[str, object], config_hash: str = "config-1") -> None:
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "configuration": {"config_hash": config_hash},
                "questions": [row],
            }
        ),
        encoding="utf-8",
    )


def _complete_row() -> dict[str, object]:
    question = _question()
    return {
        "question_id": "Q001",
        "question_fingerprint": question_fingerprint(question, "dev"),
        "generated_answer": "Generated.",
        "selected_contexts": [{"chunk_id": "chunk_a", "text": "Evidence."}],
        "retrieval_trace": [{"chunk_id": "chunk_a", "rrf_score": 0.1}],
        "generation_model_id": "qwen.qwen3-32b",
        "config_hash": "config-1",
        "generation_error": None,
    }


def test_complete_generation_artifact_is_reused(tmp_path: Path) -> None:
    path = tmp_path / "artifacts.json"
    _write(path, _complete_row())
    checkpoint = load_generation_checkpoint(
        path,
        [_question()],
        {"Q001": "dev"},
        "config-1",
    )
    assert checkpoint["Q001"]["generated_answer"] == "Generated."
    assert checkpoint["Q001"]["retrieval_trace"][0]["chunk_id"] == "chunk_a"


def test_failed_or_incomplete_generation_artifact_is_retried(tmp_path: Path) -> None:
    path = tmp_path / "artifacts.json"
    row = _complete_row()
    row["generation_error"] = "RateLimitError: retry later"
    _write(path, row)
    assert load_generation_checkpoint(
        path,
        [_question()],
        {"Q001": "dev"},
        "config-1",
    ) == {}


def test_configuration_mismatch_fails_loudly(tmp_path: Path) -> None:
    path = tmp_path / "artifacts.json"
    _write(path, _complete_row())
    with pytest.raises(ValueError, match="configuration differs"):
        load_generation_checkpoint(
            path,
            [_question()],
            {"Q001": "dev"},
            "different-config",
        )


def test_independent_judge_cannot_equal_generation_model() -> None:
    with pytest.raises(ValueError, match="must differ"):
        assert_independent_judge("qwen.qwen3-32b", "QWEN.QWEN3-32B")
