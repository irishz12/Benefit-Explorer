import pytest

from evaluation.src.report import aggregate_dual_judges


def _outcome(value: float | None, status: str = "ok") -> dict[str, object]:
    return {"value": value, "status": status, "error": None, "attempts": 1}


def _row(question_id: str, split: str, self_score: float, independent: float) -> dict:
    return {
        "question_id": question_id,
        "split": split,
        "generation_error": None,
        "judges": {
            "self": {
                "metrics": {
                    "faithfulness": _outcome(self_score),
                    "answer_correctness": _outcome(self_score),
                }
            },
            "independent": {
                "metrics": {
                    "faithfulness": _outcome(independent),
                    "answer_correctness": _outcome(independent),
                }
            },
        },
    }


def test_dual_judge_delta_uses_only_paired_scores() -> None:
    rows = [_row("Q001", "dev", 0.9, 0.7), _row("Q002", "holdout", 0.8, 0.9)]
    aggregate = aggregate_dual_judges(rows)
    dev_delta = aggregate["dev"]["paired_deltas"]["faithfulness"]
    holdout_delta = aggregate["holdout"]["paired_deltas"]["faithfulness"]
    assert dev_delta["self_minus_independent"] == pytest.approx(0.2)
    assert dev_delta["paired_n"] == 1
    assert holdout_delta["self_minus_independent"] == pytest.approx(-0.1)


def test_provider_errors_are_excluded_and_counted() -> None:
    row = _row("Q001", "dev", 0.9, 0.7)
    row["judges"]["independent"]["metrics"]["faithfulness"] = _outcome(
        None, "provider_error"
    )
    metric = aggregate_dual_judges([row])["dev"]["judges"]["independent"]["faithfulness"]
    assert metric["mean"] is None
    assert metric["effective_n"] == 0
    assert metric["eligible_n"] == 1
    assert metric["provider_errors"] == 1
