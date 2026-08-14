"""Atomic output and dual-judge evaluation summaries."""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from statistics import fmean
from typing import Any, Iterable

JUDGE_METRICS = ("faithfulness", "answer_correctness")
JUDGE_KEYS = ("self", "independent")


def write_json(payload: dict[str, Any], path: Path) -> None:
    """Write JSON atomically so interruption cannot corrupt a checkpoint."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _metric_outcome(row: dict[str, Any], judge: str, metric: str) -> dict[str, Any]:
    return row.get("judges", {}).get(judge, {}).get("metrics", {}).get(metric, {})


def _judge_aggregate(
    rows: list[dict[str, Any]],
    judge: str,
    metric: str,
) -> dict[str, Any]:
    outcomes = [_metric_outcome(row, judge, metric) for row in rows]
    values = [
        float(outcome["value"])
        for outcome in outcomes
        if outcome.get("status") == "ok" and outcome.get("value") is not None
    ]
    return {
        "mean": fmean(values) if values else None,
        "effective_n": len(values),
        "eligible_n": len(rows),
        "provider_errors": sum(outcome.get("status") == "provider_error" for outcome in outcomes),
        "metric_errors": sum(outcome.get("status") == "metric_error" for outcome in outcomes),
    }


def _paired_delta(
    rows: list[dict[str, Any]],
    metric: str,
) -> dict[str, Any]:
    paired: list[tuple[float, float]] = []
    for row in rows:
        self_outcome = _metric_outcome(row, "self", metric)
        independent_outcome = _metric_outcome(row, "independent", metric)
        if self_outcome.get("status") == independent_outcome.get("status") == "ok":
            paired.append((float(self_outcome["value"]), float(independent_outcome["value"])))
    return {
        "self_mean": fmean(pair[0] for pair in paired) if paired else None,
        "independent_mean": fmean(pair[1] for pair in paired) if paired else None,
        "self_minus_independent": fmean(pair[0] - pair[1] for pair in paired) if paired else None,
        "paired_n": len(paired),
        "eligible_n": len(rows),
    }


def aggregate_dual_judges(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    all_rows = list(rows)
    aggregates: dict[str, Any] = {}
    for split in ("dev", "holdout", "all"):
        split_rows = all_rows if split == "all" else [row for row in all_rows if row["split"] == split]
        successful = [row for row in split_rows if row.get("generation_error") is None]
        aggregates[split] = {
            "question_count": len(split_rows),
            "generation_success_n": len(successful),
            "judges": {
                judge: {
                    metric: _judge_aggregate(successful, judge, metric)
                    for metric in JUDGE_METRICS
                }
                for judge in JUDGE_KEYS
            },
            "paired_deltas": {
                metric: _paired_delta(successful, metric) for metric in JUDGE_METRICS
            },
        }
    return aggregates


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "question_id",
        "split",
        "generation_model_id",
        "self_judge_model_id",
        "independent_judge_model_id",
        "self_faithfulness",
        "independent_faithfulness",
        "self_answer_correctness",
        "independent_answer_correctness",
        "generation_error",
    ]
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "question_id": row["question_id"],
                    "split": row["split"],
                    "generation_model_id": row["generation_model_id"],
                    "self_judge_model_id": row["judge_model_ids"]["self"],
                    "independent_judge_model_id": row["judge_model_ids"]["independent"],
                    "self_faithfulness": _metric_outcome(row, "self", "faithfulness").get("value"),
                    "independent_faithfulness": _metric_outcome(row, "independent", "faithfulness").get("value"),
                    "self_answer_correctness": _metric_outcome(row, "self", "answer_correctness").get("value"),
                    "independent_answer_correctness": _metric_outcome(row, "independent", "answer_correctness").get("value"),
                    "generation_error": row.get("generation_error"),
                }
            )


def _score_cell(metric: dict[str, Any]) -> str:
    value = metric.get("mean")
    score = f"{value:.3f}" if value is not None else "N/A"
    return f"{score} ({metric['effective_n']}/{metric['eligible_n']})"


def write_markdown(report: dict[str, Any], path: Path) -> None:
    configuration = report["configuration"]
    lines = [
        "# BenefitExplorer Dual-Judge Evaluation",
        "",
        f"- Generation model: `{configuration['generation_model_id']}`",
        f"- Self-judge: `{configuration['judge_model_ids']['self']}`",
        f"- Independent judge: `{configuration['judge_model_ids']['independent']}`",
        f"- Config hash: `{configuration['config_hash']}`",
        "- Values in parentheses are `effective n / eligible n`.",
        "",
    ]
    for split in ("dev", "holdout"):
        aggregate = report["aggregate_metrics"][split]
        lines.extend(
            [
                f"## {split.title()} (n={aggregate['question_count']})",
                "",
                "| Metric | Self-judge | Independent judge | Self − independent (paired n) |",
                "|---|---:|---:|---:|",
            ]
        )
        for metric in JUDGE_METRICS:
            delta = aggregate["paired_deltas"][metric]
            delta_value = delta["self_minus_independent"]
            delta_text = f"{delta_value:+.3f}" if delta_value is not None else "N/A"
            lines.append(
                f"| {metric} | {_score_cell(aggregate['judges']['self'][metric])} | "
                f"{_score_cell(aggregate['judges']['independent'][metric])} | "
                f"{delta_text} ({delta['paired_n']}/{delta['eligible_n']}) |"
            )
        lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def print_summary(report: dict[str, Any]) -> None:
    for split in ("dev", "holdout"):
        aggregate = report["aggregate_metrics"][split]
        print(f"\n{split.title()} (n={aggregate['question_count']})")
        for metric in JUDGE_METRICS:
            self_score = aggregate["judges"]["self"][metric]
            independent = aggregate["judges"]["independent"][metric]
            delta = aggregate["paired_deltas"][metric]
            delta_value = delta["self_minus_independent"]
            delta_text = f"{delta_value:+.3f}" if delta_value is not None else "N/A"
            print(
                f"{metric:20} self={_score_cell(self_score)}  "
                f"independent={_score_cell(independent)}  "
                f"delta={delta_text} (paired {delta['paired_n']}/{delta['eligible_n']})"
            )
