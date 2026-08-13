"""JSON, CSV, Markdown, and terminal reporting."""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from statistics import fmean
from typing import Any, Iterable

METRICS = (
    "faithfulness",
    "context_recall_at_4",
    "answer_correctness",
)


def aggregate_metrics(rows: Iterable[dict[str, Any]]) -> dict[str, float | None]:
    rows = list(rows)
    aggregate: dict[str, float | None] = {}
    for metric in METRICS:
        values = [row["metrics"].get(metric) for row in rows]
        numeric = [float(value) for value in values if value is not None]
        aggregate[metric] = fmean(numeric) if numeric else None
    return aggregate


def rank_worst(rows: Iterable[dict[str, Any]], count: int) -> list[dict[str, Any]]:
    ranked: list[dict[str, Any]] = []
    for row in rows:
        scores = [row["metrics"].get(metric) for metric in METRICS]
        numeric = [float(score) for score in scores if score is not None]
        ranked.append(
            {
                "question_id": row["question_id"],
                "question": row["question"],
                "composite_score": fmean(numeric) if numeric else 0.0,
                "metrics": row["metrics"],
                "error": row.get("error"),
            }
        )
    return sorted(ranked, key=lambda item: item["composite_score"])[:count]


def write_json(payload: dict[str, Any], path: Path) -> None:
    """Write JSON atomically so an interrupted run cannot corrupt its checkpoint."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["question_id", "question", *METRICS, "error"]
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "question_id": row["question_id"],
                    "question": row["question"],
                    **row["metrics"],
                    "error": row.get("error"),
                }
            )


def write_markdown(report: dict[str, Any], path: Path) -> None:
    metrics = report["aggregate_metrics"]
    lines = [
        "# BenefitExplorer Evaluation Report",
        "",
        f"- Questions evaluated: {report['evaluated']}/{report['question_count']}",
        f"- Failed questions: {report['failed']}",
        f"- Questions with RAGAS judge errors: {report['judge_failures']}",
        f"- Generated: {report['generated_at']}",
        "",
        "## Aggregate metrics",
        "",
        "| Metric | Score |",
        "|---|---:|",
    ]
    for metric in METRICS:
        value = metrics.get(metric)
        lines.append(f"| {metric} | {value:.3f} |" if value is not None else f"| {metric} | N/A |")
    lines.extend(["", "## Worst-performing questions", ""])
    for item in report["worst_questions"]:
        lines.append(
            f"- **{item['question_id']}** ({item['composite_score']:.3f}): "
            f"{item['question']}"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def print_summary(report: dict[str, Any]) -> None:
    print("\nBenefitExplorer evaluation summary")
    for metric in METRICS:
        value = report["aggregate_metrics"].get(metric)
        print(f"{metric:28}: {value:.3f}" if value is not None else f"{metric:28}: N/A")
    print("\nWorst-performing questions")
    for item in report["worst_questions"]:
        print(f"  {item['question_id']}  {item['composite_score']:.3f}  {item['question']}")
