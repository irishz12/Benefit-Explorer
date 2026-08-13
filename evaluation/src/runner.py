"""End-to-end BenefitExplorer evaluation orchestration."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
BACKEND_DIR = PROJECT_ROOT / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from src.main import DEFAULT_AWS_REGION, DEFAULT_MANTLE_MODEL, build_pipeline, get_rag_components

from .context_recall import context_recall_at_4
from .dataset import GoldenQuestion, load_golden_dataset, validate_relevant_chunk_ids
from .report import aggregate_metrics, print_summary, rank_worst, write_csv, write_json, write_markdown

EVALUATION_DIR = PROJECT_ROOT / "evaluation"
DEFAULT_GOLDEN = EVALUATION_DIR / "golden" / "golden_questions.json"
DEFAULT_RESULTS = EVALUATION_DIR / "results" / "evaluation_results.json"
DEFAULT_CSV = EVALUATION_DIR / "results" / "per_question_metrics.csv"
DEFAULT_REPORT = EVALUATION_DIR / "reports" / "evaluation_summary.md"


def _format_metric(name: str, value: float | None) -> str:
    return f"{name}={value:.3f}" if value is not None else f"{name}=N/A"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate BenefitExplorer with three metrics.")
    parser.add_argument("--golden", type=Path, default=DEFAULT_GOLDEN)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--k", type=int, default=4, choices=(4,))
    parser.add_argument("--judge-model", default=None)
    parser.add_argument("--ragas-timeout", type=float, default=180.0)
    parser.add_argument("--worst-count", type=int, default=5)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from evaluation/results/evaluation_results.json.",
    )
    return parser


def _load_checkpoint(
    path: Path,
    questions: list[GoldenQuestion],
) -> dict[str, dict[str, Any]]:
    """Load reusable successful rows, retrying failed or legacy summary rows."""

    if not path.exists():
        raise FileNotFoundError(
            f"No evaluation checkpoint exists at {path}. Run without --resume first."
        )
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"Evaluation checkpoint is not valid JSON: {path}") from error
    saved_rows = payload.get("questions")
    if not isinstance(saved_rows, list):
        raise ValueError(f"Evaluation checkpoint has no question rows: {path}")

    current = {question.question_id: question for question in questions}
    checkpoint: dict[str, dict[str, Any]] = {}
    for row in saved_rows:
        if not isinstance(row, dict) or not isinstance(row.get("question_id"), str):
            raise ValueError("Evaluation checkpoint contains an invalid question row")
        question_id = row["question_id"]
        golden = current.get(question_id)
        if golden is None:
            # A checkpoint created with a larger --limit can safely be ignored
            # when resuming a smaller subset.
            continue
        if question_id in checkpoint:
            raise ValueError(f"Duplicate checkpoint row for {question_id}")
        # Condensed historical reports contain metrics only and cannot be used
        # as generation checkpoints. Treat them as unprocessed instead of
        # failing the entire resume operation.
        required_fields = {
            "question",
            "reference_answer",
            "relevant_chunk_ids",
            "generated_answer",
            "selected_contexts",
            "metrics",
        }
        if not required_fields.issubset(row):
            continue
        # Failed pipeline rows and partial RAGAS rows are retried on resume.
        metrics = row.get("metrics")
        if row.get("error") is not None or not isinstance(metrics, dict):
            continue
        if any(
            metrics.get(metric) is None
            for metric in ("faithfulness", "context_recall_at_4", "answer_correctness")
        ):
            continue

        expected = {
            "question": golden.question,
            "reference_answer": golden.reference_answer,
            "relevant_chunk_ids": list(golden.relevant_chunk_ids),
        }
        mismatched = [key for key, value in expected.items() if row.get(key) != value]
        if mismatched:
            fields = ", ".join(mismatched)
            raise ValueError(
                f"Checkpoint row {question_id} does not match the current golden "
                f"dataset ({fields}). Start a fresh run without --resume."
            )
        selected_contexts = row.get("selected_contexts")
        if not isinstance(selected_contexts, list):
            continue
        selected_ids = [
            str(context.get("chunk_id"))
            for context in selected_contexts
            if isinstance(context, dict) and context.get("chunk_id")
        ]
        normalized = dict(row)
        normalized["relevant_evidence_groups"] = [
            {
                "evidence_id": group.evidence_id,
                "description": group.description,
                "chunk_ids": list(group.chunk_ids),
            }
            for group in golden.relevant_evidence_groups
        ]
        normalized["metrics"] = {
            **metrics,
            "context_recall_at_4": context_recall_at_4(
                selected_ids,
                golden.relevant_evidence_groups,
            ),
        }
        checkpoint[question_id] = normalized
    return checkpoint


async def run_evaluation(args: argparse.Namespace) -> dict[str, Any]:
    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit must be positive")
    if args.worst_count < 1:
        raise ValueError("--worst-count must be positive")

    load_dotenv(BACKEND_DIR / ".env", override=True)
    api_key = os.getenv("AWS_BEARER_TOKEN_BEDROCK")
    if not api_key:
        raise RuntimeError("AWS_BEARER_TOKEN_BEDROCK is required in backend/.env")
    region = os.getenv("AWS_REGION", DEFAULT_AWS_REGION)
    base_url = os.getenv("OPENAI_BASE_URL", f"https://bedrock-mantle.{region}.api.aws/v1")
    judge_model = args.judge_model or os.getenv("RAGAS_JUDGE_MODEL") or os.getenv(
        "MANTLE_MODEL", DEFAULT_MANTLE_MODEL
    )

    questions = load_golden_dataset(args.golden, args.limit)
    checkpoint = _load_checkpoint(DEFAULT_RESULTS, questions) if args.resume else {}
    if checkpoint:
        print(
            f"Resuming from checkpoint: {len(checkpoint)}/{len(questions)} "
            "questions already processed.",
            flush=True,
        )
    pipeline = build_pipeline()
    retriever, _ = get_rag_components()
    chunk_lookup = {record.chunk_id: record for record in retriever.records}
    validate_relevant_chunk_ids(questions, set(chunk_lookup))
    # Keep RAGAS optional at import time so --help and golden validation remain
    # usable before evaluation dependencies are installed.
    from .ragas_metrics import RagasEvaluator

    ragas = RagasEvaluator(
        retriever.vector_store.embedder,
        judge_model,
        api_key,
        base_url,
        args.ragas_timeout,
    )

    rows: list[dict[str, Any]] = []
    for position, golden in enumerate(questions, start=1):
        saved = checkpoint.get(golden.question_id)
        if saved is not None:
            rows.append(saved)
            print(
                f"[{position:02d}/{len(questions):02d}] {golden.question_id} "
                "CHECKPOINT — skipped",
                flush=True,
            )
            continue
        print(f"[{position:02d}/{len(questions):02d}] {golden.question_id}", flush=True)
        try:
            run = pipeline.answer(golden.question)
            contexts = list(run.final_contexts[:4])
            selected_ids = [context.record.chunk_id for context in contexts]
            ragas_scores = await ragas.score(
                golden.question,
                run.response.answer,
                golden.reference_answer,
                [context.record.text for context in contexts],
            )
            metrics: dict[str, float | None] = {
                "faithfulness": ragas_scores.faithfulness,
                "context_recall_at_4": context_recall_at_4(
                    selected_ids, golden.relevant_evidence_groups
                ),
                "answer_correctness": ragas_scores.answer_correctness,
            }
            row = {
                "question_id": golden.question_id,
                "question": golden.question,
                "reference_answer": golden.reference_answer,
                "generated_answer": run.response.answer,
                "relevant_chunk_ids": list(golden.relevant_chunk_ids),
                "relevant_evidence_groups": [
                    {
                        "evidence_id": group.evidence_id,
                        "description": group.description,
                        "chunk_ids": list(group.chunk_ids),
                    }
                    for group in golden.relevant_evidence_groups
                ],
                "selected_contexts": [context.to_dict() for context in contexts],
                "metrics": metrics,
                "ragas_errors": list(ragas_scores.errors),
                "error": None,
            }
            print(
                "  " + "  ".join(_format_metric(name, value) for name, value in metrics.items()),
                flush=True,
            )
        except Exception as error:
            row = {
                "question_id": golden.question_id,
                "question": golden.question,
                "reference_answer": golden.reference_answer,
                "generated_answer": None,
                "relevant_chunk_ids": list(golden.relevant_chunk_ids),
                "relevant_evidence_groups": [
                    {
                        "evidence_id": group.evidence_id,
                        "description": group.description,
                        "chunk_ids": list(group.chunk_ids),
                    }
                    for group in golden.relevant_evidence_groups
                ],
                "selected_contexts": [],
                "metrics": {
                    "faithfulness": None,
                    "context_recall_at_4": None,
                    "answer_correctness": None,
                },
                "ragas_errors": [],
                "error": f"{type(error).__name__}: {error}",
            }
            print(f"  FAILED: {row['error']}", flush=True)
        rows.append(row)
        write_json(
            {
                "schema_version": 2,
                "status": "running",
                "completed": len(rows),
                "question_count": len(questions),
                "checkpointed_at": datetime.now(UTC).isoformat(),
                "questions": rows,
            },
            DEFAULT_RESULTS,
        )

    completed = [row for row in rows if row["error"] is None]
    judge_failures = sum(bool(row["ragas_errors"]) for row in rows)
    generated_at = datetime.now(UTC).isoformat()
    report = {
        "schema_version": 2,
        "generated_at": generated_at,
        "status": "complete",
        "configuration": {
            "golden_dataset": str(args.golden.resolve()),
            "context_k": 4,
            "judge_framework": "ragas",
            "judge_model": judge_model,
        },
        "question_count": len(questions),
        "evaluated": len(completed),
        "failed": len(rows) - len(completed),
        "judge_failures": judge_failures,
        "aggregate_metrics": aggregate_metrics(completed),
        "worst_questions": rank_worst(rows, args.worst_count),
        "questions": rows,
    }
    write_json(report, DEFAULT_RESULTS)
    write_csv(rows, DEFAULT_CSV)
    write_markdown(report, DEFAULT_REPORT)
    print_summary(report)
    return report
