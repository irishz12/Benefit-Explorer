"""Auditable generation, dual-judge scoring, and reporting orchestration."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
BACKEND_DIR = PROJECT_ROOT / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from src.main import DEFAULT_AWS_REGION, build_pipeline, get_rag_components

from .artifacts import (
    ARTIFACT_SCHEMA_VERSION,
    build_generation_configuration,
    load_generation_checkpoint,
    question_fingerprint,
)
from .context_recall import context_recall_at_4
from .dataset import (
    EvidenceGroup,
    GoldenQuestion,
    load_evaluation_splits,
    load_golden_dataset,
    validate_relevant_chunk_ids,
)
from .report import aggregate_dual_judges, print_summary, write_csv, write_json, write_markdown

EVALUATION_DIR = PROJECT_ROOT / "evaluation"
DEFAULT_GOLDEN = EVALUATION_DIR / "golden" / "golden_questions.json"
DEFAULT_SPLITS = EVALUATION_DIR / "golden" / "splits.json"
DEFAULT_ARTIFACTS = EVALUATION_DIR / "results" / "generation_artifacts.json"
DEFAULT_RESULTS = EVALUATION_DIR / "results" / "evaluation_results.json"
DEFAULT_CSV = EVALUATION_DIR / "results" / "per_question_metrics.csv"
DEFAULT_REPORT = EVALUATION_DIR / "reports" / "evaluation_summary.md"
# The independent judge must come from a different model family than the
# generator, so a Qwen model is never a valid choice here.
DEFAULT_INDEPENDENT_JUDGE = "openai.gpt-oss-120b"
DEFAULT_JUDGE_MAX_OUTPUT_TOKENS = 16384
RESULT_SCHEMA_VERSION = 4
# Statuses that a `--resume --retry-provider-errors` pass is allowed to re-score.
RETRYABLE_STATUSES = frozenset({"provider_error", "parse_error"})


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate once, then evaluate stored BenefitExplorer answers with two judges."
    )
    parser.add_argument(
        "stage",
        nargs="?",
        choices=("generate", "score", "report", "all"),
        default="all",
    )
    parser.add_argument("--golden", type=Path, default=DEFAULT_GOLDEN)
    parser.add_argument("--splits", type=Path, default=DEFAULT_SPLITS)
    parser.add_argument("--artifacts", type=Path, default=DEFAULT_ARTIFACTS)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--retry-provider-errors", action="store_true")
    parser.add_argument(
        "--continue-after-provider-error",
        action="store_true",
        help="Continue to later questions after exhausted provider retries.",
    )
    parser.add_argument("--independent-judge-model", default=None)
    parser.add_argument("--ragas-timeout", type=float, default=180.0)
    parser.add_argument("--judge-max-attempts", type=int, default=4)
    parser.add_argument("--judge-backoff-seconds", type=float, default=3.0)
    parser.add_argument("--judge-max-output-tokens", type=int, default=None)
    parser.add_argument(
        "--judge-repair-attempts",
        type=int,
        default=3,
        help="In-call structured-output repair attempts before a parse failure is recorded.",
    )
    return parser


def _load_inputs(args: argparse.Namespace) -> tuple[list[GoldenQuestion], dict[str, str]]:
    questions = load_golden_dataset(args.golden)
    splits = load_evaluation_splits(args.splits, questions)
    split_by_id = {
        **{question_id: "dev" for question_id in splits.dev_question_ids},
        **{question_id: "holdout" for question_id in splits.holdout_question_ids},
    }
    return questions, split_by_id


def _generation_payload(
    configuration: dict[str, Any],
    rows: list[dict[str, Any]],
    question_count: int,
    status: str,
) -> dict[str, Any]:
    return {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "status": status,
        "checkpointed_at": datetime.now(UTC).isoformat(),
        "configuration": configuration,
        "question_count": question_count,
        "completed": len(rows),
        "questions": rows,
    }


def _citation_payload(citation: Any) -> dict[str, Any]:
    return {
        "index": citation.index,
        "chunk_id": citation.chunk_id,
        "product": citation.product,
        "page": citation.page,
        "supporting_text": citation.supporting_text,
    }


async def generate_artifacts(args: argparse.Namespace) -> dict[str, Any]:
    """Run the RAG pipeline once and atomically retain all auditable artifacts."""

    load_dotenv(BACKEND_DIR / ".env", override=True)
    questions, split_by_id = _load_inputs(args)
    region = os.getenv("AWS_REGION", DEFAULT_AWS_REGION)
    generation_base_url = os.getenv(
        "OPENAI_BASE_URL",
        f"https://bedrock-mantle.{region}.api.aws/v1",
    )
    pipeline = build_pipeline()
    retriever, _ = get_rag_components()
    validate_relevant_chunk_ids(questions, {record.chunk_id for record in retriever.records})
    configuration = build_generation_configuration(
        pipeline,
        PROJECT_ROOT,
        args.golden,
        args.golden.with_name("evidence_groups.json"),
        args.splits,
        generation_base_url,
    )
    checkpoint = (
        load_generation_checkpoint(
            args.artifacts,
            questions,
            split_by_id,
            configuration["config_hash"],
        )
        if args.resume
        else {}
    )
    rows: list[dict[str, Any]] = []
    for position, golden in enumerate(questions, start=1):
        saved = checkpoint.get(golden.question_id)
        if saved is not None:
            rows.append(saved)
            print(f"[{position:02d}/30] {golden.question_id} GENERATION CHECKPOINT", flush=True)
            continue

        split = split_by_id[golden.question_id]
        print(f"[{position:02d}/30] {golden.question_id} generating ({split})", flush=True)
        started = time.monotonic()
        base_row: dict[str, Any] = {
            "question_id": golden.question_id,
            "split": split,
            "question": golden.question,
            "reference_answer": golden.reference_answer,
            "relevant_chunk_ids": list(golden.relevant_chunk_ids),
            "relevant_evidence_groups": [
                {
                    "evidence_id": group.evidence_id,
                    "description": group.description,
                    "chunk_ids": list(group.chunk_ids),
                }
                for group in golden.relevant_evidence_groups
            ],
            "question_fingerprint": question_fingerprint(golden, split),
            "generation_model_id": configuration["generation_model_id"],
            "config_hash": configuration["config_hash"],
        }
        try:
            run = await asyncio.to_thread(pipeline.answer, golden.question)
            row = {
                **base_row,
                "generated_answer": run.response.answer,
                "citations": [_citation_payload(citation) for citation in run.response.citations],
                "selected_contexts": [context.to_dict() for context in run.final_contexts[:4]],
                "retrieval_trace": [candidate.to_dict() for candidate in run.retrieval_trace],
                "detected_products": list(run.detected_products),
                "product_retrieval_mode": run.product_retrieval_mode,
                "generation_seconds": round(time.monotonic() - started, 3),
                "generated_at": datetime.now(UTC).isoformat(),
                "generation_error": None,
            }
        except Exception as error:
            row = {
                **base_row,
                "generated_answer": None,
                "citations": [],
                "selected_contexts": [],
                "retrieval_trace": [],
                "detected_products": [],
                "product_retrieval_mode": "error",
                "generation_seconds": round(time.monotonic() - started, 3),
                "generated_at": datetime.now(UTC).isoformat(),
                "generation_error": f"{type(error).__name__}: {error}",
            }
            print(f"  generation failed: {row['generation_error']}", flush=True)
        rows.append(row)
        write_json(
            _generation_payload(configuration, rows, len(questions), "running"), args.artifacts
        )

    payload = _generation_payload(configuration, rows, len(questions), "complete")
    payload["generation_success_n"] = sum(row["generation_error"] is None for row in rows)
    write_json(payload, args.artifacts)
    return payload


def _load_artifacts(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise FileNotFoundError(
            f"Generation artifacts not found at {path}. Run the generate stage first."
        ) from error
    if payload.get("schema_version") != ARTIFACT_SCHEMA_VERSION:
        raise ValueError(f"Unsupported generation artifact schema in {path}")
    rows = payload.get("questions")
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"Generation artifact contains no question rows: {path}")
    return payload


def artifact_context_recall(artifact: dict[str, Any]) -> float | None:
    """Evidence-group Context Recall@4 for one stored generation artifact.

    Returns None when generation failed, because the error path stores no
    selected contexts and a missing measurement must not be read as 0.0.
    """

    if artifact.get("generation_error") is not None:
        return None
    groups = [
        EvidenceGroup(
            str(group.get("evidence_id", "")),
            str(group.get("description", "")),
            tuple(group.get("chunk_ids", ())),
        )
        for group in artifact.get("relevant_evidence_groups", [])
    ]
    if not groups:
        return None
    selected = [context["chunk_id"] for context in artifact.get("selected_contexts", [])]
    return context_recall_at_4(selected, groups)


def _model_family(model_id: str) -> str:
    """Return the vendor prefix of a model id (`qwen.qwen3-32b` -> `qwen`)."""

    normalised = model_id.strip().casefold()
    for separator in (".", "/"):
        if separator in normalised:
            return normalised.split(separator, 1)[0]
    return normalised


def assert_independent_judge(generation_model_id: str, judge_model_id: str) -> None:
    if generation_model_id.strip().casefold() == judge_model_id.strip().casefold():
        raise ValueError(
            "Independent judge model must differ from the generation model: "
            f"both resolve to {generation_model_id!r}."
        )
    if _model_family(generation_model_id) == _model_family(judge_model_id):
        raise ValueError(
            "Independent judge model must come from a different model family than the "
            f"generator: {generation_model_id!r} and {judge_model_id!r} share the "
            f"{_model_family(generation_model_id)!r} family."
        )


def _load_score_checkpoint(
    path: Path,
    config_hash: str,
    judge_model_ids: dict[str, str],
) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != RESULT_SCHEMA_VERSION:
        return {}
    configuration = payload.get("configuration", {})
    if configuration.get("config_hash") != config_hash:
        raise ValueError("Score checkpoint was produced from a different generation artifact set")
    if configuration.get("judge_model_ids") != judge_model_ids:
        raise ValueError("Score checkpoint uses different judge model IDs")
    return {
        row["question_id"]: row
        for row in payload.get("questions", [])
        if isinstance(row, dict) and isinstance(row.get("question_id"), str)
    }


def _judge_needs_score(
    row: dict[str, Any],
    judge_key: str,
    retry_provider_errors: bool,
) -> bool:
    judge = row.get("judges", {}).get(judge_key)
    if not isinstance(judge, dict):
        return True
    metrics = judge.get("metrics", {})
    if not {"faithfulness", "answer_correctness"}.issubset(metrics):
        return True
    outcomes = metrics.values()
    statuses = {outcome.get("status") for outcome in outcomes if isinstance(outcome, dict)}
    if not statuses:
        return True
    return retry_provider_errors and bool(statuses & RETRYABLE_STATUSES)


def _has_provider_error(row: dict[str, Any], judge_key: str) -> bool:
    outcomes = row.get("judges", {}).get(judge_key, {}).get("metrics", {}).values()
    return any(outcome.get("status") == "provider_error" for outcome in outcomes)


def _score_payload(
    configuration: dict[str, Any],
    rows: list[dict[str, Any]],
    status: str,
) -> dict[str, Any]:
    payload = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "status": status,
        "checkpointed_at": datetime.now(UTC).isoformat(),
        "configuration": configuration,
        "question_count": len(rows),
        "questions": rows,
    }
    payload["aggregate_metrics"] = aggregate_dual_judges(rows)
    return payload


async def score_artifacts(args: argparse.Namespace) -> dict[str, Any]:
    """Score one stored answer set; this function never calls answer generation."""

    load_dotenv(BACKEND_DIR / ".env", override=True)
    artifacts = _load_artifacts(args.artifacts)
    artifact_configuration = artifacts["configuration"]
    generation_model_id = artifact_configuration["generation_model_id"]
    self_model_id = generation_model_id
    independent_model_id = (
        args.independent_judge_model
        or os.getenv("INDEPENDENT_JUDGE_MODEL")
        or DEFAULT_INDEPENDENT_JUDGE
    )
    assert_independent_judge(generation_model_id, independent_model_id)
    judge_model_ids = {"self": self_model_id, "independent": independent_model_id}

    aws_key = os.getenv("AWS_BEARER_TOKEN_BEDROCK")
    if not aws_key:
        raise RuntimeError("AWS_BEARER_TOKEN_BEDROCK is required for both judges")
    region = os.getenv("AWS_REGION", DEFAULT_AWS_REGION)
    judge_base_url = os.getenv(
        "OPENAI_BASE_URL",
        f"https://bedrock-mantle.{region}.api.aws/v1",
    )
    max_output_tokens = args.judge_max_output_tokens or int(
        os.getenv("RAGAS_MAX_OUTPUT_TOKENS", str(DEFAULT_JUDGE_MAX_OUTPUT_TOKENS))
    )

    retriever, _ = get_rag_components()
    from .ragas_metrics import RagasEvaluator

    # Both judges run on Bedrock Mantle with the same credentials as generation;
    # only the model id differs.
    evaluators = {
        judge_key: RagasEvaluator(
            retriever.vector_store.embedder,
            judge_model_ids[judge_key],
            aws_key,
            judge_base_url,
            args.ragas_timeout,
            args.judge_max_attempts,
            args.judge_backoff_seconds,
            max_output_tokens,
            args.judge_repair_attempts,
        )
        for judge_key in ("self", "independent")
    }
    configuration = {
        "artifact_path": str(args.artifacts.resolve()),
        "config_hash": artifact_configuration["config_hash"],
        "generation_model_id": generation_model_id,
        "judge_model_ids": judge_model_ids,
        "judge_provider": "aws-bedrock-mantle",
        "judge_base_url": judge_base_url,
        "judge_framework": "ragas",
        "judge_concurrency": 1,
        "judge_max_attempts": args.judge_max_attempts,
        "judge_backoff_seconds": args.judge_backoff_seconds,
        "judge_max_output_tokens": max_output_tokens,
        "judge_repair_attempts": args.judge_repair_attempts,
    }
    checkpoint = (
        _load_score_checkpoint(args.results, configuration["config_hash"], judge_model_ids)
        if args.resume
        else {}
    )
    rows: list[dict[str, Any]] = []
    stop_for_provider = False
    for position, artifact in enumerate(artifacts["questions"], start=1):
        saved = checkpoint.get(artifact["question_id"], {})
        row = {
            "question_id": artifact["question_id"],
            "split": artifact["split"],
            "question": artifact["question"],
            "reference_answer": artifact["reference_answer"],
            "generated_answer": artifact.get("generated_answer"),
            "selected_contexts": artifact.get("selected_contexts", []),
            "generation_error": artifact.get("generation_error"),
            "context_recall_at_4": artifact_context_recall(artifact),
            "generation_model_id": generation_model_id,
            "judge_model_ids": judge_model_ids,
            "config_hash": configuration["config_hash"],
            "judges": dict(saved.get("judges", {})),
        }
        rows.append(row)
        if row["generation_error"] is not None:
            write_json(_score_payload(configuration, rows, "running"), args.results)
            continue

        contexts = [context["text"] for context in row["selected_contexts"]]
        for judge_key in ("self", "independent"):
            if not _judge_needs_score(row, judge_key, args.retry_provider_errors):
                continue
            print(
                f"[{position:02d}/30] {row['question_id']} {judge_key} judge "
                f"({judge_model_ids[judge_key]})",
                flush=True,
            )
            scores = await evaluators[judge_key].score(
                row["question"],
                row["generated_answer"],
                row["reference_answer"],
                contexts,
            )
            row["judges"][judge_key] = {
                "model_id": judge_model_ids[judge_key],
                "metrics": scores.to_dict(),
                "scored_at": datetime.now(UTC).isoformat(),
            }
            write_json(_score_payload(configuration, rows, "running"), args.results)
            if _has_provider_error(row, judge_key) and not args.continue_after_provider_error:
                print(
                    "Provider retries exhausted; checkpoint saved. Resume later with "
                    "--resume --retry-provider-errors.",
                    flush=True,
                )
                stop_for_provider = True
                break
        if stop_for_provider:
            break

    status = "provider_interrupted" if stop_for_provider else "complete"
    report = _score_payload(configuration, rows, status)
    write_json(report, args.results)
    write_csv(rows, args.csv)
    write_markdown(report, args.report)
    print_summary(report)
    return report


def render_report(args: argparse.Namespace) -> dict[str, Any]:
    payload = json.loads(args.results.read_text(encoding="utf-8"))
    rows = payload["questions"]
    payload["aggregate_metrics"] = aggregate_dual_judges(rows)
    write_json(payload, args.results)
    write_csv(rows, args.csv)
    write_markdown(payload, args.report)
    print_summary(payload)
    return payload


async def run_evaluation(args: argparse.Namespace) -> dict[str, Any]:
    if args.judge_max_attempts < 1:
        raise ValueError("--judge-max-attempts must be positive")
    if args.judge_backoff_seconds < 0:
        raise ValueError("--judge-backoff-seconds cannot be negative")
    if args.judge_max_output_tokens is not None and args.judge_max_output_tokens < 1:
        raise ValueError("--judge-max-output-tokens must be positive")
    if args.judge_repair_attempts < 1:
        raise ValueError("--judge-repair-attempts must be positive")
    if args.stage == "generate":
        return await generate_artifacts(args)
    if args.stage == "score":
        return await score_artifacts(args)
    if args.stage == "report":
        return render_report(args)
    await generate_artifacts(args)
    return await score_artifacts(args)
