"""Auditable generation artifacts and deterministic evaluation configuration."""

from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from .dataset import GoldenQuestion


ARTIFACT_SCHEMA_VERSION = 1


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_hash(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def git_revision(project_root: Path) -> tuple[str, bool]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=project_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    dirty = bool(
        subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=project_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    )
    return commit, dirty


def build_generation_configuration(
    pipeline: Any,
    project_root: Path,
    golden_path: Path,
    evidence_groups_path: Path,
    splits_path: Path,
    generation_base_url: str,
) -> dict[str, Any]:
    """Capture every stable setting that affects retrieval or generation."""

    commit, dirty = git_revision(project_root)
    reranker = pipeline.reranker
    retriever = pipeline.retriever
    signal_config = getattr(reranker, "signal_config", None)
    hashed_configuration = {
        "generation_model_id": pipeline.generator.model,
        "generation_provider": "aws-bedrock-mantle",
        "generation_base_url": generation_base_url,
        "embedding_model_id": retriever.vector_store.embedder.model_name,
        "reranker_model_id": getattr(reranker, "model_name", None),
        "retrieval_k": pipeline.retrieval_k,
        "final_context_k": pipeline.final_context_k,
        "retrieval": {
            "rrf_k": retriever.rrf_k,
            "dense_weight": retriever.dense_weight,
            "sparse_weight": retriever.sparse_weight,
            "product_aware": retriever.product_aware,
            "min_filtered_candidates": retriever.min_filtered_candidates,
            "product_score_boost": retriever.product_score_boost,
        },
        "reranking": {
            "product_match_boost": getattr(reranker, "product_match_boost", None),
            "signals": asdict(signal_config) if is_dataclass(signal_config) else None,
        },
        "golden_sha256": file_sha256(golden_path),
        "evidence_groups_sha256": file_sha256(evidence_groups_path),
        "splits_sha256": file_sha256(splits_path),
        "git_commit": commit,
        "git_dirty": dirty,
    }
    return {
        **hashed_configuration,
        "config_hash": canonical_hash(hashed_configuration),
    }


def question_fingerprint(question: GoldenQuestion, split: str) -> str:
    return canonical_hash(
        {
            "question_id": question.question_id,
            "question": question.question,
            "reference_answer": question.reference_answer,
            "relevant_chunk_ids": list(question.relevant_chunk_ids),
            "split": split,
        }
    )


def load_generation_checkpoint(
    path: Path,
    questions: Sequence[GoldenQuestion],
    split_by_id: Mapping[str, str],
    config_hash: str,
) -> dict[str, dict[str, Any]]:
    """Load only complete, matching artifact rows; failed rows are retried."""

    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != ARTIFACT_SCHEMA_VERSION:
        raise ValueError(f"Unsupported generation artifact schema in {path}")
    if payload.get("configuration", {}).get("config_hash") != config_hash:
        raise ValueError(
            "Generation checkpoint configuration differs from the current pipeline. "
            "Use a new artifact path rather than mixing runs."
        )
    current = {question.question_id: question for question in questions}
    reusable: dict[str, dict[str, Any]] = {}
    for row in payload.get("questions", []):
        if not isinstance(row, dict):
            continue
        question_id = row.get("question_id")
        question = current.get(question_id)
        if question is None or row.get("generation_error") is not None:
            continue
        expected_fingerprint = question_fingerprint(question, split_by_id[question_id])
        required = {
            "generated_answer",
            "selected_contexts",
            "retrieval_trace",
            "generation_model_id",
            "config_hash",
        }
        if not required.issubset(row):
            continue
        if row.get("question_fingerprint") != expected_fingerprint:
            raise ValueError(f"Stored artifact for {question_id} no longer matches the golden row")
        reusable[question_id] = row
    return reusable
