"""Context Recall@4 for the final contexts selected by the RAG pipeline."""

from __future__ import annotations

from typing import Sequence

from .dataset import EvidenceGroup


def context_recall_at_4(
    selected_chunk_ids: Sequence[str],
    evidence_groups: Sequence[EvidenceGroup],
) -> float:
    """Return the fraction of evidence groups covered by the first four contexts.

    Chunks inside one group are alternatives, normally caused by ingestion
    overlap. Retrieving any member satisfies that material evidence item.
    """

    selected = set(selected_chunk_ids[:4])
    if not evidence_groups:
        return 0.0
    covered = sum(bool(selected.intersection(group.chunk_ids)) for group in evidence_groups)
    return covered / len(evidence_groups)
