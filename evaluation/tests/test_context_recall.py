from evaluation.src.context_recall import context_recall_at_4
from evaluation.src.dataset import EvidenceGroup


def test_overlapping_chunks_count_as_one_evidence_item() -> None:
    groups = (
        EvidenceGroup("E1", "same claim", ("chunk_a", "chunk_b")),
        EvidenceGroup("E2", "second claim", ("chunk_c",)),
    )
    assert context_recall_at_4(("chunk_b", "noise"), groups) == 0.5
    assert context_recall_at_4(("chunk_a", "chunk_b", "chunk_c"), groups) == 1.0


def test_context_recall_uses_only_first_four_contexts() -> None:
    groups = (EvidenceGroup("E1", "answer", ("chunk_answer",)),)
    selected = ("n1", "n2", "n3", "n4", "chunk_answer")
    assert context_recall_at_4(selected, groups) == 0.0
