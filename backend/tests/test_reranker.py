from __future__ import annotations

import numpy as np

from src.generation.reranker import BGEReranker, RerankingSignalConfig
from src.retrieval.models import ChunkRecord, HybridResult


class _CrossEncoder:
    def predict(self, pairs: list[tuple[str, str]], **_: object) -> np.ndarray:
        return np.linspace(0.9, 0.6, num=len(pairs))


class _RaisingEmbedder:
    """Fails the test if runtime reranking ever triggers a BGE embedding call."""

    def embed_query(self, text: str) -> list[float]:
        raise AssertionError("embed_query must not run when semantic windows are disabled")

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        raise AssertionError("embed_documents must not run when semantic windows are disabled")


class _SpyEmbedder:
    """Records calls so the opt-in semantic path can be proven to still work."""

    def __init__(self) -> None:
        self.embed_query_calls = 0
        self.embed_documents_calls = 0

    def embed_query(self, text: str) -> list[float]:
        self.embed_query_calls += 1
        return [1.0, 0.0]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.embed_documents_calls += 1
        return [[1.0, 0.0] for _ in texts]


def _candidate(chunk_id: str, product: str, text: str) -> HybridResult:
    record = ChunkRecord(
        chunk_id=chunk_id,
        product_name=product,
        page_number=1,
        page_numbers=(1,),
        section_type="Benefits",
        section_types=("Benefits",),
        source_file=f"{product}.pdf",
        text=text,
    )
    return HybridResult(record, 0.02, 0.8, 2.0, 1, 1, product_match=True)


def test_comparison_reranking_reserves_one_result_per_product() -> None:
    reranker = BGEReranker(model=_CrossEncoder())
    candidates = [
        _candidate("edge-1", "Kotak EDGE", "Sum Assured on Maturity is paid."),
        _candidate("edge-2", "Kotak EDGE", "Guaranteed Income may accrue."),
        _candidate("tulip-1", "Kotak TULIP", "Fund Value is payable at maturity."),
    ]

    ranked = reranker.rerank(
        "Compare Kotak EDGE and Kotak TULIP maturity benefits",
        candidates,
        top_k=2,
    )

    assert len(ranked) == 2
    assert {item.record.product_name for item in ranked} == {
        "Kotak EDGE",
        "Kotak TULIP",
    }
    assert all(item.rerank_score is not None for item in ranked)


def test_reranker_defaults_to_a_larger_batch_size() -> None:
    """batch_size=2 serialized the cross-encoder pass; 8 is a safer CPU/MPS default."""

    reranker = BGEReranker(model=_CrossEncoder())
    assert reranker.batch_size == 8


def test_focused_window_selection_is_lexical_only_by_default() -> None:
    """Runtime reranking must not embed focused windows or dedup text by default."""

    reranker = BGEReranker(model=_CrossEncoder(), focus_embedder=_RaisingEmbedder())
    candidates = [
        _candidate(
            "edge-1",
            "Kotak EDGE",
            "Sum Assured on Maturity is paid at the end of the policy term. "
            "Fund value inclusive of loyalty additions shall be payable on maturity.",
        ),
        _candidate(
            "edge-2",
            "Kotak EDGE",
            "Guaranteed Income may accrue after the deferment period ends and is "
            "credited to the policyholder every policy year.",
        ),
    ]

    # Would raise AssertionError from _RaisingEmbedder if any embedding call
    # (focused-window scoring or near-duplicate removal) ran during reranking.
    ranked = reranker.rerank(
        "What is the maturity benefit for Kotak EDGE?",
        candidates,
        top_k=2,
    )

    assert len(ranked) >= 1
    assert all(item.rerank_score is not None for item in ranked)


def test_focused_window_selection_can_opt_into_semantic_scoring() -> None:
    """The semantic capability stays available behind an explicit config flag."""

    spy = _SpyEmbedder()
    reranker = BGEReranker(
        model=_CrossEncoder(),
        focus_embedder=spy,
        signal_config=RerankingSignalConfig(use_semantic_focused_windows=True),
    )
    candidates = [
        _candidate(
            "edge-1",
            "Kotak EDGE",
            "Sum Assured on Maturity is paid at the end of the policy term.",
        ),
        _candidate(
            "edge-2",
            "Kotak EDGE",
            "Guaranteed Income may accrue after the deferment period ends.",
        ),
    ]

    reranker._score_candidates("What is the maturity benefit?", candidates)

    assert spy.embed_query_calls >= 1
    assert spy.embed_documents_calls >= 1


def test_remove_near_duplicates_drops_lexically_similar_chunk_without_embedder() -> None:
    """Near-duplicate protection must work with no focus_embedder configured."""

    reranker = BGEReranker(model=_CrossEncoder())
    original = _candidate(
        "chunk-a",
        "Kotak EDGE",
        "The maturity benefit shall be payable at the end of the policy term "
        "provided all due premiums have been paid and the policy is in force.",
    )
    near_duplicate = _candidate(
        "chunk-b",
        "Kotak EDGE",
        "The maturity benefit shall be payable at the end of the policy term "
        "provided all due premiums have been paid, and the policy remains in force.",
    )
    distinct = _candidate(
        "chunk-c",
        "Kotak EDGE",
        "Suicide within 12 months of the risk commencement date restricts the death benefit.",
    )

    kept = reranker._remove_near_duplicates([original, near_duplicate, distinct])

    kept_ids = {result.record.chunk_id for result in kept}
    assert kept_ids == {"chunk-a", "chunk-c"}
