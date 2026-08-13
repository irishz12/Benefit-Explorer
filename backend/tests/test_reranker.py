from __future__ import annotations

import numpy as np

from src.generation.reranker import BGEReranker
from src.retrieval.models import ChunkRecord, HybridResult


class _CrossEncoder:
    def predict(self, pairs: list[tuple[str, str]], **_: object) -> np.ndarray:
        return np.linspace(0.9, 0.6, num=len(pairs))


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
