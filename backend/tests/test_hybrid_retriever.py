from __future__ import annotations

from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.models import ChunkRecord, DenseHit


def _record(chunk_id: str, product: str, text: str) -> ChunkRecord:
    return ChunkRecord(
        chunk_id=chunk_id,
        product_name=product,
        page_number=1,
        page_numbers=(1,),
        section_type="Benefits",
        section_types=("Benefits",),
        source_file=f"{product}.pdf",
        text=text,
    )


class _VectorStore:
    def __init__(self, records: list[ChunkRecord]) -> None:
        self.records = records

    def load_records(self) -> list[ChunkRecord]:
        return self.records

    def dense_search(
        self,
        query: str,
        top_k: int,
        product_names: tuple[str, ...] = (),
    ) -> list[DenseHit]:
        del query
        records = [
            record
            for record in self.records
            if not product_names or record.product_name in product_names
        ][:top_k]
        return [
            DenseHit(record, score=1.0 / rank, distance=0.1 * rank, rank=rank)
            for rank, record in enumerate(records, start=1)
        ]


def test_single_product_query_uses_hard_filter() -> None:
    records = [
        _record("edge", "Kotak EDGE", "EDGE maturity benefit is paid."),
        _record("tulip", "Kotak TULIP", "TULIP maturity fund value is paid."),
    ]
    retriever = HybridRetriever(_VectorStore(records))  # type: ignore[arg-type]

    result = retriever.retrieve_with_diagnostics(
        "What maturity benefit does Kotak EDGE pay?",
        final_k=5,
    )

    assert result.detected_products == ("Kotak EDGE",)
    assert result.mode == "hard_filter"
    assert {item.record.product_name for item in result.results} == {"Kotak EDGE"}


def test_comparison_query_keeps_both_products() -> None:
    records = [
        _record("edge", "Kotak EDGE", "EDGE maturity benefit is paid."),
        _record("tulip", "Kotak TULIP", "TULIP maturity fund value is paid."),
        _record("gain", "Kotak GAIN", "GAIN income benefit is paid."),
    ]
    retriever = HybridRetriever(_VectorStore(records))  # type: ignore[arg-type]

    result = retriever.retrieve_with_diagnostics(
        "Compare Kotak EDGE and Kotak TULIP maturity benefits",
        final_k=5,
    )

    assert result.mode == "hard_filter"
    assert result.detected_products == ("Kotak EDGE", "Kotak TULIP")
    assert {item.record.product_name for item in result.results} == {
        "Kotak EDGE",
        "Kotak TULIP",
    }
