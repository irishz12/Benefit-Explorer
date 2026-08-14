"""Dense + BM25 retrieval fused with Reciprocal Rank Fusion."""

from __future__ import annotations

from dataclasses import dataclass
from .bm25_retriever import BM25Retriever
from .models import ChunkRecord, HybridResult
from .product_detection import ProductDetector
from .vector_store import ChromaVectorStore


@dataclass(slots=True)
class _FusionState:
    record: ChunkRecord
    rrf_score: float = 0.0
    dense_score: float | None = None
    bm25_score: float | None = None
    dense_rank: int | None = None
    bm25_rank: int | None = None


@dataclass(frozen=True, slots=True)
class ProductAwareRetrieval:
    """Results plus product-routing diagnostics for the QA report."""

    results: tuple[HybridResult, ...]
    detected_products: tuple[str, ...]
    mode: str


class HybridRetriever:
    """Retrieve dense and lexical candidates, then fuse them using RRF."""

    def __init__(
        self,
        vector_store: ChromaVectorStore,
        bm25_retriever: BM25Retriever | None = None,
        rrf_k: int = 60,
        dense_weight: float = 1.0,
        sparse_weight: float = 1.0,
        product_aware: bool = True,
        min_filtered_candidates: int = 2,
        product_score_boost: float = 5.0,
    ) -> None:
        if rrf_k < 1:
            raise ValueError("rrf_k must be positive")
        self.vector_store = vector_store
        self.records = vector_store.load_records()
        if not self.records:
            raise ValueError("The Chroma collection is empty. Run the index command first.")
        if min_filtered_candidates < 1:
            raise ValueError("min_filtered_candidates must be positive")
        if product_score_boost < 1.0:
            raise ValueError("product_score_boost must be at least 1.0")
        self.bm25 = bm25_retriever or BM25Retriever(self.records)
        self.rrf_k = rrf_k
        self.dense_weight = dense_weight
        self.sparse_weight = sparse_weight
        self.product_aware = product_aware
        self.min_filtered_candidates = min_filtered_candidates
        self.product_score_boost = product_score_boost
        self.product_detector = ProductDetector(record.product_name for record in self.records)

    def detect_products(self, query: str) -> tuple[str, ...]:
        """Detect canonical product names represented in the collection."""

        return self.product_detector.detect(query) if self.product_aware else ()

    def _fuse(
        self,
        query: str,
        dense_k: int,
        sparse_k: int,
        product_names: tuple[str, ...] = (),
        apply_product_boost: bool = False,
        filter_by_product: bool = True,
    ) -> list[HybridResult]:
        allowed_ids = (
            {record.chunk_id for record in self.records if record.product_name in product_names}
            if product_names and filter_by_product
            else None
        )
        dense_hits = self.vector_store.dense_search(
            query,
            top_k=dense_k,
            product_names=product_names if filter_by_product else (),
        )
        sparse_hits = self.bm25.search(
            query,
            top_k=sparse_k,
            allowed_chunk_ids=allowed_ids,
        )
        fused: dict[str, _FusionState] = {}

        for hit in dense_hits:
            fused[hit.record.chunk_id] = _FusionState(
                record=hit.record,
                rrf_score=self.dense_weight / (self.rrf_k + hit.rank),
                dense_score=hit.score,
                dense_rank=hit.rank,
            )
        for hit in sparse_hits:
            item = fused.setdefault(
                hit.record.chunk_id,
                _FusionState(record=hit.record),
            )
            item.rrf_score += self.sparse_weight / (self.rrf_k + hit.rank)
            item.bm25_score = hit.score
            item.bm25_rank = hit.rank

        candidates: list[HybridResult] = []
        for item in fused.values():
            product_match = bool(product_names and item.record.product_name in product_names)
            score = item.rrf_score
            if apply_product_boost and product_match:
                score *= self.product_score_boost
            candidates.append(
                HybridResult(
                    record=item.record,
                    rrf_score=score,
                    dense_score=item.dense_score,
                    bm25_score=item.bm25_score,
                    dense_rank=item.dense_rank,
                    bm25_rank=item.bm25_rank,
                    product_match=product_match,
                )
            )
        candidates.sort(
            key=lambda result: (
                -result.rrf_score,
                result.dense_rank or 10**9,
                result.bm25_rank or 10**9,
                result.record.chunk_id,
            )
        )
        return candidates

    def _hard_filtered_candidates(
        self,
        query: str,
        products: tuple[str, ...],
        dense_k: int,
        sparse_k: int,
    ) -> list[HybridResult]:
        if len(products) == 1:
            return self._fuse(query, dense_k, sparse_k, product_names=products)

        # Retrieve comparisons per product so each named product is represented.
        merged: dict[str, HybridResult] = {}
        for product in products:
            for candidate in self._fuse(
                query,
                dense_k,
                sparse_k,
                product_names=(product,),
            ):
                merged[candidate.record.chunk_id] = candidate
        candidates = list(merged.values())
        candidates.sort(
            key=lambda result: (
                -result.rrf_score,
                products.index(result.record.product_name),
                result.record.chunk_id,
            )
        )
        return candidates

    def retrieve_with_diagnostics(
        self,
        query: str,
        final_k: int = 20,
        dense_k: int = 25,
        sparse_k: int = 25,
        preferred_products: tuple[str, ...] | None = None,
    ) -> ProductAwareRetrieval:
        """Retrieve with hard product filtering and automatic boost fallback."""

        query = query.strip()
        if not query:
            raise ValueError("query cannot be empty")
        if min(final_k, dense_k, sparse_k) < 1:
            raise ValueError("retrieval limits must be positive")

        products = (
            self.detect_products(query) if preferred_products is None else tuple(preferred_products)
        )
        mode = "none"
        if products:
            candidates = self._hard_filtered_candidates(
                query,
                products,
                dense_k,
                sparse_k,
            )
            all_products_represented = all(
                any(candidate.record.product_name == product for candidate in candidates)
                for product in products
            )
            # A detected single product is a strong constraint: retain hard
            # filtering whenever any matching evidence exists. Comparisons keep
            # hard filtering only when every requested product is represented.
            keep_hard_filter = (
                bool(candidates)
                if len(products) == 1
                else (all_products_represented and len(candidates) >= self.min_filtered_candidates)
            )
            if keep_hard_filter:
                mode = "hard_filter"
            else:
                mode = "soft_boost"
                candidates = self._fuse(
                    query,
                    dense_k,
                    sparse_k,
                    product_names=products,
                    apply_product_boost=True,
                    filter_by_product=False,
                )
        else:
            candidates = self._fuse(query, dense_k, sparse_k)

        candidates = candidates[:final_k]
        return ProductAwareRetrieval(tuple(candidates), products, mode)

    def retrieve(
        self,
        query: str,
        final_k: int = 20,
        dense_k: int = 25,
        sparse_k: int = 25,
        preferred_products: tuple[str, ...] | None = None,
    ) -> list[HybridResult]:
        """Return candidates while preserving the original list-only API."""

        return list(
            self.retrieve_with_diagnostics(
                query,
                final_k=final_k,
                dense_k=dense_k,
                sparse_k=sparse_k,
                preferred_products=preferred_products,
            ).results
        )
