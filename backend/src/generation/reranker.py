"""Optional BGE cross-encoder reranking for hybrid candidates."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from src.retrieval.embeddings import Embedder
from src.retrieval.models import HybridResult
from src.retrieval.product_detection import ProductDetector

from .ranking_signals import (
    detect_query_intent,
    exact_match_score,
    intent_anchor_score,
    section_adjustment,
    select_focused_windows,
)

LOGGER = logging.getLogger(__name__)

DEFAULT_RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"
_MULTI_PART_QUESTION = re.compile(
    r"(?:,\s*)?\b(?:and|or)\s+(?:what|how|when|which|where|does|is|are|can)\b",
    re.IGNORECASE,
)
_OPTION_COUNT_QUESTION = re.compile(r"\bhow\s+many\b", re.IGNORECASE)


@dataclass(frozen=True, slots=True)
class RerankingSignalConfig:
    """Weights and thresholds blended with cross-encoder relevance."""

    section_match_boost: float = 0.14
    section_mismatch_penalty: float = 0.08
    focused_window_weight: float = 0.45
    focused_selection_boost: float = 0.02
    exact_match_boost: float = 0.18
    intent_anchor_boost: float = 0.18
    lexical_candidates_per_chunk: int = 3
    window_semantic_weight: float = 0.65
    near_duplicate_threshold: float = 0.88
    score_cliff_gap: float = 0.22
    score_cliff_unsupported_single_gap: float = 0.20
    score_cliff_single_result_gap: float = 0.40
    score_cliff_min_results: int = 1

    def __post_init__(self) -> None:
        if (
            min(
                self.section_match_boost,
                self.section_mismatch_penalty,
                self.focused_selection_boost,
                self.exact_match_boost,
                self.intent_anchor_boost,
            )
            < 0
        ):
            raise ValueError("Reranking boost and penalty weights cannot be negative")
        if not 0.0 <= self.focused_window_weight <= 1.0:
            raise ValueError("focused_window_weight must be between 0 and 1")
        if not 0.0 <= self.window_semantic_weight <= 1.0:
            raise ValueError("window_semantic_weight must be between 0 and 1")
        if self.lexical_candidates_per_chunk < 1:
            raise ValueError("lexical_candidates_per_chunk must be positive")
        if not 0.0 <= self.near_duplicate_threshold <= 1.0:
            raise ValueError("near_duplicate_threshold must be between 0 and 1")
        if self.score_cliff_gap < 0:
            raise ValueError("score_cliff_gap cannot be negative")
        if not (0.0 <= self.score_cliff_unsupported_single_gap <= self.score_cliff_gap):
            raise ValueError(
                "score_cliff_unsupported_single_gap must be between zero and " "score_cliff_gap"
            )
        if self.score_cliff_single_result_gap < self.score_cliff_gap:
            raise ValueError(
                "score_cliff_single_result_gap cannot be smaller than " "score_cliff_gap"
            )
        if self.score_cliff_min_results < 1:
            raise ValueError("score_cliff_min_results must be positive")


class BGEReranker:
    """Score query-passage pairs and retain the most relevant chunks."""

    def __init__(
        self,
        model_name: str = DEFAULT_RERANKER_MODEL,
        device: str | None = None,
        batch_size: int = 2,
        cache_dir: Path | None = None,
        offline: bool = False,
        show_progress: bool = False,
        product_match_boost: float = 0.35,
        focus_embedder: Embedder | None = None,
        signal_config: RerankingSignalConfig | None = None,
        model: Any | None = None,
    ) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        if product_match_boost < 0:
            raise ValueError("product_match_boost cannot be negative")
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self.cache_dir = cache_dir
        self.offline = offline
        self.show_progress = show_progress
        self.product_match_boost = product_match_boost
        self.focus_embedder = focus_embedder
        self.signal_config = signal_config or RerankingSignalConfig()
        self._model = model

    def _load(self) -> Any:
        if self._model is None:
            try:
                from sentence_transformers import CrossEncoder
            except ImportError as exc:
                raise RuntimeError(
                    "sentence-transformers is required. Install backend/requirements.txt."
                ) from exc
            self._model = CrossEncoder(
                self.model_name,
                device=self.device,
                cache_folder=str(self.cache_dir) if self.cache_dir else None,
                trust_remote_code=False,
                local_files_only=self.offline,
            )
        return self._model

    @staticmethod
    def _product_scoped_queries(
        query: str,
        products: tuple[str, ...],
    ) -> dict[str, str]:
        """Keep the common comparison intent plus each product's local qualifier."""

        spans: dict[str, tuple[int, int]] = {}
        for product in products:
            pattern = (
                r"(?<![a-z0-9])"
                + r"[^a-z0-9]+".join(
                    re.escape(token) for token in re.findall(r"[a-z0-9]+", product.casefold())
                )
                + r"(?![a-z0-9])"
            )
            match = re.search(pattern, query.casefold())
            if match is not None:
                spans[product] = match.span()
        if len(spans) != len(products):
            return {product: f"{query} Focus product: {product}." for product in products}

        first_start = min(start for start, _ in spans.values())
        common_intent = query[:first_start].strip(" ,:-")
        conjunction = re.compile(r"\b(?:and|versus|vs\.?)\b", re.IGNORECASE)
        scoped: dict[str, str] = {}
        for product in products:
            start, end = spans[product]
            boundary = conjunction.search(query, end)
            local_end = boundary.start() if boundary else len(query)
            local = query[start:local_end].strip(" ,:-")
            scoped[product] = f"{common_intent} {local}".strip()
        return scoped

    def _score_candidates(
        self,
        query: str,
        candidates: list[HybridResult],
        rerank_query: str | None = None,
    ) -> list[HybridResult]:
        """Score a homogeneous candidate list against one query."""

        if not candidates:
            return []
        scoring_query = (rerank_query or query).strip()
        config = self.signal_config
        intent = detect_query_intent(scoring_query)
        focused_windows = select_focused_windows(
            scoring_query,
            [candidate.record.text for candidate in candidates],
            intent,
            self.focus_embedder,
            lexical_candidates_per_chunk=config.lexical_candidates_per_chunk,
            semantic_weight=config.window_semantic_weight,
        )
        full_pairs = [(scoring_query, candidate.record.search_text) for candidate in candidates]
        focused_pairs = (
            [
                (
                    scoring_query,
                    f"Product: {candidate.record.product_name}\n"
                    f"Section: {candidate.record.section_type}\n\n{window.text}",
                )
                for candidate, window in zip(candidates, focused_windows, strict=True)
            ]
            if focused_windows
            else []
        )
        pairs = [*full_pairs, *focused_pairs]
        raw_scores = self._load().predict(
            pairs,
            batch_size=self.batch_size,
            show_progress_bar=self.show_progress,
            convert_to_numpy=True,
        )
        scores = np.asarray(raw_scores, dtype=float).reshape(-1)
        if len(scores) != len(pairs):
            raise RuntimeError("Reranker returned an unexpected number of scores")
        full_scores = scores[: len(candidates)]
        focused_scores = scores[len(candidates) :]
        rescored: list[HybridResult] = []
        for index, (candidate, full_score) in enumerate(zip(candidates, full_scores, strict=True)):
            window = focused_windows[index] if focused_windows else None
            focused_score = float(focused_scores[index]) if focused_windows else None
            base_score = float(full_score)
            if focused_score is not None:
                base_score = (
                    1.0 - config.focused_window_weight
                ) * base_score + config.focused_window_weight * focused_score
            section_score = section_adjustment(
                candidate.record.section_type,
                intent,
                config.section_match_boost,
                config.section_mismatch_penalty,
                candidate.record.text,
            )
            phrase_score = exact_match_score(
                scoring_query,
                candidate.record.text,
                intent,
            )
            anchor_score = intent_anchor_score(candidate.record.text, intent)
            final_score = (
                base_score
                + section_score
                + config.exact_match_boost * phrase_score
                + config.intent_anchor_boost * anchor_score
                + (config.focused_selection_boost * window.score if window else 0.0)
                + (self.product_match_boost if candidate.product_match else 0.0)
            )
            rescored.append(
                replace(
                    candidate,
                    rerank_score=final_score,
                    focused_window=window.text if window else None,
                    focused_window_score=window.score if window else None,
                    focused_rerank_score=focused_score,
                    section_adjustment=section_score,
                    exact_match_score=phrase_score,
                    intent_anchor_score=anchor_score,
                    detected_intent=intent.name,
                    rerank_query=scoring_query,
                )
            )
        rescored.sort(
            key=lambda candidate: (
                -(candidate.rerank_score or 0.0),
                -candidate.rrf_score,
                candidate.record.chunk_id,
            )
        )
        LOGGER.debug(
            "Rerank query=%r intent=%s candidates=%d",
            scoring_query,
            intent.name,
            len(candidates),
        )
        for result in rescored:
            LOGGER.debug(
                "chunk=%s product=%s section=%s section_adjustment=%.3f "
                "intent_anchor=%.3f final_score=%.6f",
                result.record.chunk_id,
                result.record.product_name,
                result.record.section_type,
                result.section_adjustment,
                result.intent_anchor_score,
                result.rerank_score or 0.0,
            )
        return rescored

    def _remove_near_duplicates(
        self,
        ranked: list[HybridResult],
    ) -> list[HybridResult]:
        """Drop lower-ranked chunks with cosine similarity above the threshold."""

        config = self.signal_config
        if self.focus_embedder is None or len(ranked) < 2:
            return ranked
        vectors = np.asarray(
            self.focus_embedder.embed_documents([result.record.text for result in ranked]),
            dtype=float,
        )
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        vectors = vectors / np.where(norms == 0.0, 1.0, norms)
        kept: list[HybridResult] = []
        kept_vectors: list[np.ndarray] = []
        for result, vector in zip(ranked, vectors, strict=True):
            duplicate_of = next(
                (
                    kept_result
                    for kept_result, kept_vector in zip(
                        kept,
                        kept_vectors,
                        strict=True,
                    )
                    if float(np.dot(vector, kept_vector)) > config.near_duplicate_threshold
                ),
                None,
            )
            if duplicate_of is not None:
                LOGGER.debug(
                    "Removed near-duplicate chunk=%s duplicate_of=%s threshold=%.2f",
                    result.record.chunk_id,
                    duplicate_of.record.chunk_id,
                    config.near_duplicate_threshold,
                )
                continue
            kept.append(result)
            kept_vectors.append(vector)
        return kept

    def _apply_score_cliff(
        self,
        ranked: list[HybridResult],
        required_products: tuple[str, ...] = (),
        minimum_results: int | None = None,
    ) -> list[HybridResult]:
        """Stop when a candidate falls significantly below the best result.

        A single result is allowed only when it is extremely dominant. Once a
        supporting result has been retained, the normal top-relative cutoff
        removes the lower-scoring tail. Comparison queries remain protected by
        ``required_products`` until every detected product is represented.
        """

        config = self.signal_config
        required_count = max(
            config.score_cliff_min_results,
            minimum_results or config.score_cliff_min_results,
        )
        if len(ranked) < 2:
            return ranked
        selected = [ranked[0]]
        top_score = ranked[0].rerank_score or 0.0
        for result in ranked[1:]:
            current_score = result.rerank_score or 0.0
            represented_products = {item.record.product_name for item in selected}
            relative_gap = top_score - current_score
            if len(selected) == 1:
                cutoff = (
                    config.score_cliff_single_result_gap
                    if result.intent_anchor_score > 0.0
                    else config.score_cliff_unsupported_single_gap
                )
            else:
                cutoff = config.score_cliff_gap
            if (
                len(selected) >= required_count
                and set(required_products).issubset(represented_products)
                and relative_gap >= cutoff
            ):
                LOGGER.debug(
                    "Top-relative score cliff after chunk=%s gap=%.3f threshold=%.3f",
                    selected[-1].record.chunk_id,
                    relative_gap,
                    cutoff,
                )
                break
            selected.append(result)
        return selected

    @staticmethod
    def _minimum_results(query: str, product_count: int) -> int:
        """Keep two contexts for questions that explicitly require multiple facts."""

        if (
            product_count >= 2
            or _OPTION_COUNT_QUESTION.search(query)
            or _MULTI_PART_QUESTION.search(query)
        ):
            return 2
        return 1

    @staticmethod
    def _protect_hybrid_results(
        selected: list[HybridResult],
        scored: list[HybridResult],
        protected_count: int,
        top_k: int,
        required_products: tuple[str, ...] = (),
    ) -> list[HybridResult]:
        """Retain the strongest hybrid evidence when intent anchors provide no signal."""

        if protected_count < 1 or not scored:
            return selected[:top_k]
        hybrid_ranked = sorted(
            scored,
            key=lambda result: (
                -result.rrf_score,
                result.dense_rank or 10**9,
                result.bm25_rank or 10**9,
                result.record.chunk_id,
            ),
        )
        protected: list[HybridResult] = []
        if required_products:
            for product in required_products:
                match = next(
                    (result for result in hybrid_ranked if result.record.product_name == product),
                    None,
                )
                if match is not None:
                    protected.append(match)
                if len(protected) >= protected_count:
                    break
        for result in hybrid_ranked:
            if len(protected) >= protected_count:
                break
            if result.record.chunk_id not in {item.record.chunk_id for item in protected}:
                protected.append(result)
        protected_ids = {result.record.chunk_id for result in protected}
        retained = list(protected)
        retained_ids = set(protected_ids)
        for result in selected:
            if len(retained) >= top_k:
                break
            if result.record.chunk_id not in retained_ids:
                retained.append(result)
                retained_ids.add(result.record.chunk_id)
        retained.sort(
            key=lambda result: (
                -(result.rerank_score or 0.0),
                -result.rrf_score,
                result.record.chunk_id,
            )
        )
        LOGGER.debug(
            "Low-confidence safeguard protected hybrid chunks=%s",
            tuple(result.record.chunk_id for result in protected),
        )
        return retained[:top_k]

    @staticmethod
    def _ensure_minimum_results(
        selected: list[HybridResult],
        scored: list[HybridResult],
        minimum_results: int,
        top_k: int,
    ) -> list[HybridResult]:
        """Backfill the best scored evidence if filtering leaves too few contexts."""

        retained = list(selected)
        retained_ids = {result.record.chunk_id for result in retained}
        for result in scored:
            if len(retained) >= min(minimum_results, top_k):
                break
            if result.record.chunk_id not in retained_ids:
                retained.append(result)
                retained_ids.add(result.record.chunk_id)
        retained.sort(
            key=lambda result: (
                -(result.rerank_score or 0.0),
                -result.rrf_score,
                result.record.chunk_id,
            )
        )
        return retained[:top_k]

    @staticmethod
    def _merge_with_product_quota(
        ranked_by_product: dict[str, list[HybridResult]],
        products: tuple[str, ...],
        top_k: int,
        fallback: list[HybridResult] | None = None,
    ) -> list[HybridResult]:
        """Reserve an even minimum representation, then fill by relevance."""

        quota = max(1, top_k // len(products))
        selected: dict[str, HybridResult] = {}
        for product in products:
            for result in ranked_by_product.get(product, [])[:quota]:
                selected[result.record.chunk_id] = result

        remaining = [
            result
            for product in products
            for result in ranked_by_product.get(product, [])[quota:]
            if result.record.chunk_id not in selected
        ]
        remaining.sort(
            key=lambda result: (
                -(result.rerank_score or 0.0),
                -result.rrf_score,
                result.record.chunk_id,
            )
        )
        for result in remaining:
            if len(selected) >= top_k:
                break
            selected[result.record.chunk_id] = result
        for result in fallback or []:
            if len(selected) >= top_k:
                break
            selected.setdefault(result.record.chunk_id, result)
        merged = list(selected.values())
        merged.sort(
            key=lambda result: (
                -(result.rerank_score or 0.0),
                -result.rrf_score,
                (
                    products.index(result.record.product_name)
                    if result.record.product_name in products
                    else len(products)
                ),
                result.record.chunk_id,
            )
        )
        return merged[:top_k]

    def rerank(
        self,
        query: str,
        candidates: list[HybridResult],
        top_k: int = 6,
    ) -> list[HybridResult]:
        """Return candidates ordered by relevance with comparison balancing."""

        query = query.strip()
        if not query:
            raise ValueError("query cannot be empty")
        if top_k < 1:
            raise ValueError("top_k must be positive")
        if not candidates:
            return []

        candidate_products = tuple(
            dict.fromkeys(candidate.record.product_name for candidate in candidates)
        )
        detected_products = ProductDetector(candidate_products).detect(query)
        minimum_results = self._minimum_results(query, len(detected_products))
        if len(detected_products) >= 2:
            scoped_queries = self._product_scoped_queries(query, detected_products)
            LOGGER.debug(
                "Comparison rerank products=%s scoped_queries=%s",
                detected_products,
                scoped_queries,
            )
            scored_by_product = {
                product: self._score_candidates(
                    query,
                    [
                        candidate
                        for candidate in candidates
                        if candidate.record.product_name == product
                    ],
                    rerank_query=scoped_queries[product],
                )
                for product in detected_products
            }
            ranked_by_product = {
                product: self._remove_near_duplicates(scored)
                for product, scored in scored_by_product.items()
            }
            fallback_candidates = [
                candidate
                for candidate in candidates
                if candidate.record.product_name not in detected_products
            ]
            fallback = (
                self._remove_near_duplicates(self._score_candidates(query, fallback_candidates))
                if fallback_candidates
                else []
            )
            merged = self._merge_with_product_quota(
                ranked_by_product,
                detected_products,
                top_k,
                fallback=fallback,
            )
            selected = self._apply_score_cliff(
                merged,
                required_products=detected_products,
                minimum_results=minimum_results,
            )
            scored = [
                result for product in detected_products for result in scored_by_product[product]
            ]
            if scored and all(result.intent_anchor_score == 0.0 for result in scored):
                selected = self._protect_hybrid_results(
                    selected,
                    scored,
                    protected_count=min(2, minimum_results),
                    top_k=top_k,
                    required_products=detected_products,
                )
            selected = self._ensure_minimum_results(
                selected,
                scored,
                minimum_results,
                top_k,
            )
            return selected[:top_k]

        LOGGER.debug(
            "Single-query rerank products=%s",
            detected_products,
        )
        scored = self._score_candidates(query, candidates)
        low_confidence = all(result.intent_anchor_score == 0.0 for result in scored)
        ranked = self._remove_near_duplicates(scored)
        ranked = self._apply_score_cliff(
            ranked,
            minimum_results=minimum_results,
        )
        if low_confidence:
            ranked = self._protect_hybrid_results(
                ranked,
                scored,
                protected_count=min(2, minimum_results),
                top_k=top_k,
            )
        ranked = self._ensure_minimum_results(
            ranked,
            scored,
            minimum_results,
            top_k,
        )
        return ranked[:top_k]
