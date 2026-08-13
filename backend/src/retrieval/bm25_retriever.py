"""Lexical BM25 retrieval over persisted chunk records."""

from __future__ import annotations

import re

import numpy as np
from rank_bm25 import BM25Okapi

from .models import BM25Hit, ChunkRecord

_TOKEN_PATTERN = re.compile(r"[a-z0-9]+(?:[-./][a-z0-9]+)*", re.IGNORECASE)


def tokenize(text: str) -> list[str]:
    """Tokenize insurance text while retaining ages, terms, and hyphenated words."""

    return _TOKEN_PATTERN.findall(text.casefold())


class BM25Retriever:
    """A deterministic BM25 index aligned to Chroma chunk IDs."""

    def __init__(self, records: list[ChunkRecord], k1: float = 1.5, b: float = 0.75) -> None:
        if not records:
            raise ValueError("BM25 requires at least one chunk record")
        self.records = records
        self._corpus = [self._document_tokens(record) for record in records]
        self._index = BM25Okapi(self._corpus, k1=k1, b=b)

    @staticmethod
    def _document_tokens(record: ChunkRecord) -> list[str]:
        """Apply modest field boosts without changing stored chunk text."""

        product_tokens = tokenize(record.product_name)
        section_tokens = tokenize(" ".join(record.section_types))
        return (
            product_tokens * 2
            + section_tokens * 4
            + tokenize(record.source_file)
            + tokenize(record.text)
        )

    def search(
        self,
        query: str,
        top_k: int = 25,
        allowed_chunk_ids: set[str] | None = None,
    ) -> list[BM25Hit]:
        """Return positive-score lexical matches in descending order."""

        if top_k < 1:
            raise ValueError("top_k must be positive")
        query_tokens = tokenize(query)
        if not query_tokens:
            return []
        scores = np.asarray(self._index.get_scores(query_tokens), dtype=float)
        ranked_indices = np.argsort(-scores, kind="stable")
        hits: list[BM25Hit] = []
        for index in ranked_indices:
            record = self.records[int(index)]
            if allowed_chunk_ids is not None and record.chunk_id not in allowed_chunk_ids:
                continue
            score = float(scores[index])
            if score <= 0:
                break
            hits.append(BM25Hit(record, score, len(hits) + 1))
            if len(hits) >= min(top_k, len(self.records)):
                break
        return hits
