"""Typed records shared by dense, sparse, and hybrid retrieval."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Any, Mapping


@dataclass(frozen=True, slots=True)
class ChunkRecord:
    """One indexed insurance brochure chunk with full provenance."""

    chunk_id: str
    product_name: str
    page_number: int
    page_numbers: tuple[int, ...]
    section_type: str
    section_types: tuple[str, ...]
    source_file: str
    text: str

    @property
    def search_text(self) -> str:
        """Enrich text with filterable labels for dense and lexical indexing."""

        sections = ", ".join(self.section_types)
        return (
            f"Product: {self.product_name}\n"
            f"Section: {sections}\n"
            f"Source: {self.source_file}\n\n"
            f"{self.text}"
        )

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ChunkRecord":
        """Validate and construct a record from processed-chunk JSON."""

        chunk_id = str(payload.get("chunk_id", "")).strip()
        text = str(payload.get("text", "")).strip()
        if not chunk_id or not text:
            raise ValueError("Every chunk requires a non-empty chunk_id and text")
        page_number = int(payload["page_number"])
        raw_pages = payload.get("page_numbers", [page_number])
        raw_sections = payload.get("section_types", [payload.get("section_type", "General")])
        return cls(
            chunk_id=chunk_id,
            product_name=str(payload.get("product_name", "Unknown Product")),
            page_number=page_number,
            page_numbers=tuple(int(page) for page in raw_pages),
            section_type=str(payload.get("section_type", "General")),
            section_types=tuple(str(section) for section in raw_sections),
            source_file=str(payload.get("source_file", "")),
            text=text,
        )

    @classmethod
    def from_chroma(
        cls,
        chunk_id: str,
        document: str | None,
        metadata: Mapping[str, Any] | None,
    ) -> "ChunkRecord":
        """Decode Chroma's scalar metadata representation."""

        metadata = metadata or {}
        text = str(metadata.get("text") or document or "")
        page_number = int(metadata.get("page_number", 1))
        page_numbers = _decode_json_list(metadata.get("page_numbers"), [page_number])
        section_type = str(metadata.get("section_type", "General"))
        section_types = _decode_json_list(metadata.get("section_types"), [section_type])
        return cls(
            chunk_id=chunk_id,
            product_name=str(metadata.get("product_name", "Unknown Product")),
            page_number=page_number,
            page_numbers=tuple(int(page) for page in page_numbers),
            section_type=section_type,
            section_types=tuple(str(section) for section in section_types),
            source_file=str(metadata.get("source_file", "")),
            text=text,
        )

    def to_chroma_metadata(self) -> dict[str, str | int | float | bool]:
        """Encode lists as JSON because Chroma metadata values are scalar."""

        return {
            "chunk_id": self.chunk_id,
            "product_name": self.product_name,
            "page_number": self.page_number,
            "page_numbers": json.dumps(self.page_numbers, separators=(",", ":")),
            "section_type": self.section_type,
            "section_types": json.dumps(self.section_types, separators=(",", ":")),
            "source_file": self.source_file,
            "text": self.text,
        }

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["page_numbers"] = list(self.page_numbers)
        payload["section_types"] = list(self.section_types)
        return payload


def _decode_json_list(value: Any, default: list[Any]) -> list[Any]:
    if value is None:
        return default
    if isinstance(value, (list, tuple)):
        return list(value)
    try:
        decoded = json.loads(str(value))
    except (json.JSONDecodeError, TypeError):
        return default
    return decoded if isinstance(decoded, list) else default


@dataclass(frozen=True, slots=True)
class DenseHit:
    record: ChunkRecord
    score: float
    distance: float
    rank: int


@dataclass(frozen=True, slots=True)
class BM25Hit:
    record: ChunkRecord
    score: float
    rank: int


@dataclass(frozen=True, slots=True)
class HybridResult:
    """A fused result with component scores and ranks for inspection."""

    record: ChunkRecord
    rrf_score: float
    dense_score: float | None
    bm25_score: float | None
    dense_rank: int | None
    bm25_rank: int | None
    rerank_score: float | None = None
    product_match: bool = False
    focused_window: str | None = None
    focused_window_score: float | None = None
    focused_rerank_score: float | None = None
    section_adjustment: float = 0.0
    exact_match_score: float = 0.0
    intent_anchor_score: float = 0.0
    detected_intent: str | None = None
    rerank_query: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "rrf_score": self.rrf_score,
            "dense_score": self.dense_score,
            "bm25_score": self.bm25_score,
            "dense_rank": self.dense_rank,
            "bm25_rank": self.bm25_rank,
            "rerank_score": self.rerank_score,
            "product_match": self.product_match,
            "focused_window": self.focused_window,
            "focused_window_score": self.focused_window_score,
            "focused_rerank_score": self.focused_rerank_score,
            "section_adjustment": self.section_adjustment,
            "exact_match_score": self.exact_match_score,
            "intent_anchor_score": self.intent_anchor_score,
            "detected_intent": self.detected_intent,
            "rerank_query": self.rerank_query,
            **self.record.to_dict(),
        }
