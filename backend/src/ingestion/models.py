"""Internal data models used by the ingestion pipeline."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class ExtractedPage:
    """Text extracted from one PDF page."""

    page_number: int
    text: str


@dataclass(frozen=True, slots=True)
class ExtractedDocument:
    """A source PDF and its page-level contents."""

    source_path: Path
    product_name: str
    pages: tuple[ExtractedPage, ...]


@dataclass(frozen=True, slots=True)
class SectionSpan:
    """A page-local section of a document."""

    page_number: int
    section_type: str
    text: str


@dataclass(frozen=True, slots=True)
class Chunk:
    """The serialized unit consumed by later retrieval stages."""

    chunk_id: str
    product_name: str
    page_number: int
    page_numbers: tuple[int, ...]
    section_type: str
    section_types: tuple[str, ...]
    source_file: str
    text: str

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""

        payload = asdict(self)
        payload["page_numbers"] = list(self.page_numbers)
        payload["section_types"] = list(self.section_types)
        return payload
