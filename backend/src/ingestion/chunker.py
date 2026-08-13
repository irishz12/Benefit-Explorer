"""Clause-aware recursive text chunking."""

from __future__ import annotations

import re
from dataclasses import dataclass

from .models import SectionSpan
from .tokenizer import TokenCounter


@dataclass(frozen=True, slots=True)
class ChunkDraft:
    """Chunk text before document metadata and IDs are attached."""

    page_numbers: tuple[int, ...]
    section_type: str
    section_types: tuple[str, ...]
    text: str
    token_count: int


@dataclass(frozen=True, slots=True)
class _Unit:
    text: str
    page_number: int
    section_type: str


@dataclass(frozen=True, slots=True)
class _PackedChunk:
    text: str
    page_numbers: tuple[int, ...]
    section_type: str
    section_types: tuple[str, ...]


class ClauseAwareChunker:
    """Recursively split sections, preferring semantic boundaries."""

    _BOUNDARIES: tuple[str, ...] = (
        r"\n(?=(?:\d+(?:\.\d+)*[.)]?|[A-Z][.)]|[•●▪◦])\s+)",
        r"\n\s*\n+",
        r"\n(?=[A-Z])",
        r"(?<=[.!?;])\s+(?=[A-Z0-9(])",
        r"\s+",
    )

    def __init__(
        self,
        token_counter: TokenCounter,
        min_tokens: int = 400,
        target_tokens: int = 550,
        max_tokens: int = 700,
        overlap_ratio: float = 0.20,
    ) -> None:
        if not 0 < min_tokens <= target_tokens <= max_tokens:
            raise ValueError("Expected 0 < min_tokens <= target_tokens <= max_tokens")
        if not 0.15 <= overlap_ratio <= 0.25:
            raise ValueError("overlap_ratio must be between 0.15 and 0.25")
        self.token_counter = token_counter
        self.min_tokens = min_tokens
        self.target_tokens = target_tokens
        self.max_tokens = max_tokens
        self.overlap_tokens = max(1, round(target_tokens * overlap_ratio))

    def _recursive_split(self, text: str, boundary_index: int = 0) -> list[str]:
        text = text.strip()
        if not text:
            return []
        if self.token_counter.count(text) <= self.max_tokens:
            return [text]
        if boundary_index >= len(self._BOUNDARIES):
            return self.token_counter.split(text, self.max_tokens)

        pieces = [piece.strip() for piece in re.split(self._BOUNDARIES[boundary_index], text) if piece.strip()]
        if len(pieces) == 1:
            return self._recursive_split(text, boundary_index + 1)

        result: list[str] = []
        buffer: list[str] = []
        for piece in pieces:
            candidate = "\n\n".join(buffer + [piece])
            if buffer and self.token_counter.count(candidate) > self.max_tokens:
                joined = "\n\n".join(buffer)
                result.extend(self._recursive_split(joined, boundary_index + 1))
                buffer = [piece]
            else:
                buffer.append(piece)
        if buffer:
            result.extend(self._recursive_split("\n\n".join(buffer), boundary_index + 1))
        return result

    @staticmethod
    def _ordered_pages(units: list[_Unit]) -> tuple[int, ...]:
        return tuple(dict.fromkeys(unit.page_number for unit in units))

    @staticmethod
    def _ordered_sections(units: list[_Unit]) -> tuple[str, ...]:
        return tuple(dict.fromkeys(unit.section_type for unit in units))

    def _packed_chunk(self, units: list[_Unit]) -> _PackedChunk:
        section_sizes: dict[str, int] = {}
        for unit in units:
            section_sizes[unit.section_type] = section_sizes.get(unit.section_type, 0) + self.token_counter.count(unit.text)
        primary_section = max(section_sizes, key=section_sizes.__getitem__)
        return _PackedChunk(
            text="\n\n".join(item.text for item in units).strip(),
            page_numbers=self._ordered_pages(units),
            section_type=primary_section,
            section_types=self._ordered_sections(units),
        )

    def _pack(self, units: list[_Unit]) -> list[_PackedChunk]:
        if not units:
            return []
        chunks: list[_PackedChunk] = []
        current: list[_Unit] = []
        for unit in units:
            candidate = "\n\n".join(item.text for item in current + [unit])
            candidate_tokens = self.token_counter.count(candidate)
            if current and candidate_tokens > self.max_tokens:
                chunks.append(self._packed_chunk(current))
                current = [unit]
                continue
            current.append(unit)
            if candidate_tokens >= self.target_tokens:
                chunks.append(self._packed_chunk(current))
                current = []
        if current:
            remainder = "\n\n".join(item.text for item in current).strip()
            if chunks and self.token_counter.count(remainder) < self.min_tokens:
                combined = chunks[-1].text + "\n\n" + remainder
                if self.token_counter.count(combined) <= self.max_tokens:
                    chunks[-1] = _PackedChunk(
                        combined,
                        tuple(dict.fromkeys(chunks[-1].page_numbers + self._ordered_pages(current))),
                        chunks[-1].section_type,
                        tuple(dict.fromkeys(chunks[-1].section_types + self._ordered_sections(current))),
                    )
                else:
                    chunks.append(self._packed_chunk(current))
            else:
                chunks.append(self._packed_chunk(current))
        return chunks

    def chunk_spans(self, spans: list[SectionSpan]) -> list[ChunkDraft]:
        """Pack intact section spans to the target size and add sliding overlap."""

        if not spans:
            return []
        units = [
            _Unit(text=piece, page_number=span.page_number, section_type=span.section_type)
            for span in spans
            for piece in self._recursive_split(span.text)
        ]
        base_chunks = self._pack(units)
        drafts: list[ChunkDraft] = []
        previous: _PackedChunk | None = None
        for base in base_chunks:
            text = base.text
            page_numbers = base.page_numbers
            section_types = base.section_types
            if previous:
                overlap = self.token_counter.tail(previous.text, self.overlap_tokens)
                available = self.max_tokens - self.token_counter.count(base.text)
                if available > 0:
                    overlap = self.token_counter.tail(overlap, min(available, self.overlap_tokens))
                    if overlap:
                        text = overlap + "\n\n" + base.text
                        page_numbers = tuple(
                            dict.fromkeys((previous.page_numbers[-1],) + base.page_numbers)
                        )
                        section_types = tuple(
                            dict.fromkeys((previous.section_types[-1],) + base.section_types)
                        )
            drafts.append(
                ChunkDraft(
                    page_numbers=page_numbers,
                    section_type=base.section_type,
                    section_types=section_types,
                    text=text.strip(),
                    token_count=self.token_counter.count(text),
                )
            )
            previous = base
        return drafts

    def chunk_sections(self, spans: list[SectionSpan]) -> list[ChunkDraft]:
        """Chunk all section spans in document order."""

        return self.chunk_spans(spans)
