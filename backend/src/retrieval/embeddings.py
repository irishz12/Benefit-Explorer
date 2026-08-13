"""Lazy, offline-aware BGE embedding support."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any, Protocol


DEFAULT_MODEL = "BAAI/bge-m3"
_BGE_EN_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


class Embedder(Protocol):
    """Minimal interface accepted by indexing and retrieval."""

    @property
    def model_name(self) -> str:
        """Stable model identifier used to protect persisted collections."""

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed passages."""

    def embed_query(self, query: str) -> list[float]:
        """Embed one search query."""


class BGEEmbedder:
    """Sentence Transformers wrapper for BGE-M3 and BGE English models."""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        device: str | None = None,
        batch_size: int = 8,
        cache_dir: Path | None = None,
        offline: bool = False,
        show_progress: bool = True,
    ) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        self._model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self.cache_dir = cache_dir
        self.offline = offline
        self.show_progress = show_progress
        self._model: Any | None = None

    @property
    def model_name(self) -> str:
        return self._model_name

    def _load(self) -> Any:
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise RuntimeError(
                    "sentence-transformers is required. Install backend/requirements.txt."
                ) from exc
            self._model = SentenceTransformer(
                self.model_name,
                device=self.device,
                cache_folder=str(self.cache_dir) if self.cache_dir else None,
                local_files_only=self.offline,
                trust_remote_code=False,
            )
        return self._model

    def _encode(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        vectors = self._load().encode(
            list(texts),
            batch_size=self.batch_size,
            show_progress_bar=self.show_progress,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return vectors.astype("float32", copy=False).tolist()

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return self._encode(texts)

    def embed_query(self, query: str) -> list[float]:
        query = query.strip()
        if not query:
            raise ValueError("query cannot be empty")
        if "bge-large-en" in self.model_name.casefold():
            query = _BGE_EN_QUERY_PREFIX + query
        return self._encode([query])[0]
