"""Persistent Chroma storage for precomputed insurance chunk embeddings."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from .embeddings import Embedder
from .models import ChunkRecord, DenseHit

LOGGER = logging.getLogger(__name__)


def load_chunk_file(path: Path) -> list[ChunkRecord]:
    """Load and validate the processed-chunk array written by ingestion."""

    with path.open("r", encoding="utf-8") as input_file:
        payload = json.load(input_file)
    if not isinstance(payload, list):
        raise ValueError(f"Expected a JSON array in {path}")
    records = [ChunkRecord.from_dict(item) for item in payload]
    if len({record.chunk_id for record in records}) != len(records):
        raise ValueError(f"Duplicate chunk IDs found in {path}")
    return records


class ChromaVectorStore:
    """Own a persistent cosine-similarity Chroma collection."""

    def __init__(
        self,
        persist_directory: Path,
        embedder: Embedder,
        collection_name: str = "insurance_products",
        reset_if_model_mismatch: bool = False,
    ) -> None:
        try:
            import chromadb
            from chromadb.config import Settings
        except ImportError as exc:
            raise RuntimeError("chromadb is required. Install backend/requirements.txt.") from exc

        self.persist_directory = persist_directory.resolve()
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        self.embedder = embedder
        self.client = chromadb.PersistentClient(
            path=str(self.persist_directory),
            settings=Settings(anonymized_telemetry=False),
        )
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine", "embedding_model": embedder.model_name},
        )
        stored_model = (self.collection.metadata or {}).get("embedding_model")
        if stored_model and stored_model != embedder.model_name:
            if reset_if_model_mismatch:
                self.client.delete_collection(collection_name)
                self.collection = self.client.create_collection(
                    name=collection_name,
                    metadata={"hnsw:space": "cosine", "embedding_model": embedder.model_name},
                )
                return
            raise ValueError(
                f"Collection uses {stored_model!r}, but {embedder.model_name!r} was requested. "
                "Use the original model, a different collection, or force a model reset while indexing."
            )

    @property
    def count(self) -> int:
        return int(self.collection.count())

    def _all_ids(self) -> set[str]:
        if self.count == 0:
            return set()
        result = self.collection.get(include=[])
        return set(result.get("ids") or [])

    def index_records(
        self,
        records: list[ChunkRecord],
        batch_size: int = 8,
        force: bool = False,
    ) -> dict[str, int]:
        """Incrementally synchronize records without re-embedding unchanged IDs."""

        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        incoming_ids = {record.chunk_id for record in records}
        existing_ids = self._all_ids()
        if force and existing_ids:
            self.collection.delete(ids=sorted(existing_ids))
            existing_ids = set()

        stale_ids = existing_ids - incoming_ids
        if stale_ids:
            self.collection.delete(ids=sorted(stale_ids))

        pending = [record for record in records if record.chunk_id not in existing_ids]
        for start in range(0, len(pending), batch_size):
            batch = pending[start : start + batch_size]
            LOGGER.info(
                "Embedding chunks %d-%d of %d",
                start + 1,
                start + len(batch),
                len(pending),
            )
            embeddings = self.embedder.embed_documents([record.search_text for record in batch])
            self.collection.upsert(
                ids=[record.chunk_id for record in batch],
                embeddings=embeddings,
                documents=[record.text for record in batch],
                metadatas=[record.to_chroma_metadata() for record in batch],
            )
        return {
            "total": len(records),
            "embedded": len(pending),
            "unchanged": len(records) - len(pending),
            "deleted": len(stale_ids),
        }

    def load_records(self) -> list[ChunkRecord]:
        """Load records from Chroma for BM25 construction."""

        if self.count == 0:
            return []
        result = self.collection.get(include=["documents", "metadatas"])
        ids = result.get("ids") or []
        documents = result.get("documents") or [None] * len(ids)
        metadatas = result.get("metadatas") or [None] * len(ids)
        records = [
            ChunkRecord.from_chroma(chunk_id, document, metadata)
            for chunk_id, document, metadata in zip(ids, documents, metadatas, strict=True)
        ]
        return sorted(records, key=lambda record: record.chunk_id)

    def dense_search(
        self,
        query: str,
        top_k: int = 25,
        product_names: tuple[str, ...] = (),
    ) -> list[DenseHit]:
        """Return cosine-ranked dense results with full records."""

        if top_k < 1:
            raise ValueError("top_k must be positive")
        if self.count == 0:
            return []
        query_embedding = self.embedder.embed_query(query)
        query_arguments: dict[str, Any] = {
            "query_embeddings": [query_embedding],
            "n_results": min(top_k, self.count),
            "include": ["documents", "metadatas", "distances"],
        }
        if len(product_names) == 1:
            query_arguments["where"] = {"product_name": {"$eq": product_names[0]}}
        elif product_names:
            query_arguments["where"] = {"product_name": {"$in": list(product_names)}}
        result: dict[str, Any] = self.collection.query(
            **query_arguments,
        )
        ids = (result.get("ids") or [[]])[0]
        documents = (result.get("documents") or [[]])[0]
        metadatas = (result.get("metadatas") or [[]])[0]
        distances = (result.get("distances") or [[]])[0]
        hits: list[DenseHit] = []
        for rank, (chunk_id, document, metadata, distance) in enumerate(
            zip(ids, documents, metadatas, distances, strict=True), start=1
        ):
            numeric_distance = float(distance)
            hits.append(
                DenseHit(
                    record=ChunkRecord.from_chroma(chunk_id, document, metadata),
                    score=1.0 - numeric_distance,
                    distance=numeric_distance,
                    rank=rank,
                )
            )
        return hits
