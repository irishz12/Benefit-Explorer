"""CLI for persistent indexing and hybrid retrieval.

Examples from ``backend``::

    python -m src.retrieval.run_retrieval index
    python -m src.retrieval.run_retrieval query "What are the surrender benefits?"
    python -m src.retrieval.run_retrieval demo --top-k 5
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from .bm25_retriever import BM25Retriever
from .embeddings import DEFAULT_MODEL, BGEEmbedder
from .hybrid_retriever import HybridRetriever
from .models import HybridResult
from .vector_store import ChromaVectorStore, load_chunk_file

BACKEND_DIR = Path(__file__).resolve().parents[2]
DEFAULT_CHUNKS = BACKEND_DIR / "data" / "processed_chunks.json"
DEFAULT_CHROMA = BACKEND_DIR / "data" / "chroma_db"
SAMPLE_QUERIES = (
    "What benefits are paid when the policyholder dies?",
    "What are the entry age and eligibility requirements?",
    "Which exclusions and waiting periods apply?",
    "Can the policy be surrendered and what value is payable?",
)


def _add_retrieval_limits(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--top-k", type=int, default=20, help="Final fused results (default: 20).")
    parser.add_argument("--dense-k", type=int, default=25)
    parser.add_argument("--sparse-k", type=int, default=25)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Index and retrieve insurance product chunks.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Sentence Transformers model name.")
    parser.add_argument("--device", default=None, help="Device such as cpu, mps, or cuda.")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--offline", action="store_true", help="Only load an already cached model.")
    parser.add_argument("--persist-dir", type=Path, default=DEFAULT_CHROMA)
    parser.add_argument("--collection", default="insurance_products")
    parser.add_argument("--verbose", action="store_true")
    subparsers = parser.add_subparsers(dest="command", required=True)

    index_parser = subparsers.add_parser("index", help="Embed chunks and synchronize Chroma.")
    index_parser.add_argument("--chunks", type=Path, default=DEFAULT_CHUNKS)
    index_parser.add_argument(
        "--force", action="store_true", help="Delete and re-embed all records."
    )

    query_parser = subparsers.add_parser("query", help="Run one hybrid query.")
    query_parser.add_argument("query")
    _add_retrieval_limits(query_parser)

    demo_parser = subparsers.add_parser("demo", help="Run representative test queries.")
    _add_retrieval_limits(demo_parser)
    return parser


def _components(args: argparse.Namespace) -> tuple[BGEEmbedder, ChromaVectorStore]:
    embedder = BGEEmbedder(
        model_name=args.model,
        device=args.device,
        batch_size=args.batch_size,
        offline=args.offline,
    )
    store = ChromaVectorStore(
        args.persist_dir,
        embedder,
        args.collection,
        reset_if_model_mismatch=args.command == "index" and args.force,
    )
    return embedder, store


def _print_results(query: str, results: list[HybridResult]) -> None:
    print(f"\nQUERY: {query}")
    for rank, result in enumerate(results, start=1):
        payload = result.to_dict()
        preview = " ".join(payload["text"].split())[:320]
        pages = ",".join(str(page) for page in payload["page_numbers"])
        print(
            f"\n[{rank}] RRF={payload['rrf_score']:.6f} "
            f"dense={payload['dense_score']} bm25={payload['bm25_score']}\n"
            f"    {payload['product_name']} | {payload['section_type']} | "
            f"pages {pages} | {payload['source_file']}\n"
            f"    {preview}"
        )


def main() -> None:
    args = build_parser().parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    _, store = _components(args)
    if args.command == "index":
        records = load_chunk_file(args.chunks)
        summary = store.index_records(records, batch_size=args.batch_size, force=args.force)
        print(json.dumps(summary, indent=2))
        return

    records = store.load_records()
    retriever = HybridRetriever(store, bm25_retriever=BM25Retriever(records))
    if args.command == "query":
        results = retriever.retrieve(
            args.query,
            final_k=args.top_k,
            dense_k=args.dense_k,
            sparse_k=args.sparse_k,
        )
        _print_results(args.query, results)
    else:
        for query in SAMPLE_QUERIES:
            results = retriever.retrieve(
                query,
                final_k=args.top_k,
                dense_k=args.dense_k,
                sparse_k=args.sparse_k,
            )
            _print_results(query, results)


if __name__ == "__main__":
    main()
