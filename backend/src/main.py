"""FastAPI application exposing the validated insurance QA pipeline."""

from __future__ import annotations

import asyncio
import json
import logging
import os
from functools import lru_cache
from pathlib import Path
from typing import AsyncIterator

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from src.generation.citation_utils import CitationValidationError
from src.generation.generator import BedrockGenerator, GroundedQAPipeline, QARun
from src.generation.reranker import BGEReranker, DEFAULT_RERANKER_MODEL
from src.retrieval.embeddings import BGEEmbedder, DEFAULT_MODEL
from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.vector_store import ChromaVectorStore

BACKEND_DIR = Path(__file__).resolve().parents[1]
DEFAULT_CHROMA = BACKEND_DIR / "data" / "chroma_db"
DEFAULT_MANTLE_MODEL = "qwen.qwen3-next-80b-a3b-instruct"
DEFAULT_AWS_REGION = "us-east-1"
# The backend's private environment file is authoritative. This prevents a
# stale shell/session Bedrock token from silently shadowing the configured key.
load_dotenv(BACKEND_DIR / ".env", override=True)
LOGGER = logging.getLogger(__name__)


def _optional_device(env_var: str) -> str | None:
    """Read a torch device override; blank/unset keeps automatic device selection."""

    return os.getenv(env_var, "").strip() or None


class ChatRequest(BaseModel):
    """One user question sent by the web application."""

    message: str = Field(min_length=1, max_length=4000)


class CitationResponse(BaseModel):
    index: int
    chunk_id: str
    product: str
    page: int
    supporting_text: str


class ChatResponse(BaseModel):
    answer: str
    citations: list[CitationResponse]
    detected_products: list[str]
    retrieval_mode: str


@lru_cache(maxsize=1)
def get_rag_components() -> tuple[HybridRetriever, BGEReranker]:
    """Load the large local retrieval and reranking models once."""

    embedding_device = _optional_device("EMBEDDING_DEVICE")
    reranker_device = _optional_device("RERANKER_DEVICE")
    embedder = BGEEmbedder(
        model_name=os.getenv("EMBEDDING_MODEL", DEFAULT_MODEL),
        device=embedding_device,
        offline=os.getenv("RAG_OFFLINE", "true").casefold() == "true",
        show_progress=False,
    )
    store = ChromaVectorStore(
        Path(os.getenv("CHROMA_PERSIST_DIR", DEFAULT_CHROMA)),
        embedder,
        os.getenv("CHROMA_COLLECTION", "insurance_products"),
    )
    retriever = HybridRetriever(
        store,
        product_aware=True,
        min_filtered_candidates=2,
        product_score_boost=5.0,
    )
    reranker = BGEReranker(
        model_name=os.getenv("RERANKER_MODEL", DEFAULT_RERANKER_MODEL),
        device=reranker_device,
        offline=os.getenv("RAG_OFFLINE", "true").casefold() == "true",
        batch_size=int(os.getenv("RERANKER_BATCH_SIZE", "8")),
        focus_embedder=embedder,
        product_match_boost=0.35,
        show_progress=False,
    )
    return retriever, reranker


def build_mantle_generator(model: str | None = None) -> BedrockGenerator:
    """Create a Bedrock Mantle Chat Completions generator from backend settings."""

    api_key = os.getenv("AWS_BEARER_TOKEN_BEDROCK")
    if not api_key:
        raise RuntimeError(
            "Set AWS_BEARER_TOKEN_BEDROCK in backend/.env before starting the API."
        )
    try:
        from openai import OpenAI
    except ImportError as error:
        raise RuntimeError("Install the openai package from backend/requirements.txt") from error
    region = os.getenv("AWS_REGION", DEFAULT_AWS_REGION)
    base_url = os.getenv(
        "OPENAI_BASE_URL",
        f"https://bedrock-mantle.{region}.api.aws/v1",
    )
    client = OpenAI(api_key=api_key, base_url=base_url)
    return BedrockGenerator(
        model=model or os.getenv("MANTLE_MODEL", DEFAULT_MANTLE_MODEL),
        client=client,
        max_completion_tokens=int(os.getenv("MANTLE_MAX_OUTPUT_TOKENS", "1100")),
        max_json_retries=1,
    )


@lru_cache(maxsize=1)
def build_pipeline() -> GroundedQAPipeline:
    """Build the QA pipeline once; all credentials remain on the backend."""

    retriever, reranker = get_rag_components()
    return GroundedQAPipeline(
        retriever=retriever,
        reranker=reranker,
        generator=build_mantle_generator(),
        retrieval_k=40,
        final_context_k=4,
    )


def _serialize_run(run: QARun) -> ChatResponse:
    response = run.response
    return ChatResponse(
        answer=response.answer,
        citations=[
            CitationResponse(
                index=citation.index,
                chunk_id=citation.chunk_id,
                product=citation.product,
                page=citation.page,
                supporting_text=citation.supporting_text,
            )
            for citation in response.citations
        ],
        detected_products=list(run.detected_products),
        retrieval_mode=run.product_retrieval_mode,
    )


app = FastAPI(title="BenefitExplorer API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        origin.strip()
        for origin in os.getenv("FRONTEND_ORIGINS", "http://localhost:3000").split(",")
        if origin.strip()
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict[str, str]:
    return {
        "status": "ok",
        "generator": "aws-bedrock-mantle",
        "model": os.getenv("MANTLE_MODEL", DEFAULT_MANTLE_MODEL),
        "credential_configured": str(
            bool(os.getenv("AWS_BEARER_TOKEN_BEDROCK"))
        ).lower(),
    }


@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    try:
        pipeline = build_pipeline()
        run = await asyncio.to_thread(pipeline.answer, request.message.strip())
        return _serialize_run(run)
    except CitationValidationError as error:
        raise HTTPException(status_code=422, detail={"message": str(error), "reasons": error.reasons}) from error
    except (RuntimeError, ValueError) as error:
        raise HTTPException(status_code=500, detail=str(error)) from error
    except Exception as error:
        code, message = _public_error(error)
        status_code = 429 if code == "rate_limited" else 401 if code == "invalid_api_key" else 502
        raise HTTPException(status_code=status_code, detail={"code": code, "message": message}) from error


def _event(event: str, **payload: object) -> bytes:
    return (json.dumps({"event": event, **payload}, ensure_ascii=False) + "\n").encode()


def _public_error(error: Exception) -> tuple[str, str]:
    """Return a safe client error without leaking provider or account details."""

    body = getattr(error, "body", None)
    error_detail = body.get("error", body) if isinstance(body, dict) else {}
    provider_message = (
        str(error_detail.get("message", "")).casefold()
        if isinstance(error_detail, dict)
        else ""
    )
    if isinstance(error, RuntimeError) and "AWS_BEARER_TOKEN_BEDROCK" in str(error):
        return "configuration_error", str(error)
    if error.__class__.__name__ == "RateLimitError":
        return (
            "rate_limited",
            "AWS Bedrock Mantle has reached a usage limit. Try again later.",
        )
    if error.__class__.__name__ in {"AuthenticationError", "PermissionDeniedError"}:
        if "not available for this account" in provider_message:
            return (
                "model_unavailable",
                "The Bedrock key is valid, but the configured model is not available for this account. A /models listing does not confirm invocation access.",
            )
        return (
            "invalid_api_key",
            "AWS Bedrock rejected the backend API key, model access, or permissions. Check backend/.env and verify the configured model with a real invocation.",
        )
    if error.__class__.__name__ == "NotFoundError":
        return (
            "model_unavailable",
            "The configured Mantle model is not invokable with this account and region. A /models listing alone does not confirm invocation access.",
        )
    return "generation_failed", "The answer could not be generated. Please try again."


async def _answer_stream(
    message: str,
) -> AsyncIterator[bytes]:
    yield _event("status", message="Searching product brochures…")
    try:
        pipeline = build_pipeline()
        run = await asyncio.to_thread(pipeline.answer, message)
        payload = _serialize_run(run)
        yield _event("status", message="Verifying citations…")
        # Structured generation is verified before streaming. Small deltas still
        # provide a responsive reading experience without exposing invalid text.
        words = payload.answer.split(" ")
        for index, word in enumerate(words):
            yield _event("answer_delta", delta=("" if index == 0 else " ") + word)
            await asyncio.sleep(0.012)
        yield _event("result", **payload.model_dump())
    except CitationValidationError as error:
        yield _event("error", message=str(error), reasons=list(error.reasons))
    except Exception as error:  # Keep provider/account details out of the browser.
        LOGGER.exception(
            "QA request failed (class=%s status=%s body=%s)",
            error.__class__.__name__,
            getattr(error, "status_code", None),
            getattr(error, "body", None),
        )
        code, message = _public_error(error)
        yield _event("error", code=code, message=message)


@app.post("/api/chat/stream")
async def chat_stream(request: ChatRequest) -> StreamingResponse:
    return StreamingResponse(
        _answer_stream(
            request.message.strip(),
        ),
        media_type="application/x-ndjson",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
