"""FastAPI Application Initialization, Middleware & CORS (Step 38).

Initializes the production FastAPI application with:
  1. CORS Middleware configured for local and deployed UI origins.
  2. Request Correlation ID Middleware (injects / passes ``X-Request-ID``).
  3. Latency & Performance Stopwatch (injects ``X-Response-Time-Ms``).
  4. Global exception handlers with structured logging.

Usage::

    # Run with uvicorn:
    # uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
"""

from __future__ import annotations

import time
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.core.config import settings
from src.core.logger import logger


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Application startup and shutdown lifespan events with eager model warming."""
    logger.info(f"Starting {settings.APP_NAME} (FastAPI Backend)...")
    logger.info(f"Environment: {settings.ENVIRONMENT} | LLM Provider: {settings.LLM_PROVIDER}")

    # Eagerly warm up ChromaDB, SentenceTransformers embedder and Cross-Encoder reranker
    try:
        from src.api.routes_rec import get_services
        logger.info("Pre-warming ChromaDB and ML models into memory...")
        t0 = time.perf_counter()
        store, embedder, reranker, recommender = get_services()
        # Warm-up inference
        _ = embedder.embed_query("anime discovery")
        logger.info(f"All ML models pre-warmed successfully in {time.perf_counter() - t0:.2f}s")
    except Exception as exc:
        logger.warning(f"Model pre-warming non-fatal warning: {exc}")

    yield
    logger.info(f"Shutting down {settings.APP_NAME}...")


def create_app() -> FastAPI:
    """FastAPI application factory."""
    application = FastAPI(
        title=settings.APP_NAME,
        version="1.0.0",
        description="Production Conversational Anime Recommendation RAG Engine",
        lifespan=lifespan,
    )

    # ── CORS Middleware ────────────────────────────────────────────────────────
    # In production, specify explicit UI domains; in dev, allow localhost & all
    application.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Request Timing & Correlation ID Middleware ─────────────────────────────
    @application.middleware("http")
    async def request_timing_and_id_middleware(request: Request, call_next: Any) -> Any:
        req_id = request.headers.get("X-Request-ID") or f"req-{uuid.uuid4().hex[:8]}"
        request.state.request_id = req_id
        start_time = time.perf_counter()

        response = await call_next(request)

        duration_ms = round((time.perf_counter() - start_time) * 1000, 2)
        response.headers["X-Request-ID"] = req_id
        response.headers["X-Response-Time-Ms"] = str(duration_ms)
        return response

    # ── Global Exception Handlers ──────────────────────────────────────────────
    @application.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        req_id = getattr(request.state, "request_id", "unknown")
        logger.error(f"Unhandled server error on {request.url.path} [{req_id}]: {exc}", exc_info=True)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "error": "Internal server error",
                "detail": str(exc) if settings.DEBUG else "An unexpected error occurred.",
                "request_id": req_id,
            },
            headers={"X-Request-ID": req_id},
        )

    # ── Register API Routers ───────────────────────────────────────────────────
    from src.api.routes_chat import router as chat_router
    from src.api.routes_meta import router as meta_router
    from src.api.routes_rec import router as rec_router

    application.include_router(rec_router)
    application.include_router(chat_router)
    application.include_router(meta_router)

    return application


app = create_app()
