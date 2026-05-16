"""
FastAPI application entry point.

This module wires together all components:
  - REST routes (``api.py``)     → ``POST /analyze``, ``GET /health``
  - WebSocket handler (``streaming.py``) → ``/ws/analyze``
  - Model loading via lifespan hook

Run locally with:
    uv run uvicorn app.main:app --reload

Or in production:
    uv run uvicorn app.main:app --host 0.0.0.0 --port 8000
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse

from app.api import router as api_router
from app.classifier import load_classifiers
from app.config import APP_NAME
from app.inference import load_model
from app.language import language_service
from app.logger import logger
from app.streaming import handle_streaming


# ---------------------------------------------------------------------------
# Lifespan: load models once at startup, clean up on shutdown
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup / shutdown lifecycle hook.

    On startup:
      1. Load the SpeechBrain ECAPA-TDNN embedding model (downloads on
         first run, cached locally thereafter).
      2. Load the trained gender and age classifiers from ``models/``.
         If the joblib files don't exist yet (classifiers not trained),
         a warning is logged and predictions return ``"unknown"``.

    On shutdown:
      - Log a clean shutdown message.
    """
    logger.info("Starting voice-attribute-service", extra={"app_name": APP_NAME})

    # Load SpeechBrain embedding model (speaker embeddings for gender/age)
    load_model()

    # Load scikit-learn classifiers (gender, age)
    load_classifiers()

    # Load SpeechBrain language-ID model (VoxLingua107)
    language_service.load_model()

    logger.info("All models loaded — service is ready")

    yield  # Application runs here

    logger.info("Shutting down voice-attribute-service")


# ---------------------------------------------------------------------------
# Create FastAPI application
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Voice Attribute Service",
    description=(
        "Streaming-first API for real-time voice attribute analysis.\n\n"
        "**Primary endpoint:** WebSocket `ws://host/ws/analyze`\n\n"
        "**Compatibility endpoint:** `POST /analyze` (file upload)\n\n"
        "Extracts speaker embeddings using SpeechBrain ECAPA-TDNN and "
        "classifies gender, age group, audio quality, and language."
    ),
    version="0.1.0",
    lifespan=lifespan,
)

# ---------------------------------------------------------------------------
# CORS middleware (permissive for development; tighten for production)
# ---------------------------------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Include REST API routes
# ---------------------------------------------------------------------------
app.include_router(api_router)


# ---------------------------------------------------------------------------
# WebSocket endpoint (primary real-time interface)
# ---------------------------------------------------------------------------
@app.websocket("/ws/analyze")
async def websocket_analyze(websocket: WebSocket):
    """WebSocket endpoint for real-time voice attribute streaming.

    Connect with a WebSocket client and stream binary audio data.
    See ``streaming.py`` for the full protocol description.
    """
    await handle_streaming(websocket)


# ---------------------------------------------------------------------------
# Root → redirect to interactive API docs
# ---------------------------------------------------------------------------
@app.get("/", include_in_schema=False)
async def root():
    """Redirect root path to the Swagger UI docs."""
    return RedirectResponse(url="/docs")
