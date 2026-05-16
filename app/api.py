"""
REST API routes (compatibility endpoint).

While the primary interface is the WebSocket ``/ws/analyze`` endpoint
(see ``streaming.py``), this module provides a traditional HTTP endpoint
for simpler integrations that don't need real-time streaming.

Endpoints:
  - ``POST /analyze``  — Upload an audio file and receive a full
                          ``AnalysisResult`` in the response body.
  - ``GET  /health``   — Health check returning model load status.
"""

from fastapi import APIRouter, File, HTTPException, UploadFile

from app.audio import load_audio, validate_audio
from app.classifier import (
    is_age_classifier_loaded,
    is_gender_classifier_loaded,
    predict_age,
    predict_gender,
)
from app.config import STREAM_MIN_SECONDS, APP_NAME
from app.inference import extract_embedding, is_model_loaded
from app.language import language_service
from app.logger import logger
from app.quality import assess_quality
from app.schemas import AnalysisResult, HealthResponse, QualityMetrics

router = APIRouter()


# ---------------------------------------------------------------------------
# POST /analyze — Compatibility endpoint (full-file upload)
# ---------------------------------------------------------------------------
@router.post(
    "/analyze",
    response_model=AnalysisResult,
    summary="Analyze an uploaded audio file",
    description=(
        "Upload an audio file (WAV, FLAC, OGG, etc.) and receive a complete "
        "voice attribute analysis including gender, age group, quality metrics, "
        "and language detection.\n\n"
        "**Note:** For real-time streaming analysis, use the WebSocket endpoint "
        "at ``/ws/analyze`` instead."
    ),
)
async def analyze_audio(
    file: UploadFile = File(..., description="Audio file to analyze"),
) -> AnalysisResult:
    """Process an uploaded audio file through the full analysis pipeline.

    Pipeline steps:
      1. Decode and resample audio to 16 kHz mono
      2. Validate minimum duration
      3. Extract SpeechBrain ECAPA-TDNN embedding (192-dim)
      4. Classify gender and age group
      5. Assess audio quality (SNR, clipping)
      6. Detect language (placeholder)
    """
    logger.info(
        "Received audio upload",
        extra={"upload_filename": file.filename, "upload_content_type": file.content_type},
    )

    # Step 1: Read and decode audio
    raw_bytes = await file.read()
    if not raw_bytes:
        raise HTTPException(status_code=400, detail="Empty audio file")

    try:
        audio_tensor = load_audio(raw_bytes)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    # Step 2: Validate duration
    if not validate_audio(audio_tensor):
        raise HTTPException(
            status_code=400,
            detail=(
                f"Audio too short. Minimum duration is "
                f"{STREAM_MIN_SECONDS}s."
            ),
        )

    # Step 3: Extract embedding
    try:
        embedding = extract_embedding(audio_tensor)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))

    # Step 4: Classify
    gender, gender_conf = predict_gender(embedding)
    age_group, age_conf = predict_age(embedding)

    # Step 5: Quality assessment
    quality = assess_quality(audio_tensor)

    # Step 6: Language detection (best-effort, never breaks the response)
    lang_result = language_service.detect_language(
        audio_tensor.numpy(), sample_rate=16000
    )

    logger.info(
        "Analysis complete",
        extra={"gender": gender, "age_group": age_group},
    )

    return AnalysisResult(
        gender=gender,
        gender_confidence=round(gender_conf, 4),
        age_group=age_group,
        age_confidence=round(age_conf, 4),
        language=lang_result["prediction"],
        language_confidence=round(lang_result["confidence"], 4),
        quality=QualityMetrics(**quality),
    )


# ---------------------------------------------------------------------------
# GET /health — Service health check
# ---------------------------------------------------------------------------
@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check",
    description="Returns the service status and model load state.",
)
async def health_check() -> HealthResponse:
    """Return service health status including model availability."""
    embedding_loaded = is_model_loaded()
    gender_loaded = is_gender_classifier_loaded()
    age_loaded = is_age_classifier_loaded()

    # Service is "ok" if at least the embedding model is loaded
    status = "ok" if embedding_loaded else "degraded"

    return HealthResponse(
        status=status,
        embedding_model_loaded=embedding_loaded,
        gender_classifier_loaded=gender_loaded,
        age_classifier_loaded=age_loaded,
        version=APP_NAME,
    )
