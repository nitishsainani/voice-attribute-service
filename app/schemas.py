"""
Pydantic schemas for request/response models.

These schemas define the API contract for both the REST and WebSocket
endpoints. They also serve as documentation in the auto-generated
OpenAPI (Swagger) docs.
"""

from typing import Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Quality metrics returned as part of every analysis
# ---------------------------------------------------------------------------
class QualityMetrics(BaseModel):
    """Audio quality assessment metrics."""
    snr_db: float = Field(..., description="Estimated signal-to-noise ratio in dB")
    clipping_ratio: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Fraction of samples that appear clipped (0.0–1.0)",
    )
    duration_seconds: float = Field(..., description="Duration of the audio in seconds")
    is_acceptable: bool = Field(
        ...,
        description="Whether the audio quality meets minimum thresholds",
    )


# ---------------------------------------------------------------------------
# Full analysis result (returned by POST /analyze and as final WS message)
# ---------------------------------------------------------------------------
class AnalysisResult(BaseModel):
    """Complete voice attribute analysis result."""
    gender: str = Field(..., description="Predicted gender label (male / female / unknown)")
    gender_confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Confidence score for gender prediction"
    )
    age_group: str = Field(
        ..., description="Predicted age group (child / young_adult / adult / senior / unknown)"
    )
    age_confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Confidence score for age prediction"
    )
    language: str = Field(
        "unknown", description="Detected language code (ISO 639-1) or 'unknown'"
    )
    language_confidence: float = Field(
        0.0, ge=0.0, le=1.0, description="Confidence for language detection"
    )
    quality: QualityMetrics = Field(..., description="Audio quality metrics")


# ---------------------------------------------------------------------------
# Streaming chunk (intermediate result sent during WebSocket streaming)
# ---------------------------------------------------------------------------
class StreamingChunk(BaseModel):
    """Partial analysis result emitted per audio chunk during streaming."""
    chunk_index: int = Field(..., description="Zero-based index of the processed chunk")
    timestamp_start: float = Field(..., description="Start time of this chunk in seconds")
    timestamp_end: float = Field(..., description="End time of this chunk in seconds")
    gender: str = Field(..., description="Per-chunk gender prediction")
    gender_confidence: float = Field(..., ge=0.0, le=1.0)
    age_group: str = Field(..., description="Per-chunk age group prediction")
    age_confidence: float = Field(..., ge=0.0, le=1.0)
    language: str = Field(
        "unknown", description="Per-chunk detected language code (ISO 639-1) or 'unknown'"
    )
    language_confidence: float = Field(
        0.0, ge=0.0, le=1.0, description="Per-chunk confidence for language detection"
    )


# ---------------------------------------------------------------------------
# Streaming final message (wraps the aggregated result)
# ---------------------------------------------------------------------------
class StreamingFinalResult(BaseModel):
    """Final aggregated result sent when the WebSocket stream ends."""
    type: str = Field("final", description="Message type identifier")
    result: AnalysisResult


# ---------------------------------------------------------------------------
# Health check response
# ---------------------------------------------------------------------------
class HealthResponse(BaseModel):
    """Health check response indicating service readiness."""
    status: str = Field(..., description="Service status: 'ok' or 'degraded'")
    embedding_model_loaded: bool = Field(
        ..., description="Whether the SpeechBrain embedding model is loaded"
    )
    gender_classifier_loaded: bool = Field(
        ..., description="Whether the gender classifier is available"
    )
    age_classifier_loaded: bool = Field(
        ..., description="Whether the age classifier is available"
    )
    version: str = Field(..., description="Application version string")
