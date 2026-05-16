"""
Tests for the POST /analyze compatibility endpoint.

Verifies that uploading an audio file returns a valid AnalysisResult
with all expected fields.
"""

import io
import struct

import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture
def client():
    """Create a FastAPI test client with lifespan."""
    with TestClient(app) as c:
        yield c


def _generate_wav_bytes(
    frequency: float = 440.0,
    duration_s: float = 4.0,
    sample_rate: int = 16000,
) -> bytes:
    """Generate a synthetic WAV file in memory.

    Creates a sine wave and encodes it as a WAV file using soundfile.

    Returns:
        Raw bytes of a valid WAV file.
    """
    t = np.linspace(0, duration_s, int(sample_rate * duration_s), dtype=np.float32)
    signal = 0.5 * np.sin(2 * np.pi * frequency * t)

    buf = io.BytesIO()
    sf.write(buf, signal, sample_rate, format="WAV", subtype="FLOAT")
    buf.seek(0)
    return buf.read()


class TestAnalyzeEndpoint:
    """Tests for POST /analyze."""

    def test_analyze_returns_200(self, client: TestClient):
        """Valid audio upload should return 200."""
        wav_bytes = _generate_wav_bytes()
        response = client.post(
            "/analyze",
            files={"file": ("test.wav", wav_bytes, "audio/wav")},
        )
        assert response.status_code == 200

    def test_analyze_response_schema(self, client: TestClient):
        """Response should contain all AnalysisResult fields."""
        wav_bytes = _generate_wav_bytes()
        response = client.post(
            "/analyze",
            files={"file": ("test.wav", wav_bytes, "audio/wav")},
        )
        data = response.json()

        assert "gender" in data
        assert "gender_confidence" in data
        assert "age_group" in data
        assert "age_confidence" in data
        assert "language" in data
        assert "language_confidence" in data
        assert "quality" in data

    def test_analyze_quality_metrics(self, client: TestClient):
        """Quality metrics should be present and well-formed."""
        wav_bytes = _generate_wav_bytes()
        response = client.post(
            "/analyze",
            files={"file": ("test.wav", wav_bytes, "audio/wav")},
        )
        quality = response.json()["quality"]

        assert "snr_db" in quality
        assert "clipping_ratio" in quality
        assert "duration_seconds" in quality
        assert "is_acceptable" in quality
        assert quality["duration_seconds"] >= 3.0
        assert 0.0 <= quality["clipping_ratio"] <= 1.0

    def test_analyze_confidence_range(self, client: TestClient):
        """Confidence scores should be in [0, 1]."""
        wav_bytes = _generate_wav_bytes()
        response = client.post(
            "/analyze",
            files={"file": ("test.wav", wav_bytes, "audio/wav")},
        )
        data = response.json()

        assert 0.0 <= data["gender_confidence"] <= 1.0
        assert 0.0 <= data["age_confidence"] <= 1.0

    def test_analyze_empty_file_returns_400(self, client: TestClient):
        """Uploading an empty file should return 400."""
        response = client.post(
            "/analyze",
            files={"file": ("empty.wav", b"", "audio/wav")},
        )
        assert response.status_code == 400

    def test_analyze_invalid_audio_returns_400(self, client: TestClient):
        """Uploading garbage bytes should return 400."""
        response = client.post(
            "/analyze",
            files={"file": ("noise.wav", b"not audio data", "audio/wav")},
        )
        assert response.status_code == 400

    def test_analyze_short_audio_returns_400(self, client: TestClient):
        """Audio shorter than MIN_AUDIO_DURATION should return 400."""
        # Generate very short audio (0.1s)
        wav_bytes = _generate_wav_bytes(duration_s=0.1)
        response = client.post(
            "/analyze",
            files={"file": ("short.wav", wav_bytes, "audio/wav")},
        )
        assert response.status_code == 400

    def test_analyze_language_prediction(self, client: TestClient):
        """Response should include a language prediction with valid confidence."""
        wav_bytes = _generate_wav_bytes()
        response = client.post(
            "/analyze",
            files={"file": ("test.wav", wav_bytes, "audio/wav")},
        )
        data = response.json()

        assert "language" in data
        assert isinstance(data["language"], str)
        assert len(data["language"]) > 0
        assert "language_confidence" in data
        assert 0.0 <= data["language_confidence"] <= 1.0
