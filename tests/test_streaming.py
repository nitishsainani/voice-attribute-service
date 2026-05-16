"""
Tests for the WebSocket streaming endpoint.

Verifies that the /ws/analyze WebSocket endpoint accepts connections,
processes binary audio chunks, and returns properly formatted JSON
messages.
"""

import json
import struct

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.config import STREAM_INFERENCE_INTERVAL_SECONDS


@pytest.fixture
def client():
    """Create a FastAPI test client with lifespan."""
    with TestClient(app) as c:
        yield c


def _generate_sine_wave(
    frequency: float = 440.0,
    duration_s: float = 3.0,
    sample_rate: int = 16000,
) -> bytes:
    """Generate a sine wave as PCM float32 little-endian bytes.

    This creates a synthetic audio signal for testing without needing
    real audio files.

    Args:
        frequency: Tone frequency in Hz.
        duration_s: Duration in seconds.
        sample_rate: Sample rate in Hz.

    Returns:
        Raw PCM float32 bytes.
    """
    t = np.linspace(0, duration_s, int(sample_rate * duration_s), dtype=np.float32)
    signal = 0.5 * np.sin(2 * np.pi * frequency * t)
    return struct.pack(f"<{len(signal)}f", *signal)


class TestWebSocketStreaming:
    """Tests for the WebSocket /ws/analyze endpoint."""

    def test_websocket_connection_accepted(self, client: TestClient):
        """WebSocket connection should be accepted."""
        with client.websocket_connect("/ws/analyze") as ws:
            # Send END to gracefully close
            ws.send_text("END")

    def test_websocket_config_ack(self, client: TestClient):
        """Sending a JSON config should receive a config_ack response."""
        with client.websocket_connect("/ws/analyze") as ws:
            config = {"sample_rate": 16000, "encoding": "pcm_f32le"}
            ws.send_text(json.dumps(config))
            response = ws.receive_json()

            assert response["type"] == "config_ack"
            assert response["config"]["sample_rate"] == 16000

            ws.send_text("END")

    def test_websocket_processes_audio_chunk(self, client: TestClient):
        """Sending enough audio data should produce a chunk response."""
        # Generate audio longer than one chunk
        chunk_duration = STREAM_INFERENCE_INTERVAL_SECONDS
        audio_bytes = _generate_sine_wave(
            duration_s=chunk_duration + 0.5,
            sample_rate=16000,
        )

        with client.websocket_connect("/ws/analyze") as ws:
            # Send all audio bytes at once
            ws.send_bytes(audio_bytes)

            # Should receive at least one chunk result
            response = ws.receive_json()
            assert response["type"] == "chunk"
            assert "chunk_index" in response
            assert "gender" in response
            assert "age_group" in response
            assert "gender_confidence" in response
            assert "age_confidence" in response

            ws.send_text("END")

    def test_websocket_chunk_has_timestamps(self, client: TestClient):
        """Chunk responses should include start/end timestamps."""
        audio_bytes = _generate_sine_wave(
            duration_s=STREAM_INFERENCE_INTERVAL_SECONDS + 0.5,
            sample_rate=16000,
        )

        with client.websocket_connect("/ws/analyze") as ws:
            ws.send_bytes(audio_bytes)
            response = ws.receive_json()

            assert "timestamp_start" in response
            assert "timestamp_end" in response
            assert response["timestamp_start"] >= 0
            assert response["timestamp_end"] > response["timestamp_start"]

            ws.send_text("END")

    def test_websocket_invalid_json_returns_error(self, client: TestClient):
        """Sending invalid JSON text should return an error message."""
        with client.websocket_connect("/ws/analyze") as ws:
            ws.send_text("this is not json")
            response = ws.receive_json()

            assert response["type"] == "error"

            ws.send_text("END")

    def test_websocket_chunk_has_language(self, client: TestClient):
        """Chunk responses should include language detection fields."""
        audio_bytes = _generate_sine_wave(
            duration_s=STREAM_INFERENCE_INTERVAL_SECONDS + 0.5,
            sample_rate=16000,
        )

        with client.websocket_connect("/ws/analyze") as ws:
            ws.send_bytes(audio_bytes)
            response = ws.receive_json()

            assert response["type"] == "chunk"
            assert "language" in response
            assert isinstance(response["language"], str)
            assert "language_confidence" in response
            assert 0.0 <= response["language_confidence"] <= 1.0

            ws.send_text("END")
