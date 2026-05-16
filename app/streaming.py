"""
WebSocket streaming handler for real-time voice analysis.

This is the **primary** interface of the service.  Clients connect via
``ws://host:port/ws/analyze`` and stream raw audio bytes in real time.

Protocol:
  1. Client opens a WebSocket connection.
  2. Client sends binary messages containing audio data (PCM float32 or
     encoded audio bytes).  An initial JSON config message may be sent
     to specify the sample rate and encoding.
  3. For each processed chunk, the server sends back a JSON
     ``StreamingChunk`` with per-chunk predictions.
  4. When the client closes the connection (or sends a text message
     ``"END"``), the server computes an aggregated ``AnalysisResult``
     over all chunks and sends it as a final JSON message before closing.

Design notes:
  - Audio is buffered internally.  Once enough data accumulates for a
    chunk (default 2 seconds), it is processed through the full pipeline
    (embedding extraction → classification → quality check).
  - This approach allows near-real-time feedback while keeping each
    embedding extraction large enough for reliable predictions.
"""

import json
import struct

import numpy as np
import torch
from fastapi import WebSocket, WebSocketDisconnect

from app.audio import load_audio_from_numpy, validate_audio, get_duration
from app.classifier import predict_gender, predict_age
from app.config import STREAM_INFERENCE_INTERVAL_SECONDS
from app.inference import extract_embedding
from app.language import language_service
from app.logger import logger
from app.quality import assess_quality
from app.schemas import (
    AnalysisResult,
    QualityMetrics,
    StreamingChunk,
    StreamingFinalResult,
)


# ---------------------------------------------------------------------------
# Default stream configuration (can be overridden per-connection)
# ---------------------------------------------------------------------------
_DEFAULT_STREAM_CONFIG = {
    "sample_rate": 16000,
    "encoding": "pcm_f32le",  # little-endian float32 PCM
}


async def handle_streaming(websocket: WebSocket) -> None:
    """Main WebSocket handler for real-time voice analysis.

    This coroutine runs for the lifetime of a single WebSocket connection.

    Args:
        websocket: The FastAPI WebSocket connection.
    """
    await websocket.accept()
    logger.info("WebSocket connection accepted")

    # Per-connection state
    stream_config = dict(_DEFAULT_STREAM_CONFIG)
    audio_buffer = bytearray()
    chunk_index = 0
    all_embeddings: list[np.ndarray] = []
    all_audio_tensors: list[torch.Tensor] = []

    chunk_size_bytes = int(
        stream_config["sample_rate"]
        * STREAM_INFERENCE_INTERVAL_SECONDS
        * 4  # 4 bytes per float32 sample
    )

    try:
        while True:
            message = await websocket.receive()

            # ----------------------------------------------------------
            # Handle text messages (config or END signal)
            # ----------------------------------------------------------
            if "text" in message:
                text = message["text"]
                if text.strip().upper() == "END":
                    logger.info("Received END signal from client")
                    break

                # Try to parse as JSON config
                try:
                    config_update = json.loads(text)
                    stream_config.update(config_update)
                    chunk_size_bytes = int(
                        stream_config["sample_rate"]
                        * STREAM_INFERENCE_INTERVAL_SECONDS
                        * 4
                    )
                    logger.info(
                        "Stream config updated",
                        extra={"config": stream_config},
                    )
                    await websocket.send_json({"type": "config_ack", "config": stream_config})
                    continue
                except (json.JSONDecodeError, TypeError):
                    await websocket.send_json({"type": "error", "message": "Invalid JSON config"})
                    continue

            # ----------------------------------------------------------
            # Handle binary messages (audio data)
            # ----------------------------------------------------------
            if "bytes" in message:
                audio_buffer.extend(message["bytes"])

                # Process complete chunks from the buffer
                while len(audio_buffer) >= chunk_size_bytes:
                    raw_chunk = bytes(audio_buffer[:chunk_size_bytes])
                    audio_buffer = audio_buffer[chunk_size_bytes:]

                    # Decode PCM float32 bytes → numpy array
                    num_samples = len(raw_chunk) // 4
                    samples = np.array(
                        struct.unpack(f"<{num_samples}f", raw_chunk),
                        dtype=np.float32,
                    )

                    # Convert to normalised torch tensor
                    audio_tensor = load_audio_from_numpy(
                        samples, stream_config["sample_rate"]
                    )

                    if not validate_audio(audio_tensor, min_duration=0.5):
                        continue

                    # Extract embedding and classify
                    embedding = extract_embedding(audio_tensor)
                    gender, gender_conf = predict_gender(embedding)
                    age_group, age_conf = predict_age(embedding)

                    # Track for final aggregation
                    all_embeddings.append(embedding)
                    all_audio_tensors.append(audio_tensor)

                    # Compute chunk timestamps
                    chunk_duration = get_duration(audio_tensor)
                    ts_start = chunk_index * STREAM_INFERENCE_INTERVAL_SECONDS
                    ts_end = ts_start + chunk_duration

                    # Language detection on the waveform
                    lang_result = language_service.detect_language(
                        audio_tensor.numpy(), sample_rate=16000
                    )

                    # Send per-chunk result
                    chunk_result = StreamingChunk(
                        chunk_index=chunk_index,
                        timestamp_start=round(ts_start, 3),
                        timestamp_end=round(ts_end, 3),
                        gender=gender,
                        gender_confidence=round(gender_conf, 4),
                        age_group=age_group,
                        age_confidence=round(age_conf, 4),
                        language=lang_result["prediction"],
                        language_confidence=round(lang_result["confidence"], 4),
                    )
                    await websocket.send_json(
                        {"type": "chunk", **chunk_result.model_dump()}
                    )
                    chunk_index += 1

    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")
    except Exception as exc:
        logger.error("WebSocket error", exc_info=exc)
        try:
            await websocket.send_json({"type": "error", "message": str(exc)})
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Send final aggregated result
    # ------------------------------------------------------------------
    if all_embeddings:
        final_result = _aggregate_results(all_embeddings, all_audio_tensors)
        try:
            final_msg = StreamingFinalResult(type="final", result=final_result)
            await websocket.send_json(final_msg.model_dump())
        except Exception:
            logger.warning("Could not send final result (connection already closed)")

    logger.info(
        "WebSocket session ended",
        extra={"total_chunks": chunk_index},
    )


def _aggregate_results(
    embeddings: list[np.ndarray],
    audio_tensors: list[torch.Tensor],
) -> AnalysisResult:
    """Aggregate per-chunk results into a single analysis result.

    Strategy:
      - Average all chunk embeddings → classify the mean embedding.
      - Concatenate all audio tensors → compute quality on the full signal.
      - Language detection runs on the concatenated audio.
    """
    # Average embedding
    mean_embedding = np.mean(embeddings, axis=0)
    gender, gender_conf = predict_gender(mean_embedding)
    age_group, age_conf = predict_age(mean_embedding)

    # Concatenate all audio for quality / language analysis
    full_audio = torch.cat(audio_tensors)
    quality = assess_quality(full_audio)
    lang_result = language_service.detect_language(
        full_audio.numpy(), sample_rate=16000
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
