"""
Audio preprocessing utilities.

Handles decoding, resampling, validation, and chunking of audio data.
All audio flowing through the system is normalised to:
  - 16 kHz sample rate
  - mono channel
  - float32 torch.Tensor

This module is used by both the REST endpoint (full file upload) and the
WebSocket streaming handler (incremental chunks).
"""

import io
from typing import Iterator

import numpy as np
import soundfile as sf
import torch
import torchaudio

from app.config import SAMPLE_RATE, STREAM_MIN_SECONDS, STREAM_INFERENCE_INTERVAL_SECONDS
from app.logger import logger


def load_audio(file_bytes: bytes) -> torch.Tensor:
    """Decode raw audio bytes into a normalised 16 kHz mono tensor.

    Supports any format that libsndfile can decode (WAV, FLAC, OGG, etc.).

    Args:
        file_bytes: Raw audio file bytes.

    Returns:
        1-D float32 tensor of shape ``(num_samples,)`` at 16 kHz.

    Raises:
        ValueError: If the audio cannot be decoded.
    """
    try:
        # soundfile handles format detection automatically
        data, sample_rate = sf.read(io.BytesIO(file_bytes), dtype="float32")
    except Exception as exc:
        raise ValueError(f"Failed to decode audio: {exc}") from exc

    # Convert to torch tensor – soundfile returns (samples,) for mono
    # or (samples, channels) for multi-channel audio.
    waveform = torch.from_numpy(data).float()

    # If stereo / multi-channel, mix down to mono
    if waveform.ndim == 2:
        waveform = waveform.mean(dim=1)

    # Ensure shape is (1, num_samples) for torchaudio resample
    waveform = waveform.unsqueeze(0)

    # Resample to target sample rate if necessary
    if sample_rate != SAMPLE_RATE:
        logger.info(
            "Resampling audio",
            extra={"from_sr": sample_rate, "to_sr": SAMPLE_RATE},
        )
        resampler = torchaudio.transforms.Resample(
            orig_freq=sample_rate, new_freq=SAMPLE_RATE
        )
        waveform = resampler(waveform)

    # Return as 1-D tensor (remove batch dim)
    return waveform.squeeze(0)


def load_audio_from_numpy(audio_array: np.ndarray, sample_rate: int) -> torch.Tensor:
    """Convert a numpy audio array to a normalised 16 kHz mono tensor.

    Useful for processing raw PCM chunks received over WebSocket.

    Args:
        audio_array: Numpy array of audio samples (float32).
        sample_rate: Original sample rate of the audio.

    Returns:
        1-D float32 tensor at 16 kHz.
    """
    waveform = torch.from_numpy(audio_array).float()

    if waveform.ndim == 2:
        waveform = waveform.mean(dim=1)

    waveform = waveform.unsqueeze(0)

    if sample_rate != SAMPLE_RATE:
        resampler = torchaudio.transforms.Resample(
            orig_freq=sample_rate, new_freq=SAMPLE_RATE
        )
        waveform = resampler(waveform)

    return waveform.squeeze(0)


def validate_audio(audio_tensor: torch.Tensor, min_duration: float | None = None) -> bool:
    """Check whether the audio meets minimum duration requirements.

    Args:
        audio_tensor: 1-D tensor at 16 kHz.
        min_duration: Override minimum duration in seconds.
                      Defaults to ``STREAM_MIN_SECONDS``.

    Returns:
        True if the audio is long enough for reliable analysis.
    """
    if min_duration is None:
        min_duration = STREAM_MIN_SECONDS
    duration_s = audio_tensor.shape[0] / SAMPLE_RATE
    return duration_s >= min_duration


def get_duration(audio_tensor: torch.Tensor) -> float:
    """Return the duration of the audio tensor in seconds."""
    return audio_tensor.shape[0] / SAMPLE_RATE


def chunk_stream(
    audio_tensor: torch.Tensor,
    chunk_seconds: float | None = None,
) -> Iterator[torch.Tensor]:
    """Yield fixed-duration chunks from a long audio tensor.

    This is used by the streaming pipeline to process audio incrementally
    as it arrives over the WebSocket connection.

    Args:
        audio_tensor: 1-D tensor at 16 kHz.
        chunk_seconds: Duration of each chunk in seconds.
                       Defaults to ``STREAM_INFERENCE_INTERVAL_SECONDS``.

    Yields:
        1-D tensors of approximately ``chunk_seconds`` duration each.
        The final chunk may be shorter.
    """
    if chunk_seconds is None:
        chunk_seconds = STREAM_INFERENCE_INTERVAL_SECONDS

    chunk_size = int(chunk_seconds * SAMPLE_RATE)
    total_samples = audio_tensor.shape[0]

    for start in range(0, total_samples, chunk_size):
        yield audio_tensor[start : start + chunk_size]
