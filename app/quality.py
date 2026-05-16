"""
Audio quality analysis utilities.

Computes lightweight signal-quality metrics that help downstream consumers
decide whether an analysis result is reliable.  Poor-quality audio (noisy,
clipped, too short) generally leads to less accurate gender/age predictions.

Metrics:
  - **SNR (Signal-to-Noise Ratio)**: Estimated by comparing the energy of
    the top-percentile samples (signal) to the bottom-percentile (noise).
  - **Clipping ratio**: Fraction of samples near the ±1.0 boundary.
  - **Duration**: Total length of the audio in seconds.
"""

import torch
import numpy as np

from app.config import SAMPLE_RATE, STREAM_MIN_SECONDS


# ---------------------------------------------------------------------------
# Thresholds – audio is considered "acceptable" when all are met
# ---------------------------------------------------------------------------
_MIN_SNR_DB = 10.0  # Minimum signal-to-noise ratio
_MAX_CLIPPING_RATIO = 0.01  # At most 1% of samples clipped
_CLIPPING_THRESHOLD = 0.99  # Absolute sample value considered "clipped"


def compute_snr(audio_tensor: torch.Tensor) -> float:
    """Estimate signal-to-noise ratio (dB) of the audio.

    Uses a simple energy-based heuristic: the ratio of the RMS energy
    of the loudest 10% of frames to the quietest 10% of frames.

    Args:
        audio_tensor: 1-D float32 tensor at 16 kHz.

    Returns:
        Estimated SNR in decibels.  Returns 0.0 for silence.
    """
    samples = audio_tensor.abs().numpy()

    if samples.max() == 0:
        return 0.0

    # Sort absolute sample values
    sorted_vals = np.sort(samples)
    n = len(sorted_vals)

    # Noise estimate: RMS of the quietest 10%
    noise_segment = sorted_vals[: max(n // 10, 1)]
    noise_rms = np.sqrt(np.mean(noise_segment ** 2)) + 1e-10

    # Signal estimate: RMS of the loudest 10%
    signal_segment = sorted_vals[-max(n // 10, 1) :]
    signal_rms = np.sqrt(np.mean(signal_segment ** 2)) + 1e-10

    snr_db = 20.0 * np.log10(signal_rms / noise_rms)
    return float(snr_db)


def detect_clipping(audio_tensor: torch.Tensor) -> float:
    """Compute the fraction of samples that appear clipped.

    Clipped samples are those whose absolute value exceeds the
    ``_CLIPPING_THRESHOLD`` (default 0.99).

    Args:
        audio_tensor: 1-D float32 tensor.

    Returns:
        Clipping ratio in range [0.0, 1.0].
    """
    if audio_tensor.numel() == 0:
        return 0.0

    clipped = (audio_tensor.abs() >= _CLIPPING_THRESHOLD).sum().item()
    return clipped / audio_tensor.numel()


def assess_quality(audio_tensor: torch.Tensor) -> dict:
    """Run all quality checks and return a combined report.

    Args:
        audio_tensor: 1-D float32 tensor at 16 kHz.

    Returns:
        Dictionary matching the ``QualityMetrics`` schema fields.
    """
    snr = compute_snr(audio_tensor)
    clipping = detect_clipping(audio_tensor)
    duration = audio_tensor.shape[0] / SAMPLE_RATE

    is_acceptable = (
        snr >= _MIN_SNR_DB
        and clipping <= _MAX_CLIPPING_RATIO
        and duration >= STREAM_MIN_SECONDS
    )

    return {
        "snr_db": round(snr, 2),
        "clipping_ratio": round(clipping, 4),
        "duration_seconds": round(duration, 3),
        "is_acceptable": is_acceptable,
    }
