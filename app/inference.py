"""
SpeechBrain embedding extraction.

Uses the ECAPA-TDNN model (``speechbrain/spkrec-ecapa-voxceleb``) to extract
192-dimensional speaker embeddings from audio.  These embeddings capture
speaker-level characteristics (voice timbre, pitch patterns, etc.) and are
used downstream by the gender and age classifiers.

Architecture overview:
    Raw audio (16 kHz mono)
        → SpeechBrain EncoderClassifier.encode_batch()
        → 192-dim embedding vector (torch.Tensor)
        → numpy array for scikit-learn classifiers

The model is loaded once at module level (singleton pattern) to avoid
re-downloading and re-initializing on every request.
"""

import torch
import numpy as np

from app.config import SPEECHBRAIN_SPEAKER_MODEL, DEVICE
from app.logger import logger

# ---------------------------------------------------------------------------
# Global model reference – populated by ``load_model()``
# ---------------------------------------------------------------------------
_encoder = None


def load_model() -> None:
    """Download (if needed) and load the SpeechBrain ECAPA-TDNN model.

    This should be called once during application startup (via FastAPI's
    lifespan hook).  The model is stored in a module-level variable so it
    can be reused across requests without reloading.
    """
    global _encoder

    # Lazy import to avoid slow import at module parse time
    from speechbrain.inference.speaker import EncoderClassifier

    _savedir = "pretrained_models/spkrec-ecapa-voxceleb"
    logger.info(
        "Loading SpeechBrain ECAPA-TDNN model",
        extra={
            "source": SPEECHBRAIN_SPEAKER_MODEL,
            "cache_dir": _savedir,
            "device": DEVICE,
        },
    )

    _encoder = EncoderClassifier.from_hparams(
        source=SPEECHBRAIN_SPEAKER_MODEL,
        savedir=_savedir,
        run_opts={"device": DEVICE},
    )

    logger.info("SpeechBrain model loaded successfully")


def is_model_loaded() -> bool:
    """Check whether the embedding model has been loaded."""
    return _encoder is not None


def extract_embedding(audio_tensor: torch.Tensor) -> np.ndarray:
    """Extract a speaker embedding from an audio tensor.

    Args:
        audio_tensor: 1-D float32 tensor at 16 kHz.

    Returns:
        1-D numpy array of shape ``(192,)`` — the ECAPA-TDNN embedding.

    Raises:
        RuntimeError: If the model has not been loaded yet.
    """
    if _encoder is None:
        raise RuntimeError(
            "Embedding model not loaded. Call load_model() during startup."
        )

    # EncoderClassifier.encode_batch expects shape (batch, time)
    waveform = audio_tensor.unsqueeze(0)

    with torch.no_grad():
        embedding = _encoder.encode_batch(waveform)

    # embedding shape: (1, 1, 192) → squeeze to (192,)
    return embedding.squeeze().cpu().numpy()
