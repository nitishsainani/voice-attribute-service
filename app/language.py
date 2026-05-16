"""
Language detection using SpeechBrain VoxLingua107 ECAPA model.

Uses the pretrained ``speechbrain/lang-id-voxlingua107-ecapa`` model to
identify the spoken language from a preprocessed audio waveform.  The model
supports 107 languages and is downloaded/cached automatically by SpeechBrain
on first run.

Language detection is **separate** from the gender/age pipeline:
  - Gender/age uses speaker embeddings + trained scikit-learn classifiers.
  - Language detection directly uses the audio waveform via SpeechBrain's
    ``classify_batch`` method — no custom training required.

Configuration:
  - ``ENABLE_LANGUAGE_DETECTION`` — toggle detection on/off (default: True)
  - ``MOCK_MODEL`` — return a fixed ``{"prediction": "en", "confidence": 0.99}``
    response for testing without loading the real model
  - ``LANGUAGE_CONFIDENCE_THRESHOLD`` — predictions below this confidence
    are reported as ``"unknown"``
"""

import numpy as np
import torch

from app.config import (
    DEVICE,
    ENABLE_LANGUAGE_DETECTION,
    LANGUAGE_CONFIDENCE_THRESHOLD,
    MOCK_MODEL,
    SPEECHBRAIN_LANGUAGE_MODEL,
)
from app.logger import logger


class LanguageDetectionService:
    """Singleton service for spoken-language identification.

    The SpeechBrain model is loaded once in ``__init__`` and reused for
    all subsequent requests.  Call ``load_model()`` during application
    startup to trigger the download (if needed) and initialisation.

    Example::

        lang_service = LanguageDetectionService()
        lang_service.load_model()

        result = lang_service.detect_language(waveform, sample_rate=16000)
        # {"prediction": "en", "confidence": 0.87}
    """

    def __init__(self) -> None:
        self._model = None
        self._enabled = ENABLE_LANGUAGE_DETECTION
        self._mock = MOCK_MODEL
        self._threshold = LANGUAGE_CONFIDENCE_THRESHOLD
        self._device = DEVICE

    # ------------------------------------------------------------------
    # Model lifecycle
    # ------------------------------------------------------------------
    def load_model(self) -> None:
        """Download (if needed) and load the VoxLingua107 language-ID model.

        When ``MOCK_MODEL`` is True or ``ENABLE_LANGUAGE_DETECTION`` is
        False, the heavy model is **not** loaded — the service will return
        placeholder responses instead.
        """
        if self._mock or not self._enabled:
            logger.info(
                "Language detection model not loaded "
                f"(enabled={self._enabled}, mock={self._mock})"
            )
            return

        from speechbrain.inference.classifiers import EncoderClassifier

        savedir = "pretrained_models/lang-id-voxlingua107-ecapa"
        logger.info(
            "Loading SpeechBrain language-ID model",
            extra={
                "source": SPEECHBRAIN_LANGUAGE_MODEL,
                "cache_dir": savedir,
                "device": self._device,
            },
        )

        self._model = EncoderClassifier.from_hparams(
            source=SPEECHBRAIN_LANGUAGE_MODEL,
            savedir=savedir,
            run_opts={"device": self._device},
        )

        logger.info("Language-ID model loaded successfully")

    def is_loaded(self) -> bool:
        """Check whether the language model has been loaded."""
        return self._model is not None

    # ------------------------------------------------------------------
    # Detection
    # ------------------------------------------------------------------
    def detect_language(
        self, waveform: np.ndarray, sample_rate: int = 16000
    ) -> dict:
        """Detect the spoken language of an audio waveform.

        Args:
            waveform: 1-D float32 numpy array at ``sample_rate`` Hz
                      (should already be mono 16 kHz).
            sample_rate: Sample rate of the waveform (expected 16000).

        Returns:
            Dictionary with keys:
              - ``prediction``: ISO 639-1 language code (e.g. ``"en"``)
                or ``"unknown"``
              - ``confidence``: float in [0, 1]
        """
        _unknown = {"prediction": "unknown", "confidence": 0.0}

        # ----- Guard: disabled -----------------------------------------
        if not self._enabled:
            return _unknown

        # ----- Guard: mock mode ----------------------------------------
        if self._mock:
            return {"prediction": "en", "confidence": 0.99}

        # ----- Guard: empty waveform -----------------------------------
        if waveform is None or len(waveform) == 0:
            return _unknown

        # ----- Guard: model not loaded ---------------------------------
        if self._model is None:
            logger.warning("Language model not loaded — returning unknown")
            return _unknown

        try:
            # Convert to torch tensor with shape [1, time]
            tensor = torch.from_numpy(waveform).float().unsqueeze(0)

            # classify_batch returns (out_prob, score, index, text_lab)
            # Note: score is in log-softmax space — convert to probability.
            out_prob, score, index, text_lab = self._model.classify_batch(
                tensor
            )

            prediction = text_lab[0]       # top-1 language label
            # Convert log-softmax score to probability and clamp to [0, 1]
            confidence = float(
                torch.clamp(score[0].exp(), min=0.0, max=1.0).item()
            )

            if confidence < self._threshold:
                return {"prediction": "unknown", "confidence": round(confidence, 4)}

            return {
                "prediction": prediction,
                "confidence": round(confidence, 4),
            }

        except Exception as exc:
            # Language detection is best-effort — never break gender/age
            logger.warning("Language detection failed", exc_info=exc)
            return _unknown


# ---------------------------------------------------------------------------
# Module-level singleton — wired into the app via ``main.py`` lifespan
# ---------------------------------------------------------------------------
language_service = LanguageDetectionService()
