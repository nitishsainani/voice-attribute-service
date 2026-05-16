"""
Configuration constants for voice-attribute-service.

All tuneable values live here as plain module-level constants.
Import individual names where needed:

    from app.config import SAMPLE_RATE, DEVICE
"""

import torch

# ---------------------------------------------------------------------------
# General
# ---------------------------------------------------------------------------
APP_NAME = "voice-attribute-service"

# ---------------------------------------------------------------------------
# Audio
# ---------------------------------------------------------------------------
SAMPLE_RATE = 16000

# ---------------------------------------------------------------------------
# Streaming window settings
#   STREAM_MIN_SECONDS  – minimum audio to buffer before first inference
#   STREAM_MAX_SECONDS  – maximum audio to keep in the rolling buffer
#   STREAM_INFERENCE_INTERVAL_SECONDS – run inference every N new seconds
# ---------------------------------------------------------------------------
STREAM_MIN_SECONDS = 3.0
STREAM_MAX_SECONDS = 5.0
STREAM_INFERENCE_INTERVAL_SECONDS = 1.0

# ---------------------------------------------------------------------------
# SpeechBrain model identifiers (HuggingFace)
# ---------------------------------------------------------------------------
SPEECHBRAIN_SPEAKER_MODEL = "speechbrain/spkrec-ecapa-voxceleb"
SPEECHBRAIN_LANGUAGE_MODEL = "speechbrain/lang-id-voxlingua107-ecapa"

# ---------------------------------------------------------------------------
# Trained classifier paths (joblib)
# ---------------------------------------------------------------------------
GENDER_CLASSIFIER_PATH = "models/gender_classifier.joblib"
AGE_CLASSIFIER_PATH = "models/age_classifier.joblib"

# ---------------------------------------------------------------------------
# Confidence thresholds – predictions below these are reported as "unknown"
# ---------------------------------------------------------------------------
GENDER_CONFIDENCE_THRESHOLD = 0.60
AGE_CONFIDENCE_THRESHOLD = 0.45
LANGUAGE_CONFIDENCE_THRESHOLD = 0.50

# ---------------------------------------------------------------------------
# Feature flags
# ---------------------------------------------------------------------------
ENABLE_LANGUAGE_DETECTION = True
MOCK_MODEL = False

# ---------------------------------------------------------------------------
# Device selection – auto-detect CUDA, fall back to CPU
# ---------------------------------------------------------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
