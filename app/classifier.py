"""
Attribute classifiers for gender and age prediction.

At runtime, these classifiers consume 192-dim ECAPA-TDNN embeddings and
output a label + confidence score.  The classifiers are standard
scikit-learn estimators serialised with joblib.

Training pipeline (offline, run via ``scripts/``):
  1. Extract ECAPA-TDNN embeddings from labelled audio (VoxCeleb, etc.)
  2. Train an SVM (gender) or LogisticRegression (age) on the embeddings
  3. Serialise with ``joblib.dump()`` → save to ``models/``

Runtime pipeline (online, this module):
  1. ``load_classifiers()`` loads the joblib files at startup
  2. ``predict_gender(embedding)`` / ``predict_age(embedding)`` return
     ``(label, confidence)`` tuples

If the model files are missing (e.g., classifiers haven't been trained
yet), the functions gracefully return ``("unknown", 0.0)``.
"""

from pathlib import Path

import joblib
import numpy as np

from app.config import GENDER_CLASSIFIER_PATH, AGE_CLASSIFIER_PATH
from app.logger import logger

# ---------------------------------------------------------------------------
# Global classifier references – populated by ``load_classifiers()``
# ---------------------------------------------------------------------------
_gender_clf = None
_age_clf = None


def load_classifiers() -> None:
    """Load trained gender and age classifiers from disk.

    Called once during application startup.  If a model file is missing,
    a warning is logged and the corresponding classifier remains ``None``
    (predictions will return ``"unknown"``).
    """
    global _gender_clf, _age_clf

    _gender_clf = _try_load(Path(GENDER_CLASSIFIER_PATH), "gender")
    _age_clf = _try_load(Path(AGE_CLASSIFIER_PATH), "age")


def _try_load(path: Path, name: str):
    """Attempt to load a joblib model file, returning None on failure."""
    if not path.exists():
        logger.warning(
            f"{name.title()} classifier not found — predictions will return 'unknown'",
            extra={"path": str(path)},
        )
        return None

    logger.info(f"Loading {name} classifier", extra={"path": str(path)})
    model = joblib.load(path)
    logger.info(f"{name.title()} classifier loaded successfully")
    return model


def is_gender_classifier_loaded() -> bool:
    """Check whether the gender classifier is available."""
    return _gender_clf is not None


def is_age_classifier_loaded() -> bool:
    """Check whether the age classifier is available."""
    return _age_clf is not None


def predict_gender(embedding: np.ndarray) -> tuple[str, float]:
    """Predict gender from a speaker embedding.

    Args:
        embedding: 1-D numpy array of shape ``(192,)``.

    Returns:
        Tuple of ``(label, confidence)`` where label is one of
        ``"male"``, ``"female"``, or ``"unknown"``.
    """
    if _gender_clf is None:
        return ("unknown", 0.0)

    # scikit-learn expects 2-D input: (n_samples, n_features)
    X = embedding.reshape(1, -1)
    label = _gender_clf.predict(X)[0]

    # Extract confidence via predict_proba if available
    confidence = _get_confidence(_gender_clf, X, label)

    return (str(label), float(confidence))


def predict_age(embedding: np.ndarray) -> tuple[str, float]:
    """Predict age group from a speaker embedding.

    Args:
        embedding: 1-D numpy array of shape ``(192,)``.

    Returns:
        Tuple of ``(age_group, confidence)`` where age_group is one of
        ``"child"``, ``"young_adult"``, ``"adult"``, ``"senior"``,
        or ``"unknown"``.
    """
    if _age_clf is None:
        return ("unknown", 0.0)

    X = embedding.reshape(1, -1)
    label = _age_clf.predict(X)[0]
    confidence = _get_confidence(_age_clf, X, label)

    return (str(label), float(confidence))


def _get_confidence(clf, X: np.ndarray, predicted_label) -> float:
    """Extract confidence score from a classifier if it supports probabilities.

    Falls back to decision function or 1.0 if neither is available.
    """
    if hasattr(clf, "predict_proba"):
        proba = clf.predict_proba(X)[0]
        # Find the index of the predicted class
        class_idx = list(clf.classes_).index(predicted_label)
        return float(proba[class_idx])
    elif hasattr(clf, "decision_function"):
        # For SVM without probability=True, use decision function magnitude
        decision = clf.decision_function(X)[0]
        # Normalise to 0–1 range using sigmoid-like mapping
        confidence = 1.0 / (1.0 + np.exp(-abs(float(decision))))
        return confidence
    else:
        return 1.0
