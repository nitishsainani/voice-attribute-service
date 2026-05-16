# Models Directory

This directory stores trained scikit-learn classifiers serialised with `joblib`.

## Expected Files

| File                       | Description                                      | Training Script                     |
|----------------------------|--------------------------------------------------|--------------------------------------|
| `gender_classifier.joblib` | SVM classifier for male/female prediction        | `scripts/train_gender_classifier.py` |
| `age_classifier.joblib`    | LogisticRegression for age group prediction       | `scripts/train_age_classifier.py`    |

## How to Train

Run the full training pipeline:

```bash
uv run python scripts/train_all.py
```

Or train individual classifiers:

```bash
# Gender classifier
uv run python scripts/train_gender_classifier.py \
    --embeddings data/embeddings/gender_embeddings.npz \
    --output models/gender_classifier.joblib

# Age classifier
uv run python scripts/train_age_classifier.py \
    --embeddings data/embeddings/age_embeddings.npz \
    --output models/age_classifier.joblib
```

## Notes

- The service will start without these files, but predictions will return `"unknown"`.
- Models are excluded from Git via `.gitignore` — share them via artifact storage.
- Typical model sizes: ~1–5 MB each (small, since they're sklearn pipelines over 192-dim embeddings).
