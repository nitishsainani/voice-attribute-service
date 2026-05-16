"""
Train age-group classifier on ECAPA-TDNN embeddings.

Loads pre-extracted embeddings from ``data/embeddings/age_embeddings.npz``
and trains a scikit-learn LogisticRegression classifier.  Logistic
Regression is preferred over SVM for the age task because it naturally
supports multi-class probability estimates and trains faster on the
typically larger age dataset.

Age groups: child, young_adult, adult, senior

Pipeline:
  1. Load embeddings + labels from .npz
  2. Train/test split (stratified)
  3. Fit LogisticRegression with L2 regularisation
  4. Evaluate on test set (classification report)
  5. Save with joblib

Usage:
    uv run python scripts/train_age_classifier.py
"""

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train an age-group classifier on ECAPA-TDNN embeddings."
    )
    parser.add_argument(
        "--embeddings",
        type=Path,
        default=Path("data/embeddings/age_embeddings.npz"),
        help="Path to the .npz file with embeddings and labels.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("models/age_classifier.joblib"),
        help="Output path for the trained model.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Fraction of data to use for testing.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.embeddings.exists():
        print(f"ERROR: Embeddings file not found: {args.embeddings}")
        print("Run scripts/extract_embeddings.py first.")
        sys.exit(1)

    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------
    print(f"Loading embeddings from {args.embeddings}")
    data = np.load(args.embeddings, allow_pickle=True)
    X = data["embeddings"]
    y = data["labels"]

    print(f"  Samples: {len(X)}")
    print(f"  Feature dim: {X.shape[1]}")
    for label in np.unique(y):
        print(f"  {label}: {(y == label).sum()}")

    # ------------------------------------------------------------------
    # Train/test split
    # ------------------------------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=y,
    )

    print(f"\nTrain: {len(X_train)}, Test: {len(X_test)}")

    # ------------------------------------------------------------------
    # Build and train pipeline
    # ------------------------------------------------------------------
    # LogisticRegression with lbfgs solver uses softmax (multinomial)
    # for proper probability calibration across all age groups.
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(
            C=1.0,
            max_iter=1000,
            solver="lbfgs",
            random_state=args.random_state,
        )),
    ])

    print("Training LogisticRegression classifier...")
    pipeline.fit(X_train, y_train)

    # ------------------------------------------------------------------
    # Evaluate
    # ------------------------------------------------------------------
    y_pred = pipeline.predict(X_test)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    accuracy = (y_pred == y_test).mean()
    print(f"Test Accuracy: {accuracy:.4f}")

    # ------------------------------------------------------------------
    # Save model
    # ------------------------------------------------------------------
    args.output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, args.output)
    print(f"\nModel saved to {args.output}")


if __name__ == "__main__":
    main()
