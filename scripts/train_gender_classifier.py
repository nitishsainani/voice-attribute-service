"""
Train gender classifier on ECAPA-TDNN embeddings.

Loads pre-extracted embeddings from ``data/embeddings/gender_embeddings.npz``
and trains a scikit-learn SVM classifier.  The trained model is saved to
``models/gender_classifier.joblib``.

Pipeline:
  1. Load embeddings + labels from .npz
  2. Train/test split (stratified)
  3. Fit SVM with RBF kernel (+ probability estimates for confidence)
  4. Evaluate on test set (classification report)
  5. Save with joblib

Usage:
    uv run python scripts/train_gender_classifier.py
    uv run python scripts/train_gender_classifier.py --embeddings data/embeddings/gender_embeddings.npz
"""

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a gender classifier on ECAPA-TDNN embeddings."
    )
    parser.add_argument(
        "--embeddings",
        type=Path,
        default=Path("data/embeddings/gender_embeddings.npz"),
        help="Path to the .npz file with embeddings and labels.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("models/gender_classifier.joblib"),
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
    # StandardScaler normalises embeddings to zero mean / unit variance,
    # which improves SVM convergence and accuracy.
    # probability=True enables predict_proba() for confidence scores.
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("svm", SVC(
            kernel="rbf",
            C=10.0,
            gamma="scale",
            probability=True,
            random_state=args.random_state,
        )),
    ])

    print("Training SVM classifier...")
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
