"""
End-to-end training orchestrator.

Runs the full training pipeline in sequence:
  1. Prepare VoxCeleb gender manifest
  2. Prepare AgeVoxCeleb age manifest
  3. Extract ECAPA-TDNN embeddings for gender
  4. Extract ECAPA-TDNN embeddings for age
  5. Train gender classifier (SVM)
  6. Train age classifier (LogisticRegression)

Usage:
    uv run python scripts/train_all.py

Each step is run as a subprocess so that failures are isolated and the
script reports which step failed.
"""

import subprocess
import sys
import time


# Each step: (description, command)
STEPS = [
    (
        "Step 1/6: Prepare VoxCeleb gender manifest",
        [
            sys.executable, "scripts/prepare_voxceleb_gender.py",
            "--meta", "data/raw/vox1_meta.csv",
            "--audio-dir", "data/raw/voxceleb1/wav",
            "--output", "data/processed/gender_manifest.csv",
        ],
    ),
    (
        "Step 2/6: Prepare AgeVoxCeleb age manifest",
        [
            sys.executable, "scripts/prepare_agevoxceleb.py",
            "--meta", "data/raw/agevoxceleb_meta.csv",
            "--audio-dir", "data/raw/voxceleb1/wav",
            "--output", "data/processed/age_manifest.csv",
        ],
    ),
    (
        "Step 3/6: Extract gender embeddings",
        [
            sys.executable, "scripts/extract_embeddings.py",
            "--manifest", "data/processed/gender_manifest.csv",
            "--label-col", "gender",
            "--output", "data/embeddings/gender_embeddings.npz",
        ],
    ),
    (
        "Step 4/6: Extract age embeddings",
        [
            sys.executable, "scripts/extract_embeddings.py",
            "--manifest", "data/processed/age_manifest.csv",
            "--label-col", "age_group",
            "--output", "data/embeddings/age_embeddings.npz",
        ],
    ),
    (
        "Step 5/6: Train gender classifier",
        [
            sys.executable, "scripts/train_gender_classifier.py",
            "--embeddings", "data/embeddings/gender_embeddings.npz",
            "--output", "models/gender_classifier.joblib",
        ],
    ),
    (
        "Step 6/6: Train age classifier",
        [
            sys.executable, "scripts/train_age_classifier.py",
            "--embeddings", "data/embeddings/age_embeddings.npz",
            "--output", "models/age_classifier.joblib",
        ],
    ),
]


def main() -> None:
    print("=" * 60)
    print("Voice Attribute Service — Full Training Pipeline")
    print("=" * 60)

    overall_start = time.time()

    for description, command in STEPS:
        print(f"\n{'─' * 60}")
        print(f"▶ {description}")
        print(f"  Command: {' '.join(command)}")
        print(f"{'─' * 60}")

        step_start = time.time()

        result = subprocess.run(
            command,
            capture_output=False,  # Show output in real time
        )

        elapsed = time.time() - step_start

        if result.returncode != 0:
            print(f"\n✗ FAILED: {description} (exit code {result.returncode})")
            print("Aborting pipeline.")
            sys.exit(result.returncode)

        print(f"✓ Completed in {elapsed:.1f}s")

    total_elapsed = time.time() - overall_start
    print(f"\n{'=' * 60}")
    print(f"✓ All steps completed successfully in {total_elapsed:.1f}s")
    print(f"{'=' * 60}")
    print("\nTrained models saved to:")
    print("  models/gender_classifier.joblib")
    print("  models/age_classifier.joblib")


if __name__ == "__main__":
    main()
