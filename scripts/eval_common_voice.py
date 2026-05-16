"""
Evaluate trained classifiers on Mozilla Common Voice.

This script provides an independent evaluation of the gender and age
classifiers on the Mozilla Common Voice dataset, which is a different
domain from VoxCeleb (read speech vs. celebrity interviews).

Expected input:
  - Common Voice TSV metadata (e.g., ``validated.tsv``) with columns:
    ``path``, ``gender``, ``age``
  - Corresponding audio clips in ``data/raw/common_voice/clips/``

Usage:
    uv run python scripts/eval_common_voice.py \\
        --tsv data/raw/common_voice/validated.tsv \\
        --clips-dir data/raw/common_voice/clips \\
        --max-samples 500
"""

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torchaudio
from sklearn.metrics import classification_report
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Common Voice age ranges → our age groups
# ---------------------------------------------------------------------------
_CV_AGE_MAP = {
    "teens": "child",
    "twenties": "young_adult",
    "thirties": "adult",
    "fourties": "adult",
    "fifties": "adult",
    "sixties": "senior",
    "seventies": "senior",
    "eighties": "senior",
    "nineties": "senior",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate classifiers on Mozilla Common Voice."
    )
    parser.add_argument(
        "--tsv",
        type=Path,
        required=True,
        help="Path to Common Voice validated.tsv file.",
    )
    parser.add_argument(
        "--clips-dir",
        type=Path,
        required=True,
        help="Directory containing Common Voice audio clips.",
    )
    parser.add_argument(
        "--gender-model",
        type=Path,
        default=Path("models/gender_classifier.joblib"),
        help="Path to the trained gender classifier.",
    )
    parser.add_argument(
        "--age-model",
        type=Path,
        default=Path("models/age_classifier.joblib"),
        help="Path to the trained age classifier.",
    )
    parser.add_argument(
        "--sb-model",
        type=str,
        default="speechbrain/spkrec-ecapa-voxceleb",
        help="SpeechBrain model identifier.",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("pretrained_models/spkrec-ecapa-voxceleb"),
        help="SpeechBrain model cache directory.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=500,
        help="Maximum samples to evaluate (per-class balanced).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.tsv.exists():
        print(f"ERROR: TSV file not found: {args.tsv}")
        sys.exit(1)

    # ------------------------------------------------------------------
    # Load Common Voice metadata
    # ------------------------------------------------------------------
    print(f"Loading metadata from {args.tsv}")
    df = pd.read_csv(args.tsv, sep="\t")

    # Filter to rows with gender info
    if "gender" in df.columns:
        df = df.dropna(subset=["gender"])
        df["gender"] = df["gender"].str.strip().str.lower()
        df = df[df["gender"].isin(["male", "female"])]
    else:
        print("WARNING: No 'gender' column found in TSV.")

    # Map age ranges if available
    if "age" in df.columns:
        df = df.dropna(subset=["age"])
        df["age_group"] = df["age"].str.strip().str.lower().map(_CV_AGE_MAP)
        df = df.dropna(subset=["age_group"])

    # Balance classes and limit samples
    if args.max_samples and "gender" in df.columns:
        df = (
            df.groupby("gender")
            .apply(lambda g: g.sample(
                n=min(len(g), args.max_samples // 2),
                random_state=42,
            ))
            .reset_index(drop=True)
        )

    print(f"Evaluating on {len(df)} samples")

    # ------------------------------------------------------------------
    # Load models
    # ------------------------------------------------------------------
    from speechbrain.inference.speaker import EncoderClassifier

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading ECAPA-TDNN model on {device}...")
    encoder = EncoderClassifier.from_hparams(
        source=args.sb_model,
        savedir=str(args.cache_dir),
        run_opts={"device": device},
    )

    gender_clf = None
    if args.gender_model.exists():
        gender_clf = joblib.load(args.gender_model)
        print(f"Loaded gender classifier from {args.gender_model}")

    age_clf = None
    if args.age_model.exists():
        age_clf = joblib.load(args.age_model)
        print(f"Loaded age classifier from {args.age_model}")

    # ------------------------------------------------------------------
    # Extract embeddings and predict
    # ------------------------------------------------------------------
    gender_true, gender_pred = [], []
    age_true, age_pred = [], []
    skipped = 0

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Evaluating"):
        clip_path = args.clips_dir / row["path"]
        if not clip_path.exists():
            # Try with .mp3 extension
            clip_path = clip_path.with_suffix(".mp3")
        if not clip_path.exists():
            skipped += 1
            continue

        try:
            # Load audio with soundfile (avoids ffmpeg dependency)
            data, sr = sf.read(str(clip_path), dtype="float32")
            waveform = torch.from_numpy(data).float()

            if waveform.ndim == 2:
                waveform = waveform.mean(dim=1)

            waveform = waveform.unsqueeze(0)

            if sr != 16000:
                waveform = torchaudio.transforms.Resample(sr, 16000)(waveform)

            with torch.no_grad():
                emb = encoder.encode_batch(waveform).squeeze().cpu().numpy()

            # Gender prediction
            if gender_clf and "gender" in df.columns:
                pred = gender_clf.predict(emb.reshape(1, -1))[0]
                gender_true.append(row["gender"])
                gender_pred.append(pred)

            # Age prediction
            if age_clf and "age_group" in df.columns:
                pred = age_clf.predict(emb.reshape(1, -1))[0]
                age_true.append(row["age_group"])
                age_pred.append(pred)

        except Exception as exc:
            print(f"WARNING: Skipping {clip_path}: {exc}")
            skipped += 1

    # ------------------------------------------------------------------
    # Print results
    # ------------------------------------------------------------------
    print(f"\nSkipped {skipped} files\n")

    if gender_true:
        print("=" * 50)
        print("GENDER CLASSIFICATION REPORT")
        print("=" * 50)
        print(classification_report(gender_true, gender_pred))

    if age_true:
        print("=" * 50)
        print("AGE GROUP CLASSIFICATION REPORT")
        print("=" * 50)
        print(classification_report(age_true, age_pred))

    if not gender_true and not age_true:
        print("No predictions were made. Check your data and model paths.")


if __name__ == "__main__":
    main()
