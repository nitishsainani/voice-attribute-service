"""
Batch-extract SpeechBrain ECAPA-TDNN embeddings from audio manifests.

This script reads a manifest CSV (produced by ``prepare_voxceleb_gender.py``
or ``prepare_agevoxceleb.py``) and extracts a 192-dimensional speaker
embedding for each audio file using the pretrained ECAPA-TDNN model.

The embeddings are saved as a single ``.npz`` file containing:
  - ``embeddings``: numpy array of shape ``(N, 192)``
  - ``labels``: list of label strings (gender or age_group)
  - ``speaker_ids``: list of speaker ID strings

Usage:
    uv run python scripts/extract_embeddings.py \\
        --manifest data/processed/gender_manifest.csv \\
        --label-col gender \\
        --output data/embeddings/gender_embeddings.npz

    uv run python scripts/extract_embeddings.py \\
        --manifest data/processed/age_manifest.csv \\
        --label-col age_group \\
        --output data/embeddings/age_embeddings.npz
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torchaudio
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract ECAPA-TDNN embeddings from an audio manifest."
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        required=True,
        help="Path to the manifest CSV (must have 'audio_path' column).",
    )
    parser.add_argument(
        "--label-col",
        type=str,
        default="gender",
        help="Column name to use as the label (e.g., 'gender', 'age_group').",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output .npz file path for the extracted embeddings.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="speechbrain/spkrec-ecapa-voxceleb",
        help="SpeechBrain model identifier on HuggingFace.",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("pretrained_models/spkrec-ecapa-voxceleb"),
        help="Local cache directory for the SpeechBrain model.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum number of samples to process (for debugging).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.manifest.exists():
        print(f"ERROR: Manifest not found: {args.manifest}")
        sys.exit(1)

    # ------------------------------------------------------------------
    # Load manifest
    # ------------------------------------------------------------------
    df = pd.read_csv(args.manifest)
    if args.label_col not in df.columns:
        print(f"ERROR: Column '{args.label_col}' not found in manifest.")
        print(f"Available columns: {list(df.columns)}")
        sys.exit(1)

    if args.max_samples:
        df = df.head(args.max_samples)

    print(f"Processing {len(df)} audio files from {args.manifest}")

    # ------------------------------------------------------------------
    # Load SpeechBrain model
    # ------------------------------------------------------------------
    from speechbrain.inference.speaker import EncoderClassifier

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading ECAPA-TDNN model on {device}...")

    encoder = EncoderClassifier.from_hparams(
        source=args.model,
        savedir=str(args.cache_dir),
        run_opts={"device": device},
    )

    # ------------------------------------------------------------------
    # Extract embeddings
    # ------------------------------------------------------------------
    embeddings = []
    labels = []
    speaker_ids = []
    skipped = 0

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Extracting embeddings"):
        audio_path = Path(row["audio_path"])
        if not audio_path.exists():
            skipped += 1
            continue

        try:
            # Load audio with soundfile (avoids ffmpeg dependency)
            data, sr = sf.read(str(audio_path), dtype="float32")
            waveform = torch.from_numpy(data).float()

            # Mix to mono if needed (soundfile returns (samples, channels))
            if waveform.ndim == 2:
                waveform = waveform.mean(dim=1)

            # Add channel dimension: (1, num_samples)
            waveform = waveform.unsqueeze(0)

            # Resample to 16 kHz if necessary
            if sr != 16000:
                resampler = torchaudio.transforms.Resample(sr, 16000)
                waveform = resampler(waveform)

            # Extract embedding
            with torch.no_grad():
                emb = encoder.encode_batch(waveform)

            embeddings.append(emb.squeeze().cpu().numpy())
            labels.append(str(row[args.label_col]))
            speaker_ids.append(str(row.get("speaker_id", "")))

        except Exception as exc:
            print(f"WARNING: Skipping {audio_path}: {exc}")
            skipped += 1

    # ------------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------------
    if not embeddings:
        print("ERROR: No embeddings were extracted!")
        sys.exit(1)

    args.output.parent.mkdir(parents=True, exist_ok=True)

    np.savez(
        args.output,
        embeddings=np.array(embeddings),
        labels=np.array(labels),
        speaker_ids=np.array(speaker_ids),
    )

    print(f"\nDone! Saved {len(embeddings)} embeddings to {args.output}")
    print(f"Skipped {skipped} files")
    print(f"Embedding shape: {np.array(embeddings).shape}")


if __name__ == "__main__":
    main()
