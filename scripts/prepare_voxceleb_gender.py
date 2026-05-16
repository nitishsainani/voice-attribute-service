"""
Prepare VoxCeleb gender metadata.

This script parses the VoxCeleb1/VoxCeleb2 metadata files and produces
a cleaned manifest CSV at ``data/processed/gender_manifest.csv``.

Expected input: VoxCeleb metadata file (``vox1_meta.csv`` or similar)
located in ``data/raw/``.  The file should contain columns:
  VoxCeleb1 ID | Name | Gender | Nationality | Set

Output CSV columns:
  speaker_id, audio_path, gender

Usage:
    uv run python scripts/prepare_voxceleb_gender.py \\
        --meta data/raw/vox1_meta.csv \\
        --audio-dir data/raw/voxceleb1/wav \\
        --output data/processed/gender_manifest.csv
"""

import argparse
import csv
import sys
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare VoxCeleb gender manifest for training."
    )
    parser.add_argument(
        "--meta",
        type=Path,
        default=Path("data/raw/vox1_meta.csv"),
        help="Path to VoxCeleb metadata CSV.",
    )
    parser.add_argument(
        "--audio-dir",
        type=Path,
        default=Path("data/raw/voxceleb1/wav"),
        help="Root directory containing VoxCeleb audio files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/gender_manifest.csv"),
        help="Output manifest CSV path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.meta.exists():
        print(f"ERROR: Metadata file not found: {args.meta}")
        print(
            "Download VoxCeleb metadata from "
            "https://mm.kaist.ac.kr/datasets/voxceleb/ "
            "and place it in data/raw/"
        )
        sys.exit(1)

    # ------------------------------------------------------------------
    # Parse the VoxCeleb metadata
    # ------------------------------------------------------------------
    print(f"Reading metadata from {args.meta}")
    df = pd.read_csv(
        args.meta,
        sep="\t",
        header=0,
        names=["speaker_id", "name", "gender", "nationality", "split"],
        skipinitialspace=True,
    )

    # Normalise gender labels to lowercase
    df["gender"] = df["gender"].str.strip().str.lower()

    # Keep only male / female
    df = df[df["gender"].isin(["m", "f"])].copy()
    df["gender"] = df["gender"].map({"m": "male", "f": "female"})

    # ------------------------------------------------------------------
    # Find corresponding audio files
    # ------------------------------------------------------------------
    rows = []
    audio_dir = args.audio_dir.resolve()

    for _, row in df.iterrows():
        speaker_dir = audio_dir / row["speaker_id"]
        if not speaker_dir.exists():
            continue

        # Each speaker has subdirectories with .wav files
        for wav_file in speaker_dir.rglob("*.wav"):
            rows.append(
                {
                    "speaker_id": row["speaker_id"],
                    "audio_path": str(wav_file),
                    "gender": row["gender"],
                }
            )

    if not rows:
        print(
            f"WARNING: No audio files found in {audio_dir}. "
            "Make sure VoxCeleb audio data is downloaded."
        )

    # ------------------------------------------------------------------
    # Write output manifest
    # ------------------------------------------------------------------
    args.output.parent.mkdir(parents=True, exist_ok=True)
    manifest = pd.DataFrame(rows)
    manifest.to_csv(args.output, index=False, quoting=csv.QUOTE_MINIMAL)

    print(f"Wrote {len(manifest)} entries to {args.output}")
    print(f"  Males:   {(manifest['gender'] == 'male').sum()}")
    print(f"  Females: {(manifest['gender'] == 'female').sum()}")


if __name__ == "__main__":
    main()
