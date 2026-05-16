"""
Prepare AgeVoxCeleb age-group metadata.

This script parses the AgeVoxCeleb dataset annotations and produces
a cleaned manifest CSV at ``data/processed/age_manifest.csv``.

AgeVoxCeleb extends VoxCeleb with age labels derived from Wikipedia
birth dates and video upload dates.

Expected input: AgeVoxCeleb metadata file (e.g., ``agevoxceleb_meta.csv``)
in ``data/raw/``.  Columns may vary by version; the script expects at
minimum: speaker_id, age (integer), and a path or utterance ID.

Output CSV columns:
  speaker_id, audio_path, age, age_group

Age groups:
  - child:       age < 18
  - young_adult: 18 <= age < 30
  - adult:       30 <= age < 60
  - senior:      age >= 60

Usage:
    uv run python scripts/prepare_agevoxceleb.py \\
        --meta data/raw/agevoxceleb_meta.csv \\
        --audio-dir data/raw/voxceleb1/wav \\
        --output data/processed/age_manifest.csv
"""

import argparse
import csv
import sys
from pathlib import Path

import pandas as pd


def age_to_group(age: int) -> str:
    """Map a numeric age to a categorical age group."""
    if age < 18:
        return "child"
    elif age < 30:
        return "young_adult"
    elif age < 60:
        return "adult"
    else:
        return "senior"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare AgeVoxCeleb age-group manifest for training."
    )
    parser.add_argument(
        "--meta",
        type=Path,
        default=Path("data/raw/agevoxceleb_meta.csv"),
        help="Path to AgeVoxCeleb metadata CSV.",
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
        default=Path("data/processed/age_manifest.csv"),
        help="Output manifest CSV path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.meta.exists():
        print(f"ERROR: Metadata file not found: {args.meta}")
        print(
            "Download AgeVoxCeleb annotations from the project page "
            "and place them in data/raw/"
        )
        sys.exit(1)

    # ------------------------------------------------------------------
    # Parse metadata
    # ------------------------------------------------------------------
    print(f"Reading metadata from {args.meta}")
    df = pd.read_csv(args.meta)

    # Ensure required columns exist
    required_cols = {"speaker_id", "age"}
    if not required_cols.issubset(df.columns):
        # Try alternative column names
        col_map = {}
        for col in df.columns:
            lc = col.lower().strip()
            if "speaker" in lc or "id" in lc:
                col_map[col] = "speaker_id"
            elif lc in ("age", "estimated_age"):
                col_map[col] = "age"
        df = df.rename(columns=col_map)

    if "speaker_id" not in df.columns or "age" not in df.columns:
        print(f"ERROR: Could not find required columns. Found: {list(df.columns)}")
        sys.exit(1)

    # Drop rows with missing age
    df = df.dropna(subset=["age"])
    df["age"] = df["age"].astype(int)
    df["age_group"] = df["age"].apply(age_to_group)

    # ------------------------------------------------------------------
    # Find corresponding audio files
    # ------------------------------------------------------------------
    rows = []
    audio_dir = args.audio_dir.resolve()

    for _, row in df.iterrows():
        speaker_dir = audio_dir / str(row["speaker_id"])
        if not speaker_dir.exists():
            continue

        for wav_file in speaker_dir.rglob("*.wav"):
            rows.append(
                {
                    "speaker_id": str(row["speaker_id"]),
                    "audio_path": str(wav_file),
                    "age": int(row["age"]),
                    "age_group": row["age_group"],
                }
            )

    if not rows:
        print(
            f"WARNING: No audio files found in {audio_dir}. "
            "Make sure VoxCeleb audio data is downloaded."
        )

    # ------------------------------------------------------------------
    # Write manifest
    # ------------------------------------------------------------------
    args.output.parent.mkdir(parents=True, exist_ok=True)
    manifest = pd.DataFrame(rows)
    manifest.to_csv(args.output, index=False, quoting=csv.QUOTE_MINIMAL)

    print(f"Wrote {len(manifest)} entries to {args.output}")
    for group in ["child", "young_adult", "adult", "senior"]:
        count = (manifest["age_group"] == group).sum() if len(manifest) > 0 else 0
        print(f"  {group:15s}: {count}")


if __name__ == "__main__":
    main()
