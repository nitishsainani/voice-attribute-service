"""
Prepare Common Voice Spontaneous Speech dataset.

This script reads the spontaneous-speech corpus TSV and produces processed
CSVs for gender training, age training, evaluation, and a universal
audio smoke-test manifest.

Expected input layout::

    data/raw/common_voice_spontaneous_en/
    ├── ss-corpus-en.tsv
    └── audios/
        ├── spontaneous-speech-en-1.mp3
        └── ...

Outputs (when labels are present):
  - data/processed/gender_train.csv        (audio_path, gender)
  - data/processed/age_train.csv           (audio_path, age_group)
  - data/processed/eval_common_voice.csv   (audio_path, gender, age_group)
  - data/processed/audio_smoke_test.csv    (audio_path, dataset)  — always

Usage:
    uv run python scripts/prepare_common_voice_spontaneous.py \\
        --dataset-root data/raw/common_voice_spontaneous_en \\
        --max-rows 500
"""

import argparse
import csv
import re
import sys
from pathlib import Path

import pandas as pd


# -----------------------------------------------------------------------
# Age-range mapping helpers
# -----------------------------------------------------------------------
# Common Voice uses descriptive labels like "twenties", "thirties", etc.
# We also handle explicit numeric ages and range strings like "25-34".
_CV_AGE_LABEL_MAP: dict[str, str] = {
    "teens": "18-30",
    "twenties": "18-30",
    "thirties": "31-45",
    "fourties": "31-45",
    "forties": "31-45",
    "fifties": "46-60",
    "sixties": "60+",
    "seventies": "60+",
    "eighties": "60+",
    "nineties": "60+",
}

_AGE_GROUPS = ["18-30", "31-45", "46-60", "60+"]


def _numeric_age_to_group(age: int) -> str:
    """Map a numeric age to one of the four canonical age groups."""
    if age <= 30:
        return "18-30"
    elif age <= 45:
        return "31-45"
    elif age <= 60:
        return "46-60"
    else:
        return "60+"


def _parse_age_range_string(value: str) -> str | None:
    """Handle explicit range strings like '25-34', '60-69', etc."""
    m = re.match(r"(\d+)\s*[-–]\s*(\d+)", value)
    if m:
        mid = (int(m.group(1)) + int(m.group(2))) // 2
        return _numeric_age_to_group(mid)
    return None


def map_age(raw_value: str) -> str | None:
    """Convert an arbitrary age representation to a canonical age group."""
    if pd.isna(raw_value):
        return None

    value = str(raw_value).strip().lower()
    if not value:
        return None

    # 1. Descriptive label (e.g. "twenties")
    if value in _CV_AGE_LABEL_MAP:
        return _CV_AGE_LABEL_MAP[value]

    # 2. Already a canonical group
    if value in _AGE_GROUPS:
        return value

    # 3. Explicit range string ("25-34")
    group = _parse_age_range_string(value)
    if group:
        return group

    # 4. Plain integer
    try:
        return _numeric_age_to_group(int(float(value)))
    except (ValueError, OverflowError):
        return None


# -----------------------------------------------------------------------
# Gender normalization
# -----------------------------------------------------------------------
_GENDER_MAP: dict[str, str] = {
    "m": "male",
    "male": "male",
    "male_masculine": "male",
    "masculine": "male",
    "f": "female",
    "female": "female",
    "female_feminine": "female",
    "feminine": "female",
}


def normalize_gender(raw_value: str) -> str | None:
    """Normalize gender to male/female, or None if unrecognised."""
    if pd.isna(raw_value):
        return None
    value = str(raw_value).strip().lower()
    return _GENDER_MAP.get(value)


# -----------------------------------------------------------------------
# Column detection
# -----------------------------------------------------------------------
_AUDIO_COLUMN_CANDIDATES = ["path", "audio", "filename", "audio_path", "audio_file"]


def detect_column(columns: list[str], candidates: list[str]) -> str | None:
    """Return the first column name that matches any candidate (case-insensitive)."""
    col_lower = {c.lower().strip(): c for c in columns}
    for candidate in candidates:
        if candidate in col_lower:
            return col_lower[candidate]
    return None


# -----------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare Common Voice Spontaneous Speech dataset.",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("data/raw/common_voice_spontaneous_en"),
        help="Root directory of the Common Voice Spontaneous dataset.",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Optional: limit the number of TSV rows to process.",
    )
    return parser.parse_args()


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------
def main() -> None:
    args = parse_args()

    tsv_path = args.dataset_root / "ss-corpus-en.tsv"
    audios_dir = args.dataset_root / "audios"
    output_dir = Path("data/processed")

    if not tsv_path.exists():
        print(f"ERROR: TSV file not found: {tsv_path}")
        sys.exit(1)

    # ------------------------------------------------------------------
    # 1. Read TSV
    # ------------------------------------------------------------------
    print(f"Reading {tsv_path} ...")
    df = pd.read_csv(tsv_path, sep="\t", nrows=args.max_rows)
    print(f"Loaded {len(df)} rows.\n")

    # ------------------------------------------------------------------
    # 2. Print all available columns
    # ------------------------------------------------------------------
    print("Available columns:")
    for col in df.columns:
        print(f"  - {col}")
    print()

    # ------------------------------------------------------------------
    # 3. Detect audio filename column
    # ------------------------------------------------------------------
    audio_col = detect_column(list(df.columns), _AUDIO_COLUMN_CANDIDATES)
    if audio_col is None:
        print(
            f"ERROR: Could not detect audio filename column. "
            f"Tried: {_AUDIO_COLUMN_CANDIDATES}. "
            f"Found columns: {list(df.columns)}"
        )
        sys.exit(1)
    print(f"Detected audio filename column: '{audio_col}'\n")

    # ------------------------------------------------------------------
    # 4. Build full audio paths and filter to existing files
    # ------------------------------------------------------------------
    df["audio_path"] = df[audio_col].apply(
        lambda fn: str(audios_dir / str(fn).strip()) if pd.notna(fn) else None
    )
    df = df.dropna(subset=["audio_path"])

    valid_mask = df["audio_path"].apply(lambda p: Path(p).exists())
    n_total = len(df)
    df_valid = df[valid_mask].copy()
    n_valid = len(df_valid)
    print(f"Audio files: {n_valid}/{n_total} exist on disk.\n")

    # ------------------------------------------------------------------
    # 5–6. Detect gender and age columns
    # ------------------------------------------------------------------
    has_gender = "gender" in [c.lower().strip() for c in df_valid.columns]
    has_age = "age" in [c.lower().strip() for c in df_valid.columns]

    gender_col = detect_column(list(df_valid.columns), ["gender"])
    age_col = detect_column(list(df_valid.columns), ["age"])

    gender_df: pd.DataFrame | None = None
    age_df: pd.DataFrame | None = None

    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Gender ---
    if gender_col:
        df_valid["gender_norm"] = df_valid[gender_col].apply(normalize_gender)
        gender_df = df_valid.dropna(subset=["gender_norm"]).copy()

        if len(gender_df) > 0:
            gender_out = output_dir / "gender_train.csv"
            gender_df[["audio_path", "gender_norm"]].rename(
                columns={"gender_norm": "gender"}
            ).to_csv(gender_out, index=False, quoting=csv.QUOTE_MINIMAL)
            print(f"✓ Wrote {len(gender_df)} rows → {gender_out}")
            print(f"  Males:   {(gender_df['gender_norm'] == 'male').sum()}")
            print(f"  Females: {(gender_df['gender_norm'] == 'female').sum()}")
            print()
        else:
            print(
                "Gender column exists but all values are empty or unrecognised. "
                "Skipping gender_train.csv.\n"
            )
            gender_df = None
    else:
        has_gender = False

    # --- Age ---
    if age_col:
        df_valid["age_group"] = df_valid[age_col].apply(map_age)
        age_df = df_valid.dropna(subset=["age_group"]).copy()

        if len(age_df) > 0:
            age_out = output_dir / "age_train.csv"
            age_df[["audio_path", "age_group"]].to_csv(
                age_out, index=False, quoting=csv.QUOTE_MINIMAL
            )
            print(f"✓ Wrote {len(age_df)} rows → {age_out}")
            for group in _AGE_GROUPS:
                count = (age_df["age_group"] == group).sum()
                print(f"  {group:8s}: {count}")
            print()
        else:
            print(
                "Age column exists but all values are empty or unrecognised. "
                "Skipping age_train.csv.\n"
            )
            age_df = None
    else:
        has_age = False

    # ------------------------------------------------------------------
    # 7. Combined eval CSV (only if both gender and age have data)
    # ------------------------------------------------------------------
    if gender_df is not None and age_df is not None:
        eval_df = df_valid.dropna(subset=["gender_norm", "age_group"]).copy()
        if len(eval_df) > 0:
            eval_out = output_dir / "eval_common_voice.csv"
            eval_df[["audio_path", "gender_norm", "age_group"]].rename(
                columns={"gender_norm": "gender"}
            ).to_csv(eval_out, index=False, quoting=csv.QUOTE_MINIMAL)
            print(f"✓ Wrote {len(eval_df)} rows → {eval_out}\n")

    # ------------------------------------------------------------------
    # 8. Message if no labels
    # ------------------------------------------------------------------
    if gender_df is None and age_df is None:
        print(
            "This dataset does not contain age/gender labels, "
            "so it cannot be used for classifier training.\n"
        )

    # ------------------------------------------------------------------
    # 9. Always create audio smoke test CSV
    # ------------------------------------------------------------------
    smoke_out = output_dir / "audio_smoke_test.csv"
    smoke_df = df_valid[["audio_path"]].copy()
    smoke_df["dataset"] = "common_voice_spontaneous"
    smoke_df.to_csv(smoke_out, index=False, quoting=csv.QUOTE_MINIMAL)
    print(f"✓ Wrote {len(smoke_df)} rows → {smoke_out}")


if __name__ == "__main__":
    main()
