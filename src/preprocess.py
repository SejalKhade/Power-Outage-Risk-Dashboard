"""
preprocess.py
─────────────
PURPOSE : Convert raw merged CSV → clean Parquet file
INPUT   : data/raw/merged_utility_storm_2024.csv   (~3.4M rows, ~2GB)
OUTPUT  : data/processed/dashboard_clean_dataset.parquet
          data/processed/dashboard_clean_dataset.csv
"""

import os
import sys
import logging
import pandas as pd
import numpy as np
from src.utils import (
    safe_num,
    convert_damage_to_float,
    ensure_dir,
    log_step,
    log_section,
    get_available_columns,
)

# ── Logging setup ──────────────────────────────────────────────────
# WHY logging instead of print:
#   print() has no timestamp, no severity level, no way to turn it off.
#   logging gives you timestamps, levels (INFO/WARNING/ERROR),
#   and you can redirect to a file in production with one line change.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════

CHUNK_SIZE = 200_000

VALID_US_STATES = {
    'AL','AK','AZ','AR','CA','CO','CT','DE','FL','GA',
    'HI','ID','IL','IN','IA','KS','KY','LA','ME','MD',
    'MA','MI','MN','MS','MO','MT','NE','NV','NH','NJ',
    'NM','NY','NC','ND','OH','OK','OR','PA','RI','SC',
    'SD','TN','TX','UT','VT','VA','WA','WV','WI','WY'
}

KEEP_COLUMNS = [
    "Utility Number",
    "Utility Name",
    "State",
    "Ownership",
    "NERC Region",
    "County_Count",
    "IEEE_AllEvents_SAIDI_min_per_yr",
    "IEEE_AllEvents_SAIFI_times_per_yr",
    "IEEE_AllEvents_CAIDI_min_per_interruption",
    "IEEE_NoMED_SAIDI_min_per_yr",
    "IEEE_NoMED_SAIFI_times_per_yr",
    "IEEE_NoMED_CAIDI_min_per_interruption",
    "EVENT_TYPE",
    "MONTH_NAME",
    "MAGNITUDE",
    "DAMAGE_PROPERTY",
    "DAMAGE_CROPS",
    "INJURIES_DIRECT",
    "INJURIES_INDIRECT",
    "DEATHS_DIRECT",
    "DEATHS_INDIRECT",
    "BEGIN_YEARMONTH",
]

NUMERIC_COLS = [
    "MAGNITUDE",
    "INJURIES_DIRECT", "INJURIES_INDIRECT",
    "DEATHS_DIRECT",   "DEATHS_INDIRECT",
    "IEEE_AllEvents_SAIDI_min_per_yr",
    "IEEE_AllEvents_SAIFI_times_per_yr",
    "IEEE_AllEvents_CAIDI_min_per_interruption",
    "IEEE_NoMED_SAIDI_min_per_yr",
    "IEEE_NoMED_SAIFI_times_per_yr",
    "IEEE_NoMED_CAIDI_min_per_interruption",
    "County_Count",
]

TEXT_COLS = ["EVENT_TYPE", "MONTH_NAME", "Ownership", "NERC Region"]

PROCESSED_DIR  = "data/processed"
OUTPUT_PARQUET = os.path.join(PROCESSED_DIR, "dashboard_clean_dataset.parquet")
OUTPUT_CSV     = os.path.join(PROCESSED_DIR, "dashboard_clean_dataset.csv")


# ══════════════════════════════════════════════════════════════════
# MAIN FUNCTION
# ══════════════════════════════════════════════════════════════════

def create_clean_dataset(input_filepath: str) -> pd.DataFrame:
    """
    Reads raw CSV in chunks, cleans each chunk, combines, saves.

    WHY CHUNKS:
        3.4M rows × 58 columns ≈ 2GB RAM if loaded at once.
        Processing 200K rows at a time keeps peak memory under 1GB.
    """
    ensure_dir(PROCESSED_DIR)
    log_section("STAGE 1 — PREPROCESSING")
    logger.info("Input file: %s", input_filepath)

    if not os.path.exists(input_filepath):
        raise FileNotFoundError(
            f"\nERROR: Cannot find {input_filepath}\n"
            f"Download from Google Drive link in README and place in data/raw/\n"
        )

    # ── Find which desired columns exist in this file ──────────────
    logger.info("Scanning column headers...")
    available_cols = get_available_columns(input_filepath, KEEP_COLUMNS)
    logger.info(
        "Found %d of %d desired columns",
        len(available_cols), len(KEEP_COLUMNS)
    )

    # ── Process chunks ─────────────────────────────────────────────
    cleaned_chunks = []
    total_raw_rows = 0
    chunk_number   = 0

    logger.info("Processing chunks (size=%d)...", CHUNK_SIZE)

    for chunk in pd.read_csv(
        input_filepath,
        chunksize=CHUNK_SIZE,
        low_memory=False,
        encoding="utf-8",
        on_bad_lines="skip",
    ):
        chunk_number   += 1
        raw_count       = len(chunk)
        total_raw_rows += raw_count

        # FIX: select columns FIRST, then pass to cleaner
        # Previous version selected cols into `df` but then
        # passed original `chunk` to _clean_single_chunk — bug.
        cols_present = [c for c in KEEP_COLUMNS if c in chunk.columns]
        chunk_selected = chunk[cols_present].copy()

        cleaned    = _clean_single_chunk(chunk_selected)
        clean_count = len(cleaned)
        pct_kept    = (clean_count / raw_count * 100) if raw_count > 0 else 0

        logger.info(
            "Chunk %3d | raw=%8s | clean=%8s | kept=%.1f%%",
            chunk_number,
            f"{raw_count:,}",
            f"{clean_count:,}",
            pct_kept,
        )

        if clean_count > 0:
            cleaned_chunks.append(cleaned)

    # ── Combine all chunks ─────────────────────────────────────────
    logger.info("Total raw rows processed: %s", f"{total_raw_rows:,}")
    logger.info("Combining %d chunks...", len(cleaned_chunks))
    df = pd.concat(cleaned_chunks, ignore_index=True)

    # ── Remove duplicates ──────────────────────────────────────────
    before = len(df)
    df     = df.drop_duplicates()
    removed = before - len(df)
    if removed > 0:
        logger.info("Removed %s duplicate rows", f"{removed:,}")

    # ── Save outputs ───────────────────────────────────────────────
    df.to_parquet(OUTPUT_PARQUET, index=False)
    df.to_csv(OUTPUT_CSV, index=False)

    logger.info("Saved → %s", OUTPUT_PARQUET)
    logger.info("Saved → %s", OUTPUT_CSV)
    log_step("PREPROCESS", df)

    return df


# ══════════════════════════════════════════════════════════════════
# PRIVATE HELPER
# ══════════════════════════════════════════════════════════════════

def _clean_single_chunk(chunk: pd.DataFrame) -> pd.DataFrame:
    """
    Applies all cleaning operations to one chunk.
    Receives a chunk that has already been column-filtered.
    """
    df = chunk.copy()

    # 1. Filter to valid US states
    if "State" in df.columns:
        before = len(df)
        df     = df[df["State"].isin(VALID_US_STATES)]
        dropped = before - len(df)
        if dropped > 0:
            logger.debug("State filter dropped %d rows", dropped)

    if df.empty:
        return df

    # 2. Convert damage strings to floats
    for damage_col in ["DAMAGE_PROPERTY", "DAMAGE_CROPS"]:
        if damage_col in df.columns:
            new_col = damage_col + "_USD"
            df[new_col] = df[damage_col].apply(convert_damage_to_float)

    # 3. Fix numeric columns
    for col in NUMERIC_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    # 4. Fill missing text
    for col in TEXT_COLS:
        if col in df.columns:
            df[col] = df[col].fillna("Unknown").astype(str).str.strip()

    # 5. Keep all rows — storm event rows with SAIDI=0 are needed
    # for weather features. Aggregation happens in features.py.

    return df


# ══════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    input_file = (
        sys.argv[1]
        if len(sys.argv) > 1
        else "data/raw/merged_utility_storm_2024.csv"
    )

    df = create_clean_dataset(input_file)

    logger.info("=" * 50)
    logger.info("PREPROCESSING COMPLETE")
    logger.info("Final shape: %s rows x %d columns", f"{df.shape[0]:,}", df.shape[1])
    logger.info("States covered: %d", df["State"].nunique())
    logger.info("Output: %s", OUTPUT_PARQUET)
    logger.info("=" * 50)