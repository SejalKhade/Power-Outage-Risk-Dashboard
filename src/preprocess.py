"""
preprocess.py
─────────────
PURPOSE : Convert raw merged CSV → clean Parquet file
INPUT   : data/raw/merged_utility_storm_2024.csv   (~3.4M rows, ~2GB)
OUTPUT  : data/processed/dashboard_clean_dataset.parquet
          data/processed/dashboard_clean_dataset.csv

WHY PARQUET INSTEAD OF CSV:
    CSV is plain text. Every time you load it, Python reads character
    by character and figures out types. Slow.
    Parquet is binary and columnar. Types are stored. Loads 10x faster.
    5x smaller file size. Industry standard for analytics pipelines.
    Your dashboard loads in 2 seconds instead of 20.
"""

import os
import sys
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

# ══════════════════════════════════════════════════════════════════
# CONFIGURATION — change these values in one place, affects everything
# ══════════════════════════════════════════════════════════════════

# How many rows to process at once
# 200,000 rows ≈ 500MB RAM per chunk — safe for most laptops
CHUNK_SIZE = 200_000

# Only keep these 50 states + DC
# Filters out Puerto Rico (PR), Guam (GU), Virgin Islands (VI)
VALID_US_STATES = {
    'AL','AK','AZ','AR','CA','CO','CT','DE','FL','GA',
    'HI','ID','IL','IN','IA','KS','KY','LA','ME','MD',
    'MA','MI','MN','MS','MO','MT','NE','NV','NH','NJ',
    'NM','NY','NC','ND','OH','OK','OR','PA','RI','SC',
    'SD','TN','TX','UT','VT','VA','WA','WV','WI','WY'
    # DC excluded — not a utility state in EIA data
    # CN excluded — Canadian utilities in some EIA records
}

# Only keep these columns — everything else is dropped
# This reduces memory from ~2GB to ~300MB
KEEP_COLUMNS = [
    # Who the utility is
    "Utility Number",
    "Utility Name",
    "State",
    "Ownership",
    "NERC Region",
    "County_Count",

    # How reliable their grid is (IEEE 1366 standard metrics)
    # SAIDI = System Average Interruption Duration Index
    #         How many minutes per year does the average customer lose power?
    #         Lower = better. National average ≈ 322 min/yr
    "IEEE_AllEvents_SAIDI_min_per_yr",

    # SAIFI = System Average Interruption Frequency Index
    #         How many times per year does the average customer lose power?
    #         Lower = better. National average ≈ 1.3 times/yr
    "IEEE_AllEvents_SAIFI_times_per_yr",

    # CAIDI = Customer Average Interruption Duration Index
    #         When power goes out, how long does it stay out on average?
    #         CAIDI = SAIDI / SAIFI
    "IEEE_AllEvents_CAIDI_min_per_interruption",

    # Same metrics but excluding Major Event Days (storms, hurricanes)
    "IEEE_NoMED_SAIDI_min_per_yr",
    "IEEE_NoMED_SAIFI_times_per_yr",
    "IEEE_NoMED_CAIDI_min_per_interruption",

    # Storm event details from NOAA
    "EVENT_TYPE",           # tornado, hurricane, thunderstorm, etc.
    "MONTH_NAME",           # which month the event occurred
    "MAGNITUDE",            # storm intensity measurement
    "DAMAGE_PROPERTY",      # string like '$1.5M' — we convert this
    "DAMAGE_CROPS",         # string like '$200K' — we convert this
    "INJURIES_DIRECT",      # people injured directly by storm
    "INJURIES_INDIRECT",    # people injured indirectly
    "DEATHS_DIRECT",        # fatalities directly from storm
    "DEATHS_INDIRECT",      # fatalities indirectly from storm
    "BEGIN_YEARMONTH",      # date field for time filtering
]

# Where to save outputs
PROCESSED_DIR  = "data/processed"
OUTPUT_PARQUET = os.path.join(PROCESSED_DIR, "dashboard_clean_dataset.parquet")
OUTPUT_CSV     = os.path.join(PROCESSED_DIR, "dashboard_clean_dataset.csv")


# ══════════════════════════════════════════════════════════════════
# MAIN FUNCTION
# ══════════════════════════════════════════════════════════════════

def create_clean_dataset(input_filepath: str) -> pd.DataFrame:
    """
    Reads the raw CSV in chunks, cleans each chunk, combines, saves.

    WHY CHUNKS:
        3.4 million rows × 58 columns ≈ 2GB RAM if loaded all at once.
        Most laptops have 8-16GB RAM but also run Chrome, VS Code, etc.
        Processing 200K rows at a time keeps peak memory under 1GB.

    RETURNS:
        Clean DataFrame with valid US states, fixed data types,
        and standardised column names.
    """
    ensure_dir(PROCESSED_DIR)
    log_section("STAGE 1 — PREPROCESSING")
    print(f"  Input file: {input_filepath}")

    if not os.path.exists(input_filepath):
        raise FileNotFoundError(
            f"\nERROR: Cannot find {input_filepath}\n"
            f"Please download the data file from the Google Drive link in README.md\n"
            f"and place it in data/raw/\n"
        )

    # ── Find which of our desired columns actually exist in this file ──
    print("\n  Scanning column headers...")
    available_cols = get_available_columns(input_filepath, KEEP_COLUMNS)
    print(f"  Found {len(available_cols)} of {len(KEEP_COLUMNS)} desired columns")

    # ── Process chunks ──
    cleaned_chunks = []
    total_raw_rows = 0
    chunk_number   = 0

    print("\n  Processing chunks:")
    print(f"  {'Chunk':>5}  {'Raw rows':>10}  {'Clean rows':>10}  {'Kept %':>7}")
    print(f"  {'─'*5}  {'─'*10}  {'─'*10}  {'─'*7}")

    for chunk in pd.read_csv(
        input_filepath,
        chunksize=CHUNK_SIZE,
        low_memory=False,
        encoding='utf-8',
        on_bad_lines='skip',
    ):
        chunk_number   += 1
        raw_count       = len(chunk)
        total_raw_rows += raw_count

        # Clean this chunk
        df = chunk.copy()

    # Select only the columns we need from the full 71-column chunk
    # WHY: We moved column selection here instead of at read time
    # to prevent pandas from silently dropping rows when using usecols
        cols_to_keep = [c for c in KEEP_COLUMNS if c in df.columns]
        df = df[cols_to_keep]
        cleaned = _clean_single_chunk(chunk)
        clean_count = len(cleaned)

        pct_kept = (clean_count / raw_count * 100) if raw_count > 0 else 0
        print(f"  {chunk_number:>5}  {raw_count:>10,}  {clean_count:>10,}  {pct_kept:>6.1f}%")

        if clean_count > 0:
            cleaned_chunks.append(cleaned)

    # ── Combine all chunks ──
    print(f"\n  Total raw rows processed: {total_raw_rows:,}")
    print("  Combining chunks...")
    df = pd.concat(cleaned_chunks, ignore_index=True)

    # ── Remove duplicates ──
    before = len(df)
    df = df.drop_duplicates()
    print(f"  Removed {before - len(df):,} duplicate rows")

    # ── Save outputs ──
    df.to_parquet(OUTPUT_PARQUET, index=False)
    df.to_csv(OUTPUT_CSV, index=False)

    print(f"\n  ✓ Saved → {OUTPUT_PARQUET}")
    print(f"  ✓ Saved → {OUTPUT_CSV}")
    log_step("PREPROCESS", df)

    return df


# ══════════════════════════════════════════════════════════════════
# PRIVATE HELPER — only called by create_clean_dataset()
# The underscore prefix _ means "internal, do not call from outside"
# ══════════════════════════════════════════════════════════════════

def _clean_single_chunk(chunk: pd.DataFrame) -> pd.DataFrame:
    """
    Applies all cleaning operations to one chunk.

    CLEANING OPERATIONS (in order):
        1. Filter to valid US states
        2. Convert damage strings to floats
        3. Fix numeric columns with mixed types
        4. Fill missing text with 'Unknown'
        5. Remove rows where ALL reliability metrics are zero
    """

    df = chunk.copy()   # never modify the original chunk

    # ── 1. Filter to valid US states ──────────────────────────────
    # WHY: Raw data includes US territories (Puerto Rico, Guam etc.)
    # These use different reliability standards. We exclude them.
    if "State" in df.columns:
        df = df[df["State"].isin(VALID_US_STATES)]

    if df.empty:
        return df  # nothing left after state filter, skip this chunk

    # ── 2. Convert damage strings to floats ───────────────────────
    # WHY: NOAA stores damage as '$1.5M'. We need 1500000.0
    # We create NEW columns with _USD suffix and keep originals
    for damage_col in ["DAMAGE_PROPERTY", "DAMAGE_CROPS"]:
        if damage_col in df.columns:
            new_col = damage_col + "_USD"
            df[new_col] = df[damage_col].apply(convert_damage_to_float)

    # ── 3. Fix numeric columns ────────────────────────────────────
    # WHY: pd.read_csv sometimes reads numbers as strings ('3.14' not 3.14)
    # especially when a column has mixed types like '3.14' and 'N/A'.
    # pd.to_numeric with errors='coerce' converts what it can,
    # turns failures into NaN, then fillna(0) replaces NaN with 0.
    numeric_cols = [
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
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)

    # ── 4. Fill missing text columns ──────────────────────────────
    # WHY: Missing strings become float NaN in pandas which causes
    # .str operations to fail. Replacing with 'Unknown' keeps them
    # as strings throughout the pipeline.
    text_cols = ["EVENT_TYPE", "MONTH_NAME", "Ownership", "NERC Region"]
    for col in text_cols:
        if col in df.columns:
            df[col] = df[col].fillna("Unknown").str.strip()

    # ── 5. Keep ALL rows at this stage ────────────────────────────
    # WHY: Your merged file has one row per utility-storm combination.
    # Reliability metrics (SAIDI/SAIFI) only appear on ONE row per
    # utility per year. The other rows (storm event records) show 0
    # because they are NOAA storm records, not reliability records.
    #
    # If we drop zero-reliability rows here, we lose all the storm
    # event context (damage, injuries, event types) that we need
    # for weather features in the ML model.
    #
    # We will handle aggregation in features.py where we take the
    # MAX of SAIDI/SAIFI per utility — which correctly picks up the
    # one row that has the real reliability value.
    pass  # keep all rows

    return df


# ══════════════════════════════════════════════════════════════════
# RUN AS SCRIPT
# Usage: python -m src.preprocess
#    or: python -m src.preprocess data/raw/your_file.csv
# ══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    input_file = (
        sys.argv[1]
        if len(sys.argv) > 1
        else "data/raw/merged_utility_storm_2024.csv"
    )

    df = create_clean_dataset(input_file)

    print(f"\n{'═'*50}")
    print("  PREPROCESSING COMPLETE")
    print(f"  Final shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
    print(f"  States covered: {df['State'].nunique()}")
    print(f"  Output: {OUTPUT_PARQUET}")
    print(f"{'═'*50}\n")