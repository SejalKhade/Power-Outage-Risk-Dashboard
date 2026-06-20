"""
utils.py
--------
PURPOSE: Shared helper functions used by every other file.
         Nothing here does analysis. It just makes other files cleaner.

RULE: If you find yourself writing the same code in two different files,
      move it here instead.
"""

import os
import pandas as pd
import numpy as np


# ══════════════════════════════════════════════════════════════════
# SECTION 1: FOLDER HELPERS
# ══════════════════════════════════════════════════════════════════

def ensure_dir(path: str) -> str:
    """
    Creates a folder if it does not exist yet.
    Returns the path so you can use it on the same line.

    EXAMPLE:
        filepath = os.path.join(ensure_dir("outputs"), "results.csv")
        # This creates the outputs/ folder AND gives you the path in one line

    WHY os.makedirs with exist_ok=True:
        os.mkdir() crashes if the folder already exists.
        exist_ok=True says "if it exists, that is fine, do nothing".
    """
    os.makedirs(path, exist_ok=True)
    return path


# ══════════════════════════════════════════════════════════════════
# SECTION 2: DATA TYPE CONVERTERS
# ══════════════════════════════════════════════════════════════════

def safe_num(val) -> float:
    """
    Converts any value to float. Returns 0.0 if conversion fails.

    WHY THIS EXISTS:
        EIA and NOAA data contains values like '--', 'N/A', ' ', empty strings.
        pd.to_numeric() with errors='coerce' turns these into NaN (Not a Number).
        NaN causes problems in calculations and ML models.
        We use 0.0 as a safe default instead.

    EXAMPLES:
        safe_num(42)        → 42.0
        safe_num("3.14")    → 3.14
        safe_num("N/A")     → 0.0
        safe_num(None)      → 0.0
        safe_num("--")      → 0.0
    """
    try:
        return float(val)
    except (ValueError, TypeError):
        return 0.0


def convert_damage_to_float(val) -> float:
    """
    Converts NOAA damage strings to actual dollar amounts.

    WHY THIS EXISTS:
        NOAA Storm Events stores property damage as strings like:
        '$1.5M', '$200K', '$1B', '1500000', 'nan', ''
        We need actual numbers for calculations and ML features.

    HOW IT WORKS:
        1. Check for null/empty → return 0.0
        2. Remove $ and commas
        3. Check last character for K/M/B suffix
        4. Multiply by the right amount
        5. If no suffix, just convert to float directly

    EXAMPLES:
        convert_damage_to_float('$1.5M')   → 1_500_000.0
        convert_damage_to_float('$200K')   → 200_000.0
        convert_damage_to_float('$2B')     → 2_000_000_000.0
        convert_damage_to_float('nan')     → 0.0
        convert_damage_to_float('')        → 0.0
        convert_damage_to_float(1500000)   → 1_500_000.0
    """
    # Handle null/empty values first
    if pd.isna(val):
        return 0.0

    val_str = str(val).strip()

    if val_str in ('', 'nan', 'NaN', 'None', '--', 'N/A'):
        return 0.0

    # Remove currency symbols and commas
    # '$1,500,000' → '1500000'
    val_str = val_str.replace('$', '').replace(',', '').upper()

    # Check for K/M/B suffixes
    # Dictionary: suffix → multiplier
    multipliers = {
        'K': 1_000,           # Thousand
        'M': 1_000_000,       # Million
        'B': 1_000_000_000,   # Billion
    }

    for suffix, multiplier in multipliers.items():
        if val_str.endswith(suffix):
            try:
                # Remove the suffix letter and multiply
                number_part = val_str[:-1]   # '1.5M' → '1.5'
                return float(number_part) * multiplier
            except ValueError:
                return 0.0

    # No suffix — try direct conversion
    try:
        return float(val_str)
    except ValueError:
        return 0.0


# ══════════════════════════════════════════════════════════════════
# SECTION 3: LOGGING HELPERS
# ══════════════════════════════════════════════════════════════════

def log_step(step_name: str, df: pd.DataFrame) -> None:
    """
    Prints a standardised summary line after each pipeline step.

    WHY THIS EXISTS:
        When your pipeline runs, you need to see what is happening.
        Without logging, a silent bug could process 0 rows and you
        would never know. This gives you consistent output to check.

    OUTPUT EXAMPLE:
        [PREPROCESS  ] rows= 1,910,188  cols=58  memory= 147.3 MB

    HOW MEMORY IS CALCULATED:
        df.memory_usage(deep=True) returns bytes per column.
        .sum() adds them up.
        / 1_048_576 converts bytes to megabytes (1MB = 1024 * 1024 bytes)
    """
    mem_mb = df.memory_usage(deep=True).sum() / 1_048_576

    print(
        f"[{step_name.upper():<12}] "   # left-aligned, 12 chars wide
        f"rows={len(df):>10,}  "         # right-aligned with comma separator
        f"cols={df.shape[1]:>3}  "       # right-aligned, 3 chars wide
        f"memory={mem_mb:>7.1f} MB"      # right-aligned, 1 decimal place
    )


def log_section(title: str) -> None:
    """
    Prints a visual section separator for readability in terminal output.

    OUTPUT EXAMPLE:
        ══════════════════════════════════════════════
          STAGE 2 — FEATURE ENGINEERING
        ══════════════════════════════════════════════
    """
    width = 50
    print(f"\n{'═' * width}")
    print(f"  {title}")
    print(f"{'═' * width}")


# ══════════════════════════════════════════════════════════════════
# SECTION 4: FILE HELPERS
# ══════════════════════════════════════════════════════════════════

def get_available_columns(filepath: str, required_cols: list) -> list:
    """
    Reads ONLY the header row of a CSV to check which columns exist.
    Returns the subset of required_cols that are actually in the file.

    WHY READ ONLY THE HEADER:
        Your CSV is 2GB. Reading the whole file just to get column names
        wastes 30+ seconds. pd.read_csv with nrows=0 reads zero data rows
        but still loads the header. Instant.

    WHY THIS EXISTS:
        EIA data format changes slightly between years.
        A column called 'NERC Region' in 2024 data might be called
        'NERC_Region' in 2025 data. This function handles that gracefully
        instead of crashing with KeyError.
    """
    header_only = pd.read_csv(filepath, nrows=0)
    available = set(header_only.columns)

    # Return only the columns that exist in the actual file
    found     = [c for c in required_cols if c in available]
    not_found = [c for c in required_cols if c not in available]

    if not_found:
        print(f"  WARNING: These columns not found in file: {not_found}")

    return found


# ══════════════════════════════════════════════════════════════════
# SECTION 5: SELF-TEST
# Run this file directly to verify everything works:
#   python -m src.utils
# ══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("Testing utils.py...\n")

    # Test safe_num
    assert safe_num(42)      == 42.0,  "FAIL: safe_num(42)"
    assert safe_num("3.14")  == 3.14,  "FAIL: safe_num('3.14')"
    assert safe_num("N/A")   == 0.0,   "FAIL: safe_num('N/A')"
    assert safe_num(None)    == 0.0,   "FAIL: safe_num(None)"
    print("✓ safe_num        — all tests passed")

    # Test convert_damage_to_float
    assert convert_damage_to_float("$1.5M")  == 1_500_000.0
    assert convert_damage_to_float("$200K")  == 200_000.0
    assert convert_damage_to_float("$2B")    == 2_000_000_000.0
    assert convert_damage_to_float("nan")    == 0.0
    assert convert_damage_to_float("")       == 0.0
    print("✓ convert_damage  — all tests passed")

    # Test ensure_dir
    test_path = "outputs/test_dir"
    ensure_dir(test_path)
    assert os.path.exists(test_path), "FAIL: ensure_dir"
    os.rmdir(test_path)
    print("✓ ensure_dir      — all tests passed")

    # Test log_step
    test_df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    log_step("TEST", test_df)
    print("✓ log_step        — all tests passed")

    print("\n✅ All tests passed. utils.py is working correctly.")