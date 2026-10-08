"""
01_data_audit.py

Phase 1 of the F1 2026 Data-Driven Aerodynamic Simulator project.

Purpose
-------
Perform a complete first-pass audit of the Mode-1 dataset before any
modelling is performed.

Checks performed
----------------
1. File loading and dataset dimensions
2. Column names and data types
3. Missing values
4. Duplicate rows
5. Unique-value counts
6. Constant / near-constant columns
7. Numerical validation
8. Descriptive statistics
9. Plausible range checks
10. Correlation matrix
11. Continuous-input / continuous-output validation
12. Saving an audit report and column summary

Expected dataset
----------------
data/F1_2026_Synthetic_Dataset_Mode1.csv

Expected columns
----------------
Inputs:
    velocity_ms
    flap_front_angle
    flap_rear_angle
    ride_height_front
    ride_height_rear
    battery_soc

Targets:
    cl_total
    cd_total
    cop_shift
    energy_drain_kw

The script intentionally does NOT transform, remove, or modify the dataset.
It is an audit only.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_PATH = PROJECT_ROOT / "data" / "F1_2026_Synthetic_Dataset_Mode1.csv"
REPORT_DIR = PROJECT_ROOT / "reports" / "01_data_audit"

INPUT_COLUMNS = [
    "velocity_ms",
    "flap_front_angle",
    "flap_rear_angle",
    "ride_height_front",
    "ride_height_rear",
    "battery_soc",
]

TARGET_COLUMNS = [
    "cl_total",
    "cd_total",
    "cop_shift",
    "energy_drain_kw",
]

EXPECTED_COLUMNS = INPUT_COLUMNS + TARGET_COLUMNS


# ---------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------

def print_section(title: str) -> None:
    """Print a clear section heading."""
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def save_text_report(path: Path, text: str) -> None:
    """Save text content to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def check_expected_columns(df: pd.DataFrame) -> None:
    """Validate that the dataset contains exactly the expected columns."""
    actual = list(df.columns)

    missing = [col for col in EXPECTED_COLUMNS if col not in actual]
    unexpected = [col for col in actual if col not in EXPECTED_COLUMNS]

    if missing:
        raise ValueError(
            "The dataset is missing expected columns:\n"
            + "\n".join(f"  - {col}" for col in missing)
        )

    if unexpected:
        print(
            "\nWARNING: Unexpected columns found:\n"
            + "\n".join(f"  - {col}" for col in unexpected)
        )

    if actual != EXPECTED_COLUMNS:
        print("\nNOTE: Column order differs from the expected order.")
        print("The script will continue because column names are used explicitly.")


def audit_continuous_variables(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a per-column audit table.

    A variable is considered suitable as a continuous numerical variable
    when it has a numeric dtype and more than one unique value.
    """
    rows = []

    for col in df.columns:
        series = df[col]

        rows.append(
            {
                "column": col,
                "dtype": str(series.dtype),
                "is_numeric": pd.api.types.is_numeric_dtype(series),
                "n_unique": int(series.nunique(dropna=False)),
                "n_missing": int(series.isna().sum()),
                "missing_pct": float(series.isna().mean() * 100),
                "n_infinite": (
                    int(np.isinf(series.to_numpy(dtype=float)).sum())
                    if pd.api.types.is_numeric_dtype(series)
                    else np.nan
                ),
                "is_constant": bool(series.nunique(dropna=False) <= 1),
                "min": float(series.min()) if pd.api.types.is_numeric_dtype(series) else np.nan,
                "max": float(series.max()) if pd.api.types.is_numeric_dtype(series) else np.nan,
                "mean": float(series.mean()) if pd.api.types.is_numeric_dtype(series) else np.nan,
                "std": float(series.std()) if pd.api.types.is_numeric_dtype(series) else np.nan,
            }
        )

    return pd.DataFrame(rows)


def run_plausibility_checks(df: pd.DataFrame) -> list[str]:
    """
    Perform simple domain/range checks.

    These are sanity checks, NOT data-cleaning rules.
    A warning does not automatically mean an observation is wrong.
    """
    warnings = []

    checks = {
        "velocity_ms": lambda s: s < 0,
        "flap_front_angle": lambda s: s < 0,
        "flap_rear_angle": lambda s: s < 0,
        "ride_height_front": lambda s: s <= 0,
        "ride_height_rear": lambda s: s <= 0,
        "battery_soc": lambda s: (s < 0) | (s > 100),
    }

    for col, condition in checks.items():
        if col not in df.columns:
            continue

        bad = condition(df[col])
        count = int(bad.sum())

        if count > 0:
            warnings.append(
                f"{col}: {count:,} rows triggered the basic plausibility check."
            )

    return warnings


def build_report(
    df: pd.DataFrame,
    audit_df: pd.DataFrame,
    correlation: pd.DataFrame,
    plausibility_warnings: list[str],
) -> str:
    """Build a human-readable text report."""
    lines = []

    lines.append("F1 2026 AERODYNAMIC DATASET — DATA AUDIT REPORT")
    lines.append("=" * 60)
    lines.append(f"Dataset: {DATA_PATH}")
    lines.append("")

    lines.append("1. DATASET OVERVIEW")
    lines.append("-" * 60)
    lines.append(f"Rows:                  {len(df):,}")
    lines.append(f"Columns:               {df.shape[1]:,}")
    lines.append(f"Duplicate rows:        {df.duplicated().sum():,}")
    lines.append(f"Total missing values:  {int(df.isna().sum().sum()):,}")
    lines.append("")

    lines.append("2. INPUT VARIABLES")
    lines.append("-" * 60)
    for col in INPUT_COLUMNS:
        lines.append(f"  - {col}")

    lines.append("")
    lines.append("3. TARGET VARIABLES")
    lines.append("-" * 60)
    for col in TARGET_COLUMNS:
        lines.append(f"  - {col}")

    lines.append("")
    lines.append("4. COLUMN AUDIT")
    lines.append("-" * 60)
    lines.append(audit_df.to_string(index=False))

    lines.append("")
    lines.append("5. DESCRIPTIVE STATISTICS")
    lines.append("-" * 60)
    lines.append(df.describe().T.to_string())

    lines.append("")
    lines.append("6. CORRELATION MATRIX")
    lines.append("-" * 60)
    lines.append(correlation.to_string())

    lines.append("")
    lines.append("7. BASIC PLAUSIBILITY CHECKS")
    lines.append("-" * 60)
    if plausibility_warnings:
        for warning in plausibility_warnings:
            lines.append(f"WARNING: {warning}")
    else:
        lines.append("No basic plausibility violations were detected.")

    lines.append("")
    lines.append("8. CONTINUOUS-MODELLING VALIDATION")
    lines.append("-" * 60)

    non_numeric = [
        col
        for col in EXPECTED_COLUMNS
        if col in df.columns and not pd.api.types.is_numeric_dtype(df[col])
    ]
    constant = [
        col
        for col in EXPECTED_COLUMNS
        if col in df.columns and df[col].nunique(dropna=False) <= 1
    ]

    if not non_numeric:
        lines.append("All expected variables are numerical.")
    else:
        lines.append("NON-NUMERIC VARIABLES:")
        for col in non_numeric:
            lines.append(f"  - {col}")

    if not constant:
        lines.append("No expected variable is constant.")
    else:
        lines.append("CONSTANT VARIABLES:")
        for col in constant:
            lines.append(f"  - {col}")

    lines.append("")
    lines.append("Conclusion:")
    if not non_numeric and not constant:
        lines.append(
            "The supplied Mode-1 dataset is suitable for a continuous-input, "
            "continuous-output regression pipeline, subject to further modelling checks."
        )
    else:
        lines.append(
            "The dataset needs structural review before modelling because one or more "
            "expected variables are non-numeric or constant."
        )

    return "\n".join(lines)


# ---------------------------------------------------------------------
# Main audit
# ---------------------------------------------------------------------

def main() -> int:
    print_section("F1 AERODYNAMIC DATASET — PHASE 1 DATA AUDIT")

    # 1. Locate dataset
    print(f"Looking for dataset:\n  {DATA_PATH}")

    if not DATA_PATH.exists():
        print(
            "\nERROR: Dataset was not found.\n"
            "Expected location:\n"
            f"  {DATA_PATH}\n\n"
            "Place the CSV inside the project's data/ directory and run again."
        )
        return 1

    # 2. Load dataset
    print_section("1. LOADING DATASET")

    try:
        df = pd.read_csv(DATA_PATH)
    except Exception as exc:
        print(f"ERROR: Could not read CSV: {exc}")
        return 1

    print(f"Successfully loaded: {DATA_PATH.name}")

    # 3. Basic dimensions
    print_section("2. DATASET OVERVIEW")

    print(f"Rows:    {df.shape[0]:,}")
    print(f"Columns: {df.shape[1]:,}")
    print("\nColumns:")
    for i, col in enumerate(df.columns, start=1):
        print(f"  {i:2d}. {col}")

    # 4. Check expected columns
    print_section("3. COLUMN VALIDATION")
    check_expected_columns(df)
    print("Expected continuous input columns:")
    print("  " + ", ".join(INPUT_COLUMNS))
    print("\nExpected continuous target columns:")
    print("  " + ", ".join(TARGET_COLUMNS))

    # 5. Data types
    print_section("4. DATA TYPES")
    print(df.dtypes.to_string())

    # 6. Missing values
    print_section("5. MISSING VALUES")
    missing = df.isna().sum()
    missing_pct = (missing / len(df) * 100).round(4)

    missing_table = pd.DataFrame(
        {
            "missing_count": missing,
            "missing_percentage": missing_pct,
        }
    )

    print(missing_table.to_string())

    total_missing = int(missing.sum())
    print(f"\nTotal missing values: {total_missing:,}")

    # 7. Duplicate rows
    print_section("6. DUPLICATE ROWS")
    duplicate_count = int(df.duplicated().sum())
    print(f"Duplicate rows: {duplicate_count:,}")

    # 8. Unique values
    print_section("7. UNIQUE VALUE COUNTS")
    unique_counts = df.nunique(dropna=False)
    print(unique_counts.to_string())

    # 9. Numerical validation
    print_section("8. NUMERICAL / CONTINUOUS VALIDATION")

    non_numeric_columns = [
        col for col in EXPECTED_COLUMNS
        if col in df.columns and not pd.api.types.is_numeric_dtype(df[col])
    ]

    constant_columns = [
        col for col in EXPECTED_COLUMNS
        if col in df.columns and df[col].nunique(dropna=False) <= 1
    ]

    if not non_numeric_columns:
        print("All expected variables are numerical: PASS")
    else:
        print("Non-numeric variables: FAIL")
        for col in non_numeric_columns:
            print(f"  - {col}")

    if not constant_columns:
        print("No constant variables: PASS")
    else:
        print("Constant variables: WARNING")
        for col in constant_columns:
            print(f"  - {col}")

    # 10. Infinite values
    print_section("9. INFINITE VALUES")

    numeric_df = df.select_dtypes(include=np.number)
    infinite_counts = pd.Series(
        np.isinf(numeric_df.to_numpy()).sum(axis=0),
        index=numeric_df.columns,
    )

    print(infinite_counts.to_string())
    print(f"\nTotal infinite values: {int(infinite_counts.sum()):,}")

    # 11. Descriptive statistics
    print_section("10. DESCRIPTIVE STATISTICS")

    stats = df.describe().T
    print(stats.to_string())

    # 12. Basic plausibility
    print_section("11. BASIC PLAUSIBILITY CHECKS")

    plausibility_warnings = run_plausibility_checks(df)

    if plausibility_warnings:
        for warning in plausibility_warnings:
            print(f"WARNING: {warning}")
    else:
        print("No basic plausibility violations detected.")

    # 13. Correlation matrix
    print_section("12. CORRELATION MATRIX")

    correlation = df.corr(numeric_only=True)
    print(correlation.round(4).to_string())

    # 14. Per-column audit table
    print_section("13. COLUMN-BY-COLUMN AUDIT")

    audit_df = audit_continuous_variables(df)
    print(audit_df.to_string(index=False))

    # 15. Save outputs
    print_section("14. SAVING AUDIT RESULTS")

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    audit_csv = REPORT_DIR / "column_audit.csv"
    stats_csv = REPORT_DIR / "descriptive_statistics.csv"
    corr_csv = REPORT_DIR / "correlation_matrix.csv"
    missing_csv = REPORT_DIR / "missing_values.csv"
    report_txt = REPORT_DIR / "audit_report.txt"

    audit_df.to_csv(audit_csv, index=False)
    stats.to_csv(stats_csv)
    correlation.to_csv(corr_csv)
    missing_table.to_csv(missing_csv)

    report = build_report(
        df=df,
        audit_df=audit_df,
        correlation=correlation,
        plausibility_warnings=plausibility_warnings,
    )
    save_text_report(report_txt, report)

    print(f"Saved: {audit_csv}")
    print(f"Saved: {stats_csv}")
    print(f"Saved: {corr_csv}")
    print(f"Saved: {missing_csv}")
    print(f"Saved: {report_txt}")

    # 16. Final status
    print_section("15. AUDIT SUMMARY")

    total_missing = int(df.isna().sum().sum())
    total_infinite = int(infinite_counts.sum())

    checks = {
        "Dataset loaded": True,
        "Expected columns present": len(
            [c for c in EXPECTED_COLUMNS if c in df.columns]
        ) == len(EXPECTED_COLUMNS),
        "All expected variables numeric": len(non_numeric_columns) == 0,
        "No missing values": total_missing == 0,
        "No infinite values": total_infinite == 0,
        "No constant expected variables": len(constant_columns) == 0,
    }

    for name, passed in checks.items():
        status = "PASS" if passed else "FAIL"
        print(f"[{status}] {name}")

    if all(checks.values()):
        print(
            "\nDATA AUDIT COMPLETE: The dataset passes the structural checks "
            "required to proceed to exploratory analysis."
        )
        return 0

    print(
        "\nDATA AUDIT COMPLETE: One or more checks failed. "
        "Review the report before proceeding."
    )
    return 2


if __name__ == "__main__":
    sys.exit(main())
