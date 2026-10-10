"""
02_data_exploration.py

Phase 2 of the F1 2026 Data-Driven Aerodynamic Simulator project.

Purpose
-------
Explore the cleaned Mode-1 dataset and understand the aerodynamic structure
before PCA, clustering, and predictive modelling.

This script DOES NOT modify the source CSV.

Outputs
-------
All figures and tables are written to:

    reports/02_exploratory_analysis/

Main analyses
-------------
1. Dataset overview
2. Distribution plots for all variables
3. Correlation matrix
4. Input -> target correlation ranking
5. Aerodynamic response maps:
      - front flap angle x rear flap angle -> Cd
      - front flap angle x rear flap angle -> Cl
      - front ride height x rear ride height -> Cd
      - front ride height x rear ride height -> Cl
6. Velocity vs Cd and Cl
7. Velocity vs derived aerodynamic drag/downforce
8. Derived drag/downforce summaries
9. Pairwise relationship matrix
10. Basic nonlinear relationship diagnostics using binned means

The project uses the following continuous variables:

Inputs
------
velocity_ms
flap_front_angle
flap_rear_angle
ride_height_front
ride_height_rear
battery_soc

Targets
-------
cl_total
cd_total
cop_shift
energy_drain_kw

Derived quantities
------------------
F_D = 0.5 * rho * Cd * A * v^2
F_L = 0.5 * rho * Cl * A * v^2

The values of rho and A are kept as clearly labelled simulator assumptions.
They are NOT learned from the dataset.

Run
---
python 02_data_exploration.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


# =====================================================================
# Configuration
# =====================================================================

PROJECT_ROOT = Path(__file__).resolve().parent

DATA_PATH = PROJECT_ROOT / "data" / "F1_2026_Synthetic_Dataset_Mode1.csv"
REPORT_DIR = PROJECT_ROOT / "reports" / "02_exploratory_analysis"
FIGURE_DIR = REPORT_DIR / "figures"
TABLE_DIR = REPORT_DIR / "tables"

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

ALL_COLUMNS = INPUT_COLUMNS + TARGET_COLUMNS

# ---------------------------------------------------------------------
# Simulator assumptions for derived quantities.
#
# These are intentionally explicit rather than hidden inside the code.
# They are engineering assumptions used only to convert coefficients
# into approximate forces for exploration/simulation.
# ---------------------------------------------------------------------

AIR_DENSITY = 1.225       # kg/m^3
FRONTAL_AREA = 1.5        # m^2


# =====================================================================
# Utility functions
# =====================================================================

def print_section(title: str) -> None:
    """Print a clear section heading."""
    print("\n" + "=" * 82)
    print(title)
    print("=" * 82)


def ensure_output_directories() -> None:
    """Create output directories if they do not exist."""
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)


def save_figure(fig: plt.Figure, filename: str) -> None:
    """Save a figure with consistent presentation-ready settings."""
    path = FIGURE_DIR / filename
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure: {path}")


def save_table(df: pd.DataFrame, filename: str, index: bool = True) -> None:
    """Save a DataFrame to CSV."""
    path = TABLE_DIR / filename
    df.to_csv(path, index=index)
    print(f"Saved table: {path}")


def check_required_columns(df: pd.DataFrame) -> None:
    """Ensure the expected modelling columns exist."""
    missing = [col for col in ALL_COLUMNS if col not in df.columns]

    if missing:
        raise ValueError(
            "The following required columns are missing:\n"
            + "\n".join(f"  - {col}" for col in missing)
        )


def add_derived_quantities(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add aerodynamic force quantities without modifying the original dataset.

    Returns
    -------
    pd.DataFrame
        Copy of the input DataFrame with derived columns.
    """
    out = df.copy()

    velocity = out["velocity_ms"]
    cd = out["cd_total"]
    cl = out["cl_total"]

    out["drag_force_N"] = (
        0.5
        * AIR_DENSITY
        * cd
        * FRONTAL_AREA
        * velocity**2
    )

    out["downforce_N"] = (
        0.5
        * AIR_DENSITY
        * cl
        * FRONTAL_AREA
        * velocity**2
    )

    out["drag_force_kN"] = out["drag_force_N"] / 1000.0
    out["downforce_kN"] = out["downforce_N"] / 1000.0

    # Aerodynamic efficiency: downforce generated per unit drag.
    # This is an interpretable diagnostic, not a model target.
    out["lift_to_drag_ratio"] = np.where(
        out["cd_total"] != 0,
        out["cl_total"] / out["cd_total"],
        np.nan,
    )

    return out


def make_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Return descriptive statistics for all original continuous columns."""
    summary = df[ALL_COLUMNS].describe().T
    summary["range"] = summary["max"] - summary["min"]
    summary["cv_percent"] = (
        np.where(
            summary["mean"] != 0,
            summary["std"].abs() / summary["mean"].abs() * 100,
            np.nan,
        )
    )
    return summary


# =====================================================================
# 1. Distributions
# =====================================================================

def plot_distributions(df: pd.DataFrame) -> None:
    """Create one distribution figure per variable."""
    print_section("1. VARIABLE DISTRIBUTIONS")

    for col in ALL_COLUMNS:
        fig, ax = plt.subplots(figsize=(9, 5.5))

        sns.histplot(
            data=df,
            x=col,
            bins=50,
            kde=True,
            ax=ax,
        )

        role = "Input" if col in INPUT_COLUMNS else "Target"

        ax.set_title(f"{role}: Distribution of {col}", fontsize=15)
        ax.set_xlabel(col)
        ax.set_ylabel("Count")
        ax.grid(alpha=0.2)

        save_figure(fig, f"distribution_{col}.png")


# =====================================================================
# 2. Correlation analysis
# =====================================================================

def plot_correlation_matrix(df: pd.DataFrame) -> None:
    """Plot the full Pearson correlation matrix."""
    print_section("2. FULL CORRELATION MATRIX")

    corr = df[ALL_COLUMNS].corr()

    save_table(corr.round(6), "correlation_matrix.csv")

    fig, ax = plt.subplots(figsize=(11, 9))

    sns.heatmap(
        corr,
        cmap="coolwarm",
        center=0,
        square=True,
        fmt=".2f",
        linewidths=0.4,
        ax=ax,
    )

    ax.set_title("Correlation Matrix — Mode 1", fontsize=16)
    save_figure(fig, "correlation_matrix.png")


def build_target_correlation_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate input-target Pearson correlations and rank each target.

    Positive/negative sign indicates direction of linear association.
    Absolute correlation is used for ranking strength.
    """
    rows = []

    for target in TARGET_COLUMNS:
        for feature in INPUT_COLUMNS:
            r = df[feature].corr(df[target])

            rows.append(
                {
                    "target": target,
                    "feature": feature,
                    "pearson_r": r,
                    "absolute_r": abs(r),
                }
            )

    result = pd.DataFrame(rows)
    result = result.sort_values(
        ["target", "absolute_r"],
        ascending=[True, False],
    )

    return result


def plot_target_correlation_ranking(df: pd.DataFrame) -> None:
    """Create horizontal bar charts showing input-target correlation strength."""
    print_section("3. INPUT -> TARGET CORRELATION RANKING")

    corr_table = build_target_correlation_table(df)
    save_table(
        corr_table.round(6),
        "input_target_correlation_ranking.csv",
        index=False,
    )

    for target in TARGET_COLUMNS:
        sub = corr_table[corr_table["target"] == target].copy()
        sub = sub.sort_values("pearson_r")

        fig, ax = plt.subplots(figsize=(9, 5.5))

        ax.barh(
            sub["feature"],
            sub["pearson_r"],
        )

        ax.axvline(0, linewidth=1)
        ax.set_title(f"Input Correlation with {target}", fontsize=15)
        ax.set_xlabel("Pearson correlation")
        ax.set_ylabel("Input variable")
        ax.grid(axis="x", alpha=0.2)

        save_figure(fig, f"correlation_ranking_{target}.png")


# =====================================================================
# 3. Aerodynamic response maps
# =====================================================================

def create_binned_response_map(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    target_col: str,
    filename: str,
    title: str,
    x_label: str | None = None,
    y_label: str | None = None,
) -> None:
    """
    Create a 2D heatmap from binned averages.

    Using binned means avoids a huge overplotted scatter cloud and makes
    the aerodynamic response surface easier to interpret.
    """
    work = df[[x_col, y_col, target_col]].dropna().copy()

    # Use quantile bins so that each region has a reasonable number of points.
    x_bins = pd.qcut(work[x_col], q=25, duplicates="drop")
    y_bins = pd.qcut(work[y_col], q=25, duplicates="drop")

    grouped = (
        work.assign(
            x_bin=x_bins,
            y_bin=y_bins,
        )
        .groupby(["y_bin", "x_bin"], observed=True)[target_col]
        .mean()
        .unstack()
    )

    # Label bins by their midpoint for a cleaner heatmap.
    grouped.columns = [
        f"{interval.mid:.3g}" for interval in grouped.columns
    ]
    grouped.index = [
        f"{interval.mid:.3g}" for interval in grouped.index
    ]

    fig, ax = plt.subplots(figsize=(11, 8))

    sns.heatmap(
        grouped,
        cmap="viridis",
        ax=ax,
        cbar_kws={"label": target_col},
    )

    ax.set_title(title, fontsize=16)
    ax.set_xlabel(x_label or x_col)
    ax.set_ylabel(y_label or y_col)

    save_figure(fig, filename)


def plot_aerodynamic_response_maps(df: pd.DataFrame) -> None:
    """Create the main two-variable aerodynamic response maps."""
    print_section("4. AERODYNAMIC RESPONSE MAPS")

    create_binned_response_map(
        df,
        "flap_front_angle",
        "flap_rear_angle",
        "cd_total",
        "response_front_rear_flap_to_cd.png",
        "Aerodynamic Drag Coefficient vs Front and Rear Flap Angles",
        "Front flap angle",
        "Rear flap angle",
    )

    create_binned_response_map(
        df,
        "flap_front_angle",
        "flap_rear_angle",
        "cl_total",
        "response_front_rear_flap_to_cl.png",
        "Lift Coefficient vs Front and Rear Flap Angles",
        "Front flap angle",
        "Rear flap angle",
    )

    create_binned_response_map(
        df,
        "ride_height_front",
        "ride_height_rear",
        "cd_total",
        "response_front_rear_ride_to_cd.png",
        "Aerodynamic Drag Coefficient vs Front and Rear Ride Heights",
        "Front ride height",
        "Rear ride height",
    )

    create_binned_response_map(
        df,
        "ride_height_front",
        "ride_height_rear",
        "cl_total",
        "response_front_rear_ride_to_cl.png",
        "Lift Coefficient vs Front and Rear Ride Heights",
        "Front ride height",
        "Rear ride height",
    )


# =====================================================================
# 4. Velocity relationships
# =====================================================================

def plot_velocity_vs_coefficients(df: pd.DataFrame) -> None:
    """Plot velocity against Cd and Cl using sampled points."""
    print_section("5. VELOCITY VS AERODYNAMIC COEFFICIENTS")

    # Deterministic sample keeps rendering manageable while preserving the
    # overall structure of the 100k-row dataset.
    sample = df.sample(
        n=min(15000, len(df)),
        random_state=42,
    )

    for target in ["cd_total", "cl_total"]:
        fig, ax = plt.subplots(figsize=(10, 6))

        sns.scatterplot(
            data=sample,
            x="velocity_ms",
            y=target,
            alpha=0.18,
            s=12,
            edgecolor=None,
            ax=ax,
        )

        ax.set_title(
            f"Velocity vs {target}",
            fontsize=16,
        )
        ax.set_xlabel("Velocity (m/s)")
        ax.set_ylabel(target)
        ax.grid(alpha=0.2)

        save_figure(fig, f"velocity_vs_{target}.png")


def plot_velocity_vs_forces(df: pd.DataFrame) -> None:
    """Plot velocity against derived aerodynamic drag and downforce."""
    print_section("6. VELOCITY VS DERIVED AERODYNAMIC FORCES")

    sample = df.sample(
        n=min(15000, len(df)),
        random_state=42,
    )

    # Add derived quantities only to this plotting copy.
    sample = add_derived_quantities(sample)

    for target, ylabel, title, filename in [
        (
            "drag_force_kN",
            "Drag force (kN)",
            "Velocity vs Aerodynamic Drag",
            "velocity_vs_drag_force.png",
        ),
        (
            "downforce_kN",
            "Downforce (kN)",
            "Velocity vs Aerodynamic Downforce",
            "velocity_vs_downforce.png",
        ),
    ]:
        fig, ax = plt.subplots(figsize=(10, 6))

        sns.scatterplot(
            data=sample,
            x="velocity_ms",
            y=target,
            alpha=0.18,
            s=12,
            edgecolor=None,
            ax=ax,
        )

        ax.set_title(title, fontsize=16)
        ax.set_xlabel("Velocity (m/s)")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.2)

        save_figure(fig, filename)


# =====================================================================
# 5. Derived force summary
# =====================================================================

def save_derived_force_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate and save summary statistics for derived force quantities."""
    print_section("7. DERIVED AERODYNAMIC QUANTITIES")

    enriched = add_derived_quantities(df)

    derived_columns = [
        "drag_force_N",
        "drag_force_kN",
        "downforce_N",
        "downforce_kN",
        "lift_to_drag_ratio",
    ]

    summary = enriched[derived_columns].describe().T
    save_table(summary.round(6), "derived_force_statistics.csv")

    print(summary.to_string())

    return enriched


# =====================================================================
# 6. Pairwise relationship matrix
# =====================================================================

def plot_pairwise_relationships(df: pd.DataFrame) -> None:
    """
    Create a compact pairwise plot for the most important variables.

    We deliberately use a selected subset rather than all ten variables so
    the figure remains interpretable.
    """
    print_section("8. PAIRWISE RELATIONSHIP ANALYSIS")

    selected = [
        "velocity_ms",
        "flap_front_angle",
        "flap_rear_angle",
        "ride_height_front",
        "ride_height_rear",
        "cl_total",
        "cd_total",
    ]

    sample = df[selected].sample(
        n=min(7000, len(df)),
        random_state=42,
    )

    # Pairplot is useful for broad structural inspection. This is not intended
    # to be the main presentation figure.
    grid = sns.pairplot(
        sample,
        corner=True,
        diag_kind="hist",
        plot_kws={
            "s": 8,
            "alpha": 0.18,
        },
        height=1.65,
    )

    grid.fig.suptitle(
        "Pairwise Relationships — Selected Aerodynamic Variables",
        y=1.02,
        fontsize=16,
    )

    path = FIGURE_DIR / "pairwise_relationships_selected.png"
    grid.fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(grid.fig)
    print(f"Saved figure: {path}")


# =====================================================================
# 7. Binned nonlinear diagnostics
# =====================================================================

def plot_binned_mean_relationship(
    df: pd.DataFrame,
    feature: str,
    target: str,
    filename: str,
    title: str,
    n_bins: int = 30,
) -> None:
    """
    Plot target mean within feature bins.

    This makes nonlinear trends easier to see than a dense scatter plot.
    """
    work = df[[feature, target]].dropna().copy()

    work["bin"] = pd.qcut(
        work[feature],
        q=n_bins,
        duplicates="drop",
    )

    grouped = (
        work.groupby("bin", observed=True)
        .agg(
            feature_mean=(feature, "mean"),
            target_mean=(target, "mean"),
            target_std=(target, "std"),
        )
        .reset_index(drop=True)
    )

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(
        grouped["feature_mean"],
        grouped["target_mean"],
        marker="o",
        markersize=4,
        linewidth=1.8,
    )

    ax.set_title(title, fontsize=16)
    ax.set_xlabel(feature)
    ax.set_ylabel(f"Mean {target}")
    ax.grid(alpha=0.2)

    save_figure(fig, filename)


def plot_key_nonlinear_relationships(df: pd.DataFrame) -> None:
    """Create binned-mean plots for physically interesting relationships."""
    print_section("9. NONLINEAR RELATIONSHIP DIAGNOSTICS")

    relationships = [
        (
            "velocity_ms",
            "cd_total",
            "binned_velocity_cd.png",
            "Binned Mean: Velocity vs Drag Coefficient",
        ),
        (
            "velocity_ms",
            "cl_total",
            "binned_velocity_cl.png",
            "Binned Mean: Velocity vs Lift Coefficient",
        ),
        (
            "ride_height_front",
            "cl_total",
            "binned_front_ride_cl.png",
            "Binned Mean: Front Ride Height vs Lift Coefficient",
        ),
        (
            "ride_height_rear",
            "cl_total",
            "binned_rear_ride_cl.png",
            "Binned Mean: Rear Ride Height vs Lift Coefficient",
        ),
        (
            "flap_front_angle",
            "cd_total",
            "binned_front_flap_cd.png",
            "Binned Mean: Front Flap Angle vs Drag Coefficient",
        ),
        (
            "flap_rear_angle",
            "cd_total",
            "binned_rear_flap_cd.png",
            "Binned Mean: Rear Flap Angle vs Drag Coefficient",
        ),
    ]

    for feature, target, filename, title in relationships:
        plot_binned_mean_relationship(
            df,
            feature,
            target,
            filename,
            title,
        )


# =====================================================================
# 8. Report
# =====================================================================

def create_exploration_report(
    df: pd.DataFrame,
    summary: pd.DataFrame,
    target_corr: pd.DataFrame,
) -> None:
    """Save a concise markdown report summarizing the EDA."""
    report_path = REPORT_DIR / "exploration_summary.md"

    lines = [
        "# F1 2026 Mode-1 Dataset — Exploratory Analysis Summary",
        "",
        "## Dataset",
        "",
        f"- Rows: {len(df):,}",
        f"- Original variables: {len(ALL_COLUMNS)}",
        "- Aero mode: fixed to Mode 1 before this analysis",
        "- Inputs: 6 continuous variables",
        "- Targets: 4 continuous variables",
        "",
        "## Continuous inputs",
        "",
    ]

    for col in INPUT_COLUMNS:
        lines.append(f"- `{col}`")

    lines.extend(
        [
            "",
            "## Continuous targets",
            "",
        ]
    )

    for col in TARGET_COLUMNS:
        lines.append(f"- `{col}`")

    lines.extend(
        [
            "",
            "## Derived quantities used only for analysis/simulation",
            "",
            f"- Air density assumption: `{AIR_DENSITY} kg/m^3`",
            f"- Frontal area assumption: `{FRONTAL_AREA} m^2`",
            "- Drag force: `F_D = 0.5 * rho * Cd * A * v^2`",
            "- Downforce: `F_L = 0.5 * rho * Cl * A * v^2`",
            "",
            "## Descriptive statistics",
            "",
        ]
    )

    lines.append("```text")
    lines.append(summary.round(5).to_string())
    lines.append("```")

    lines.extend(
        [
            "",
            "## Strongest linear input-target relationships",
            "",
        ]
    )

    for target in TARGET_COLUMNS:
        lines.append(f"### `{target}`")
        top = target_corr[
            target_corr["target"] == target
        ].head(3)

        for _, row in top.iterrows():
            lines.append(
                f"- `{row['feature']}`: Pearson r = "
                f"{row['pearson_r']:.4f}"
            )

        lines.append("")

    lines.extend(
        [
            "## Interpretation guidance",
            "",
            "The correlation analysis should be treated as a first-order "
            "diagnostic only. Aerodynamic relationships are expected to be "
            "nonlinear and interactive. The response maps and binned-mean "
            "plots therefore provide information that simple correlation "
            "cannot capture.",
            "",
            "The next project stage should use the six continuous input "
            "variables for PCA and exploratory clustering before final "
            "regression models are selected.",
        ]
    )

    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved report: {report_path}")


# =====================================================================
# Main
# =====================================================================

def main() -> int:
    print_section("F1 2026 AERODYNAMIC SIMULATOR — PHASE 2")

    if not DATA_PATH.exists():
        print(
            "ERROR: Dataset not found.\n"
            f"Expected path: {DATA_PATH}"
        )
        return 1

    ensure_output_directories()

    # -----------------------------------------------------------------
    # Load
    # -----------------------------------------------------------------
    print_section("LOADING DATASET")

    try:
        df = pd.read_csv(DATA_PATH)
    except Exception as exc:
        print(f"ERROR: Could not read dataset: {exc}")
        return 1

    try:
        check_required_columns(df)
    except ValueError as exc:
        print(f"ERROR: {exc}")
        return 1

    print(f"Loaded: {DATA_PATH.name}")
    print(f"Shape: {df.shape[0]:,} rows x {df.shape[1]} columns")

    # -----------------------------------------------------------------
    # Clean plotting environment only — source data is untouched.
    # -----------------------------------------------------------------
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 160)

    # -----------------------------------------------------------------
    # Basic overview
    # -----------------------------------------------------------------
    print_section("BASIC OVERVIEW")

    print("Input variables:")
    for col in INPUT_COLUMNS:
        print(f"  - {col}")

    print("\nTarget variables:")
    for col in TARGET_COLUMNS:
        print(f"  - {col}")

    print("\nMissing values:")
    print(df[ALL_COLUMNS].isna().sum().to_string())

    # -----------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------
    print_section("DESCRIPTIVE SUMMARY")

    summary = make_summary(df)
    print(summary.round(5).to_string())

    save_table(
        summary.round(6),
        "descriptive_statistics.csv",
    )

    # -----------------------------------------------------------------
    # Visual analysis
    # -----------------------------------------------------------------
    plot_distributions(df)
    plot_correlation_matrix(df)
    plot_target_correlation_ranking(df)
    plot_aerodynamic_response_maps(df)
    plot_velocity_vs_coefficients(df)
    plot_velocity_vs_forces(df)

    # -----------------------------------------------------------------
    # Derived quantities
    # -----------------------------------------------------------------
    enriched_df = save_derived_force_summary(df)

    # Save a compact derived sample, not the complete dataset.
    # This is only for inspection and avoids creating unnecessary large files.
    derived_sample = enriched_df[
        INPUT_COLUMNS
        + TARGET_COLUMNS
        + [
            "drag_force_N",
            "drag_force_kN",
            "downforce_N",
            "downforce_kN",
            "lift_to_drag_ratio",
        ]
    ].sample(
        n=min(10000, len(enriched_df)),
        random_state=42,
    )

    save_table(
        derived_sample.round(6),
        "derived_quantities_sample.csv",
        index=False,
    )

    # -----------------------------------------------------------------
    # Pairwise + nonlinear diagnostics
    # -----------------------------------------------------------------
    plot_pairwise_relationships(df)
    plot_key_nonlinear_relationships(df)

    # -----------------------------------------------------------------
    # Final report
    # -----------------------------------------------------------------
    target_corr = build_target_correlation_table(df)
    create_exploration_report(
        df,
        summary,
        target_corr,
    )

    # -----------------------------------------------------------------
    # Completion
    # -----------------------------------------------------------------
    print_section("PHASE 2 COMPLETE")

    print(
        "Exploratory analysis is complete.\n"
        f"Figures: {FIGURE_DIR}\n"
        f"Tables:  {TABLE_DIR}\n"
        f"Report:  {REPORT_DIR / 'exploration_summary.md'}"
    )

    print(
        "\nNext step: Phase 3 — PCA and unsupervised structure analysis."
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
