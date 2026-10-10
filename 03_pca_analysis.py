"""
03_pca_analysis.py

Phase 3 of the F1 2026 Data-Driven Aerodynamic Simulator project.

Purpose
-------
Perform Principal Component Analysis (PCA) on the SIX CONTINUOUS INPUT
variables only.

The objective is not simply to reduce dimensionality. We want to answer:

    1. How many independent dimensions explain most of the operating space?
    2. Which physical variables dominate each principal component?
    3. Does the six-dimensional input space contain meaningful structure?
    4. Can PCA provide a useful representation for the next clustering stage?

Important
---------
- The source CSV is NOT modified.
- PCA is fitted only on the training subset to avoid information leakage.
- The same fitted scaler/PCA is then applied to the held-out test subset.
- Targets are NOT included in PCA.
- Aero mode is NOT present and is not used.
- All PCA inputs are continuous.

Expected input columns
----------------------
velocity_ms
flap_front_angle
flap_rear_angle
ride_height_front
ride_height_rear
battery_soc

Outputs
-------
reports/03_pca_analysis/
├── figures/
│   ├── explained_variance.png
│   ├── cumulative_explained_variance.png
│   ├── pca_scree_and_cumulative.png
│   ├── pca_loadings_heatmap.png
│   ├── pca_score_pc1_pc2.png
│   ├── pca_score_pc1_pc3.png
│   ├── pca_score_pc2_pc3.png
│   └── pca_feature_contribution.png
│
├── tables/
│   ├── pca_explained_variance.csv
│   ├── pca_loadings.csv
│   ├── pca_absolute_loadings.csv
│   ├── pca_feature_contribution.csv
│   ├── pca_scores_sample.csv
│   └── train_test_pca_summary.csv
│
└── pca_summary.md

Run
---
python 03_pca_analysis.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# =====================================================================
# Configuration
# =====================================================================

PROJECT_ROOT = Path(__file__).resolve().parent

DATA_PATH = PROJECT_ROOT / "data" / "F1_2026_Synthetic_Dataset_Mode1.csv"

REPORT_DIR = PROJECT_ROOT / "reports" / "03_pca_analysis"
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

RANDOM_STATE = 42
TEST_SIZE = 0.20

# Number of PCA components shown in score-space plots.
# This does NOT decide the final number of components.
SCORE_PLOT_COMPONENTS = 3

# A conventional threshold for identifying a compact representation.
# We will still report the full curve rather than forcing this threshold.
CUMULATIVE_VARIANCE_TARGET = 0.95


# =====================================================================
# Utility functions
# =====================================================================

def print_section(title: str) -> None:
    """Print a clear section heading."""
    print("\n" + "=" * 82)
    print(title)
    print("=" * 82)


def ensure_output_directories() -> None:
    """Create output folders."""
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)


def save_figure(fig: plt.Figure, filename: str) -> None:
    """Save a presentation-quality figure."""
    path = FIGURE_DIR / filename
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure: {path}")


def save_table(
    table: pd.DataFrame,
    filename: str,
    index: bool = True,
) -> None:
    """Save a DataFrame as CSV."""
    path = TABLE_DIR / filename
    table.to_csv(path, index=index)
    print(f"Saved table: {path}")


def dataframe_to_markdown_table(
    table: pd.DataFrame,
    index: bool = False,
    float_digits: int = 4,
) -> str:
    """
    Convert a DataFrame to a simple Markdown table without requiring
    pandas' optional 'tabulate' dependency.
    """
    work = table.copy()

    if index:
        work = work.reset_index()

    # Format numbers consistently while keeping text unchanged.
    for column in work.columns:
        if pd.api.types.is_numeric_dtype(work[column]):
            work[column] = work[column].map(
                lambda value: (
                    f"{value:.{float_digits}f}"
                    if pd.notna(value)
                    else ""
                )
            )
        else:
            work[column] = work[column].astype(str)

    headers = [str(col) for col in work.columns]

    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]

    for _, row in work.iterrows():
        values = [str(value).replace("|", r"\|") for value in row]
        lines.append("| " + " | ".join(values) + " |")

    return "\\n".join(lines)


def check_required_columns(df: pd.DataFrame) -> None:
    """Check that all six PCA input columns exist and are numeric."""
    missing = [col for col in INPUT_COLUMNS if col not in df.columns]

    if missing:
        raise ValueError(
            "Required PCA input columns are missing:\n"
            + "\n".join(f"  - {col}" for col in missing)
        )

    non_numeric = [
        col
        for col in INPUT_COLUMNS
        if not pd.api.types.is_numeric_dtype(df[col])
    ]

    if non_numeric:
        raise TypeError(
            "PCA inputs must be continuous numerical variables. "
            "Non-numeric columns found:\n"
            + "\n".join(f"  - {col}" for col in non_numeric)
        )


# =====================================================================
# PCA calculations
# =====================================================================

def fit_pca(
    df: pd.DataFrame,
) -> tuple[
    StandardScaler,
    PCA,
    np.ndarray,
    np.ndarray,
]:
    """
    Split the data, standardize inputs, and fit PCA on training data only.

    Returns
    -------
    scaler
        Fitted StandardScaler.
    pca
        Fitted PCA object.
    X_train_pca
        PCA scores for the training set.
    X_test_pca
        PCA scores for the test set.
    """
    X = df[INPUT_COLUMNS].to_numpy(dtype=float)

    X_train, X_test = train_test_split(
        X,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    pca = PCA()
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)

    return scaler, pca, X_train_pca, X_test_pca


def build_explained_variance_table(pca: PCA) -> pd.DataFrame:
    """Build a component-level explained-variance table."""
    n_components = len(pca.explained_variance_ratio_)

    component_names = [
        f"PC{i + 1}" for i in range(n_components)
    ]

    table = pd.DataFrame(
        {
            "component": component_names,
            "explained_variance": pca.explained_variance_,
            "explained_variance_ratio": pca.explained_variance_ratio_,
            "explained_variance_percent":
                pca.explained_variance_ratio_ * 100.0,
            "cumulative_variance_ratio":
                np.cumsum(pca.explained_variance_ratio_),
            "cumulative_variance_percent":
                np.cumsum(pca.explained_variance_ratio_) * 100.0,
        }
    )

    return table


def build_loading_table(pca: PCA) -> pd.DataFrame:
    """
    Return PCA loadings.

    For standardized variables, sklearn's PCA components multiplied by the
    square root of the corresponding eigenvalue give loadings. Loadings are
    easier to interpret as the contribution of each standardized input to a
    principal component.
    """
    loadings = (
        pca.components_.T
        * np.sqrt(pca.explained_variance_)
    )

    table = pd.DataFrame(
        loadings,
        index=INPUT_COLUMNS,
        columns=[
            f"PC{i + 1}"
            for i in range(pca.n_components_)
        ],
    )

    return table


def build_absolute_loading_table(
    loading_table: pd.DataFrame,
) -> pd.DataFrame:
    """Return absolute PCA loadings."""
    return loading_table.abs()


def build_feature_contribution_table(
    loading_table: pd.DataFrame,
) -> pd.DataFrame:
    """
    Summarize each feature's contribution across components.

    The contribution for feature j is based on squared standardized loading
    divided by the sum of squared loadings across all variables in that PC,
    multiplied by the variance explained by the PC.

    This produces a useful global ranking of feature influence in PCA space.
    """
    loading_squared = loading_table**2

    # Normalize feature contributions within each component.
    component_contribution = loading_squared.div(
        loading_squared.sum(axis=0),
        axis=1,
    )

    explained = np.array(
        [
            float(value)
            for value in []
        ]
    )

    # Extract explained ratios from the loading/eigenvalue relationship.
    # The sum of squared loadings for each PC equals its eigenvalue.
    # We reconstruct normalized explained weights below from the columns.
    total_variance = float(loading_squared.sum(axis=1).sum())

    # A robust alternative is to weight each PC by its share of total
    # eigenvalue/variance. Since standardized six-variable PCA has total
    # variance equal to six, the PC weight is represented by its eigenvalue
    # divided by total eigenvalue.
    eigenvalue_by_pc = loading_squared.sum(axis=0).to_numpy()
    pc_weights = eigenvalue_by_pc / eigenvalue_by_pc.sum()

    weighted_contribution = component_contribution.mul(
        pc_weights,
        axis=1,
    )

    feature_table = pd.DataFrame(
        {
            "feature": INPUT_COLUMNS,
            "global_pca_contribution": weighted_contribution.sum(axis=1),
            "global_pca_contribution_percent":
                weighted_contribution.sum(axis=1) * 100.0,
        }
    ).sort_values(
        "global_pca_contribution_percent",
        ascending=False,
    )

    return feature_table


def number_of_components_for_threshold(
    explained_variance_ratio: np.ndarray,
    threshold: float,
) -> int:
    """Return the smallest component count reaching the variance threshold."""
    cumulative = np.cumsum(explained_variance_ratio)

    indices = np.where(cumulative >= threshold)[0]

    if len(indices) == 0:
        return len(explained_variance_ratio)

    return int(indices[0] + 1)


# =====================================================================
# Visualizations
# =====================================================================

def plot_explained_variance(
    variance_table: pd.DataFrame,
) -> None:
    """Plot variance explained by each principal component."""
    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(1, len(variance_table) + 1)
    y = variance_table["explained_variance_percent"]

    bars = ax.bar(
        x,
        y,
        alpha=0.85,
    )

    ax.set_xticks(x)
    ax.set_xticklabels(
        variance_table["component"],
        rotation=0,
    )

    ax.set_xlabel("Principal component")
    ax.set_ylabel("Explained variance (%)")
    ax.set_title(
        "Variance Explained by Principal Components",
        fontsize=16,
    )
    ax.grid(axis="y", alpha=0.2)

    # Label values without cluttering the chart.
    for bar, value in zip(bars, y):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(y) * 0.01,
            f"{value:.1f}%",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    save_figure(fig, "explained_variance.png")


def plot_cumulative_variance(
    variance_table: pd.DataFrame,
) -> None:
    """Plot cumulative explained variance."""
    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(1, len(variance_table) + 1)
    cumulative = variance_table["cumulative_variance_percent"]

    ax.plot(
        x,
        cumulative,
        marker="o",
        linewidth=2,
    )

    ax.axhline(
        CUMULATIVE_VARIANCE_TARGET * 100,
        linestyle="--",
        linewidth=1.2,
        label=f"{CUMULATIVE_VARIANCE_TARGET * 100:.0f}% target",
    )

    ax.set_xticks(x)
    ax.set_xticklabels(
        variance_table["component"],
    )

    ax.set_ylim(0, 105)
    ax.set_xlabel("Number of principal components")
    ax.set_ylabel("Cumulative explained variance (%)")
    ax.set_title(
        "Cumulative Explained Variance",
        fontsize=16,
    )
    ax.grid(alpha=0.2)
    ax.legend()

    save_figure(fig, "cumulative_explained_variance.png")


def plot_scree_and_cumulative(
    variance_table: pd.DataFrame,
) -> None:
    """Create a combined scree/cumulative diagnostic figure."""
    fig, ax1 = plt.subplots(figsize=(10, 6))

    x = np.arange(1, len(variance_table) + 1)

    ax1.plot(
        x,
        variance_table["explained_variance_percent"],
        marker="o",
        linewidth=2,
        label="Individual variance",
    )

    ax1.set_xlabel("Principal component")
    ax1.set_ylabel("Explained variance (%)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(variance_table["component"])
    ax1.grid(alpha=0.2)

    ax2 = ax1.twinx()

    ax2.plot(
        x,
        variance_table["cumulative_variance_percent"],
        marker="s",
        linestyle="--",
        linewidth=2,
        label="Cumulative variance",
    )

    ax2.axhline(
        CUMULATIVE_VARIANCE_TARGET * 100,
        linestyle=":",
        linewidth=1.2,
    )

    ax2.set_ylabel("Cumulative explained variance (%)")
    ax2.set_ylim(0, 105)

    ax1.set_title(
        "PCA Scree Plot and Cumulative Variance",
        fontsize=16,
    )

    save_figure(fig, "pca_scree_and_cumulative.png")


def plot_loading_heatmap(
    loading_table: pd.DataFrame,
) -> None:
    """Plot PCA loadings as a heatmap."""
    fig, ax = plt.subplots(figsize=(11, 6))

    sns.heatmap(
        loading_table,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        linewidths=0.4,
        ax=ax,
        cbar_kws={"label": "PCA loading"},
    )

    ax.set_title(
        "PCA Loadings — Continuous Vehicle/Aero Inputs",
        fontsize=16,
    )

    ax.set_xlabel("Principal component")
    ax.set_ylabel("Input variable")

    save_figure(fig, "pca_loadings_heatmap.png")


def plot_feature_contribution(
    feature_contribution: pd.DataFrame,
) -> None:
    """Plot overall PCA-space contribution of each input."""
    table = feature_contribution.sort_values(
        "global_pca_contribution_percent"
    )

    fig, ax = plt.subplots(figsize=(9, 5.5))

    bars = ax.barh(
        table["feature"],
        table["global_pca_contribution_percent"],
    )

    ax.set_xlabel("Weighted contribution to PCA space (%)")
    ax.set_ylabel("Input variable")
    ax.set_title(
        "Overall Feature Contribution to PCA Representation",
        fontsize=16,
    )
    ax.grid(axis="x", alpha=0.2)

    for bar, value in zip(
        bars,
        table["global_pca_contribution_percent"],
    ):
        ax.text(
            value + 0.3,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.1f}%",
            va="center",
            fontsize=9,
        )

    save_figure(fig, "pca_feature_contribution.png")


def plot_pca_scores(
    X_train_pca: np.ndarray,
    loading_table: pd.DataFrame,
) -> None:
    """
    Plot PCA score spaces for PC1/PC2, PC1/PC3, and PC2/PC3.

    These are unsupervised score plots. The points are not coloured by target
    values because this stage is intended to understand the input geometry.
    """
    sample_size = min(10000, len(X_train_pca))

    rng = np.random.default_rng(RANDOM_STATE)
    indices = rng.choice(
        len(X_train_pca),
        size=sample_size,
        replace=False,
    )

    scores = X_train_pca[indices]

    pairs = [
        (0, 1, "pca_score_pc1_pc2.png", "PC1", "PC2"),
        (0, 2, "pca_score_pc1_pc3.png", "PC1", "PC3"),
        (1, 2, "pca_score_pc2_pc3.png", "PC2", "PC3"),
    ]

    for i, j, filename, xlabel, ylabel in pairs:
        fig, ax = plt.subplots(figsize=(9, 7))

        ax.scatter(
            scores[:, i],
            scores[:, j],
            s=9,
            alpha=0.16,
        )

        ax.axhline(0, linewidth=0.8, alpha=0.5)
        ax.axvline(0, linewidth=0.8, alpha=0.5)

        ax.set_xlabel(
            f"{xlabel} score",
        )
        ax.set_ylabel(
            f"{ylabel} score",
        )
        ax.set_title(
            f"PCA Score Space: {xlabel} vs {ylabel}",
            fontsize=16,
        )
        ax.grid(alpha=0.2)

        save_figure(fig, filename)


# =====================================================================
# Report generation
# =====================================================================

def create_markdown_report(
    df: pd.DataFrame,
    scaler: StandardScaler,
    pca: PCA,
    variance_table: pd.DataFrame,
    loading_table: pd.DataFrame,
    feature_contribution: pd.DataFrame,
    train_test_summary: pd.DataFrame,
) -> None:
    """Create a concise but detailed PCA report."""
    report_path = REPORT_DIR / "pca_summary.md"

    n_95 = number_of_components_for_threshold(
        pca.explained_variance_ratio_,
        CUMULATIVE_VARIANCE_TARGET,
    )

    lines = [
        "# F1 2026 Mode-1 Dataset — PCA Analysis",
        "",
        "## Objective",
        "",
        "PCA was applied to the six continuous input variables only. "
        "The purpose is to understand the dimensional structure of the "
        "vehicle/aerodynamic operating space before clustering and supervised "
        "regression.",
        "",
        "## PCA inputs",
        "",
    ]

    for feature in INPUT_COLUMNS:
        lines.append(f"- `{feature}`")

    lines.extend(
        [
            "",
            "## Method",
            "",
            f"- Dataset size: {len(df):,} observations",
            f"- Training fraction: {1 - TEST_SIZE:.0%}",
            f"- Test fraction: {TEST_SIZE:.0%}",
            f"- Random state: {RANDOM_STATE}",
            "- Standardization: `StandardScaler`",
            "- PCA fitting: training set only",
            f"- Variance threshold reported: {CUMULATIVE_VARIANCE_TARGET:.0%}",
            "",
            "## Standardization parameters",
            "",
            "The PCA was fitted after standardizing each continuous input "
            "using training-set statistics.",
            "",
            "| Feature | Mean | Standard deviation |",
            "|---|---:|---:|",
        ]
    )

    for feature, mean, scale in zip(
        INPUT_COLUMNS,
        scaler.mean_,
        scaler.scale_,
    ):
        lines.append(
            f"| `{feature}` | {mean:.6f} | {scale:.6f} |"
        )

    lines.extend(
        [
            "",
            "## Explained variance",
            "",
            f"The first `{n_95}` principal component(s) are sufficient to "
            f"reach approximately {CUMULATIVE_VARIANCE_TARGET:.0%} cumulative "
            "explained variance in the fitted training representation.",
            "",
            "| Component | Variance explained (%) | Cumulative (%) |",
            "|---|---:|---:|",
        ]
    )

    for _, row in variance_table.iterrows():
        lines.append(
            f"| {row['component']} | "
            f"{row['explained_variance_percent']:.4f} | "
            f"{row['cumulative_variance_percent']:.4f} |"
        )

    lines.extend(
        [
            "",
            "## PCA loadings",
            "",
            "Loadings indicate how strongly each standardized input "
            "contributes to each principal component. The sign gives the "
            "direction of the relationship; the magnitude indicates "
            "strength within the component.",
            "",
        ]
    )

    loading_for_report = loading_table.copy()
    loading_for_report.insert(
        0,
        "feature",
        loading_for_report.index,
    )
    loading_for_report.index = range(len(loading_for_report))

    lines.append(
        dataframe_to_markdown_table(
            loading_for_report,
            index=False,
            float_digits=4,
        )
    )

    lines.extend(
        [
            "",
            "## Overall feature contribution",
            "",
        ]
    )

    lines.append(
        dataframe_to_markdown_table(
            feature_contribution,
            index=False,
            float_digits=4,
        )
    )

    lines.extend(
        [
            "",
            "## Train/test PCA score summary",
            "",
        ]
    )

    lines.append(
        dataframe_to_markdown_table(
            train_test_summary,
            index=False,
            float_digits=5,
        )
    )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "The PCA analysis should be used to understand the geometry of "
            "the continuous operating space rather than to replace the "
            "original variables automatically. A reduced representation "
            "should only be used downstream when it preserves the information "
            "needed for the intended clustering or modelling task.",
            "",
            "The next step is to investigate whether the PCA score space "
            "contains distinct operating regimes using clustering. The "
            "clustering stage should be evaluated quantitatively rather than "
            "assuming that a particular number of regimes exists.",
        ]
    )

    report_path.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )

    print(f"Saved report: {report_path}")


# =====================================================================
# Main
# =====================================================================

def main() -> int:
    print_section(
        "F1 2026 AERODYNAMIC SIMULATOR — PHASE 3 PCA ANALYSIS"
    )

    if not DATA_PATH.exists():
        print(
            "ERROR: Dataset not found.\n"
            f"Expected path: {DATA_PATH}"
        )
        return 1

    ensure_output_directories()

    # ---------------------------------------------------------------
    # Load data
    # ---------------------------------------------------------------
    print_section("1. LOADING DATASET")

    try:
        df = pd.read_csv(DATA_PATH)
    except Exception as exc:
        print(f"ERROR: Could not read dataset: {exc}")
        return 1

    try:
        check_required_columns(df)
    except (ValueError, TypeError) as exc:
        print(f"ERROR: {exc}")
        return 1

    print(f"Loaded: {DATA_PATH.name}")
    print(f"Shape: {df.shape[0]:,} rows x {df.shape[1]} columns")

    # ---------------------------------------------------------------
    # Input audit before PCA
    # ---------------------------------------------------------------
    print_section("2. PCA INPUT VARIABLES")

    for col in INPUT_COLUMNS:
        series = df[col]
        print(
            f"{col:22s} | "
            f"min={series.min():.6f} | "
            f"max={series.max():.6f} | "
            f"mean={series.mean():.6f} | "
            f"std={series.std():.6f}"
        )

    if df[INPUT_COLUMNS].isnull().any().any():
        print("ERROR: Missing values exist in PCA inputs.")
        return 1

    if np.isinf(df[INPUT_COLUMNS].to_numpy()).any():
        print("ERROR: Infinite values exist in PCA inputs.")
        return 1

    # ---------------------------------------------------------------
    # Fit PCA
    # ---------------------------------------------------------------
    print_section("3. STANDARDIZATION + PCA")

    scaler, pca, X_train_pca, X_test_pca = fit_pca(df)

    print(
        f"Training observations: {X_train_pca.shape[0]:,}"
    )
    print(
        f"Testing observations:  {X_test_pca.shape[0]:,}"
    )
    print(
        f"Number of PCA components: {pca.n_components_}"
    )

    # ---------------------------------------------------------------
    # Explained variance
    # ---------------------------------------------------------------
    variance_table = build_explained_variance_table(pca)

    n_95 = number_of_components_for_threshold(
        pca.explained_variance_ratio_,
        CUMULATIVE_VARIANCE_TARGET,
    )

    print_section("4. EXPLAINED VARIANCE")

    print(
        variance_table[
            [
                "component",
                "explained_variance_percent",
                "cumulative_variance_percent",
            ]
        ].round(4).to_string(index=False)
    )

    print(
        f"\nComponents required for "
        f"{CUMULATIVE_VARIANCE_TARGET * 100:.0f}% variance: {n_95}"
    )

    save_table(
        variance_table.round(8),
        "pca_explained_variance.csv",
        index=False,
    )

    # ---------------------------------------------------------------
    # Loadings
    # ---------------------------------------------------------------
    loading_table = build_loading_table(pca)
    absolute_loading_table = build_absolute_loading_table(
        loading_table
    )

    print_section("5. PCA LOADINGS")

    print(
        loading_table.round(4).to_string()
    )

    save_table(
        loading_table.round(8),
        "pca_loadings.csv",
        index=True,
    )

    save_table(
        absolute_loading_table.round(8),
        "pca_absolute_loadings.csv",
        index=True,
    )

    # ---------------------------------------------------------------
    # Global feature contribution
    # ---------------------------------------------------------------
    feature_contribution = build_feature_contribution_table(
        loading_table
    )

    print_section("6. OVERALL FEATURE CONTRIBUTION")

    print(
        feature_contribution.round(4).to_string(index=False)
    )

    save_table(
        feature_contribution.round(8),
        "pca_feature_contribution.csv",
        index=False,
    )

    # ---------------------------------------------------------------
    # PCA score summaries
    # ---------------------------------------------------------------
    print_section("7. PCA SCORE SUMMARY")

    score_columns = [
        f"PC{i + 1}"
        for i in range(pca.n_components_)
    ]

    train_score_summary = pd.DataFrame(
        {
            "set": "train",
            "component": score_columns,
            "mean": X_train_pca.mean(axis=0),
            "std": X_train_pca.std(axis=0),
            "min": X_train_pca.min(axis=0),
            "max": X_train_pca.max(axis=0),
        }
    )

    test_score_summary = pd.DataFrame(
        {
            "set": "test",
            "component": score_columns,
            "mean": X_test_pca.mean(axis=0),
            "std": X_test_pca.std(axis=0),
            "min": X_test_pca.min(axis=0),
            "max": X_test_pca.max(axis=0),
        }
    )

    train_test_summary = pd.concat(
        [
            train_score_summary,
            test_score_summary,
        ],
        ignore_index=True,
    )

    print(
        train_test_summary.round(5).to_string(index=False)
    )

    save_table(
        train_test_summary.round(8),
        "train_test_pca_summary.csv",
        index=False,
    )

    # ---------------------------------------------------------------
    # Save a score sample for inspection
    # ---------------------------------------------------------------
    rng = np.random.default_rng(RANDOM_STATE)

    sample_size = min(10000, len(X_train_pca))
    sample_indices = rng.choice(
        len(X_train_pca),
        size=sample_size,
        replace=False,
    )

    score_sample = pd.DataFrame(
        X_train_pca[sample_indices],
        columns=score_columns,
    )

    save_table(
        score_sample.round(8),
        "pca_scores_sample.csv",
        index=False,
    )

    # ---------------------------------------------------------------
    # Figures
    # ---------------------------------------------------------------
    print_section("8. GENERATING PCA FIGURES")

    plot_explained_variance(
        variance_table,
    )

    plot_cumulative_variance(
        variance_table,
    )

    plot_scree_and_cumulative(
        variance_table,
    )

    plot_loading_heatmap(
        loading_table,
    )

    plot_feature_contribution(
        feature_contribution,
    )

    if pca.n_components_ >= SCORE_PLOT_COMPONENTS:
        plot_pca_scores(
            X_train_pca,
            loading_table,
        )
    else:
        print(
            "Skipping PC1/PC3 and PC2/PC3 plots because fewer than "
            f"{SCORE_PLOT_COMPONENTS} components exist."
        )

    # ---------------------------------------------------------------
    # Markdown report
    # ---------------------------------------------------------------
    print_section("9. WRITING PCA REPORT")

    create_markdown_report(
        df=df,
        scaler=scaler,
        pca=pca,
        variance_table=variance_table,
        loading_table=loading_table,
        feature_contribution=feature_contribution,
        train_test_summary=train_test_summary,
    )

    # ---------------------------------------------------------------
    # Completion
    # ---------------------------------------------------------------
    print_section("PHASE 3 COMPLETE")

    print(
        f"PCA figures saved to:\n  {FIGURE_DIR}\n\n"
        f"PCA tables saved to:\n  {TABLE_DIR}\n\n"
        f"PCA report saved to:\n  {REPORT_DIR / 'pca_summary.md'}"
    )

    print(
        "\nNext step: Phase 4 — clustering of the PCA representation."
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
