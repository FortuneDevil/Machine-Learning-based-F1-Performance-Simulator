"""
04_nonlinear_structure.py

Phase 4 of the F1 2026 Data-Driven Aerodynamic Simulator project.

Purpose
-------
Investigate nonlinear structure and interactions in the six-dimensional
continuous operating space BEFORE final predictive modelling.

Why this phase exists
---------------------
The audit showed that the six inputs are almost mutually uncorrelated.
PCA consequently found little useful dimensionality reduction.

That does NOT mean the inputs are unimportant.

A variable can be nearly independent of the other inputs and still have a
strong nonlinear relationship with a target. This phase therefore asks:

    1. Which inputs are important to each continuous target?
    2. Does Pearson correlation miss nonlinear/monotonic structure?
    3. Do nonlinear models capture substantial additional predictive power?
    4. Which pairs of inputs interact strongly?
    5. What does the Cd-Cl aerodynamic trade-off look like?
    6. Which regions of the operating space are difficult to model?
    7. Is there evidence for meaningful operating regimes in OUTPUT space?

This script is exploratory. It does not fit the final production models.

Expected continuous inputs
--------------------------
velocity_ms
flap_front_angle
flap_rear_angle
ride_height_front
ride_height_rear
battery_soc

Expected continuous targets
---------------------------
cl_total
cd_total
cop_shift
energy_drain_kw

Methods used
------------
- Pearson correlation
- Spearman correlation
- Mutual information
- Linear regression baseline
- Extra Trees preliminary nonlinear model
- Permutation importance
- 2D response surfaces / binned means
- Pairwise interaction screening
- Drag vs downforce trade-off analysis
- Residual analysis
- Quantile-bin error analysis

Important methodological choices
---------------------------------
- The source CSV is NEVER modified.
- Train/test split is performed before supervised diagnostics.
- Mutual information is estimated on the training set only.
- Preliminary predictive models are trained on training data only.
- Test-set metrics are used only for comparison.
- No target is used as an input to the simulator features.
- Aero mode is not used.

Outputs
-------
reports/04_nonlinear_structure/
├── figures/
├── tables/
└── nonlinear_structure_summary.md

Run
---
python 04_nonlinear_structure.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Iterable

import matplotlib
matplotlib.use("Agg")  # Headless backend: prevents Tkinter GUI/thread crashes.

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.ensemble import ExtraTreesRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import mutual_info_regression


# =====================================================================
# Configuration
# =====================================================================

PROJECT_ROOT = Path(__file__).resolve().parent

DATA_PATH = PROJECT_ROOT / "data" / "F1_2026_Synthetic_Dataset_Mode1.csv"

REPORT_DIR = PROJECT_ROOT / "reports" / "04_nonlinear_structure"
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

RANDOM_STATE = 42
TEST_SIZE = 0.20

# A moderate model is enough here. This is a structural diagnostic, not the
# final hyperparameter-tuned model.
N_ESTIMATORS = 250

# Number of quantile bins used for response surfaces.
N_BINS_2D = 30

# Number of bins used for one-dimensional nonlinear diagnostics.
N_BINS_1D = 30

# Number of rows used for dense plotting.
PLOT_SAMPLE_SIZE = 15000

# Number of rows used by permutation importance. Using a subset prevents
# unnecessary runtime while preserving stable rankings for a 100k dataset.
PERMUTATION_SAMPLE_SIZE = 12000

# Number of repetitions for permutation importance.
PERMUTATION_REPEATS = 5


# =====================================================================
# Utility functions
# =====================================================================

def print_section(title: str) -> None:
    """Print a clear section heading."""
    print("\n" + "=" * 84)
    print(title)
    print("=" * 84)


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
    """Save a DataFrame to CSV."""
    path = TABLE_DIR / filename
    table.to_csv(path, index=index)
    print(f"Saved table: {path}")


def dataframe_to_markdown_table(
    table: pd.DataFrame,
    index: bool = False,
    float_digits: int = 4,
) -> str:
    """
    Convert a DataFrame to Markdown without requiring the optional
    'tabulate' package.
    """
    work = table.copy()

    if index:
        work = work.reset_index()

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
        values = [
            str(value).replace("|", r"\|")
            for value in row
        ]
        lines.append("| " + " | ".join(values) + " |")

    return "\n".join(lines)


def check_required_columns(df: pd.DataFrame) -> None:
    """Check expected continuous inputs/targets."""
    expected = INPUT_COLUMNS + TARGET_COLUMNS

    missing = [col for col in expected if col not in df.columns]

    if missing:
        raise ValueError(
            "Required columns are missing:\n"
            + "\n".join(f"  - {col}" for col in missing)
        )

    non_numeric = [
        col
        for col in expected
        if not pd.api.types.is_numeric_dtype(df[col])
    ]

    if non_numeric:
        raise TypeError(
            "All modelling variables must be continuous numerical "
            "variables. Non-numeric columns found:\n"
            + "\n".join(f"  - {col}" for col in non_numeric)
        )


def finite_matrix(df: pd.DataFrame, columns: list[str]) -> np.ndarray:
    """Return a finite floating-point matrix for selected columns."""
    values = df[columns].to_numpy(dtype=float)

    if not np.isfinite(values).all():
        raise ValueError(
            f"Non-finite values found in columns: {columns}"
        )

    return values


def sample_dataframe(
    df: pd.DataFrame,
    size: int,
    random_state: int = RANDOM_STATE,
) -> pd.DataFrame:
    """Return a deterministic sample or the full frame if smaller."""
    n = min(size, len(df))
    return df.sample(
        n=n,
        random_state=random_state,
    )


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate root mean squared error."""
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


# =====================================================================
# 1. Pearson / Spearman / Mutual Information
# =====================================================================

def build_dependence_table(
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculate several measures of input-target dependence.

    Pearson:
        linear association.

    Spearman:
        monotonic association, including nonlinear monotonic relations.

    Mutual information:
        more general statistical dependence measure.
    """
    rows = []

    for target in TARGET_COLUMNS:
        y = y_train[target].to_numpy(dtype=float)

        mi_values = mutual_info_regression(
            X_train,
            y,
            random_state=RANDOM_STATE,
        )

        for feature, mi in zip(INPUT_COLUMNS, mi_values):
            pearson = X_train[feature].corr(y_train[target], method="pearson")
            spearman = X_train[feature].corr(
                y_train[target],
                method="spearman",
            )

            pearson_abs = abs(float(pearson))
            spearman_abs = abs(float(spearman))

            rows.append(
                {
                    "target": target,
                    "feature": feature,
                    "pearson_r": float(pearson),
                    "spearman_r": float(spearman),
                    "absolute_pearson": pearson_abs,
                    "absolute_spearman": spearman_abs,
                    "mutual_information": float(mi),
                    "spearman_minus_pearson_abs":
                        spearman_abs - pearson_abs,
                }
            )

    table = pd.DataFrame(rows)

    table["rank_by_mutual_information"] = (
        table.groupby("target")["mutual_information"]
        .rank(
            ascending=False,
            method="min",
        )
    )

    table["rank_by_absolute_spearman"] = (
        table.groupby("target")["absolute_spearman"]
        .rank(
            ascending=False,
            method="min",
        )
    )

    return table.sort_values(
        ["target", "mutual_information"],
        ascending=[True, False],
    )


def plot_dependence_heatmaps(
    dependence_table: pd.DataFrame,
) -> None:
    """Plot dependence measures for all inputs and targets."""
    print_section("1. DEPENDENCE ANALYSIS")

    for measure, label, filename in [
        (
            "absolute_pearson",
            "|Pearson r|",
            "dependence_absolute_pearson.png",
        ),
        (
            "absolute_spearman",
            "|Spearman r|",
            "dependence_absolute_spearman.png",
        ),
        (
            "mutual_information",
            "Mutual information",
            "dependence_mutual_information.png",
        ),
    ]:
        pivot = dependence_table.pivot(
            index="feature",
            columns="target",
            values=measure,
        )

        fig, ax = plt.subplots(figsize=(10, 7))

        sns.heatmap(
            pivot,
            annot=True,
            fmt=".3f",
            cmap="viridis",
            linewidths=0.4,
            ax=ax,
        )

        ax.set_title(
            f"Input–Target Dependence: {label}",
            fontsize=16,
        )
        ax.set_xlabel("Continuous target")
        ax.set_ylabel("Continuous input")

        save_figure(fig, filename)


def plot_dependence_rank_comparison(
    dependence_table: pd.DataFrame,
) -> None:
    """
    Show where nonlinear/monotonic measures change the ranking relative to
    Pearson correlation.
    """
    for target in TARGET_COLUMNS:
        sub = dependence_table[
            dependence_table["target"] == target
        ].copy()

        sub = sub.sort_values(
            "mutual_information",
            ascending=True,
        )

        fig, ax = plt.subplots(figsize=(10, 5.5))

        y = np.arange(len(sub))
        width = 0.36

        ax.barh(
            y - width / 2,
            sub["absolute_pearson"],
            height=width,
            label="|Pearson r|",
        )

        # MI has no fixed upper bound comparable with correlation, so it is
        # displayed only as a ranking score below. This bar plot therefore
        # intentionally compares normalized MI.
        mi = sub["mutual_information"].to_numpy(dtype=float)
        mi_norm = (
            mi / mi.max()
            if np.max(mi) > 0
            else mi
        )

        ax.barh(
            y + width / 2,
            mi_norm,
            height=width,
            label="Normalized mutual information",
        )

        ax.set_yticks(y)
        ax.set_yticklabels(sub["feature"])
        ax.set_xlabel("Association strength / normalized MI")
        ax.set_title(
            f"Linear vs General Dependence: {target}",
            fontsize=16,
        )
        ax.legend()
        ax.grid(axis="x", alpha=0.2)

        save_figure(
            fig,
            f"dependence_comparison_{target}.png",
        )


# =====================================================================
# 2. Preliminary linear vs nonlinear predictive comparison
# =====================================================================

def fit_preliminary_models(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.DataFrame,
    y_test: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compare a linear baseline with Extra Trees.

    This is a structural diagnostic, not the final modelling stage.
    """
    print_section("2. LINEAR VS NONLINEAR PREDICTIVE STRUCTURE")

    rows = []

    for target in TARGET_COLUMNS:
        ytr = y_train[target]
        yte = y_test[target]

        linear = LinearRegression()
        linear.fit(X_train, ytr)

        linear_pred = linear.predict(X_test)

        nonlinear = ExtraTreesRegressor(
            n_estimators=N_ESTIMATORS,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            max_features=1.0,
        )

        nonlinear.fit(X_train, ytr)
        nonlinear_pred = nonlinear.predict(X_test)

        linear_r2 = r2_score(yte, linear_pred)
        nonlinear_r2 = r2_score(yte, nonlinear_pred)

        linear_rmse = rmse(yte, linear_pred)
        nonlinear_rmse = rmse(yte, nonlinear_pred)

        linear_mae = mean_absolute_error(yte, linear_pred)
        nonlinear_mae = mean_absolute_error(yte, nonlinear_pred)

        rows.extend(
            [
                {
                    "target": target,
                    "model": "Multiple Linear Regression",
                    "r2": linear_r2,
                    "rmse": linear_rmse,
                    "mae": linear_mae,
                },
                {
                    "target": target,
                    "model": "Extra Trees",
                    "r2": nonlinear_r2,
                    "rmse": nonlinear_rmse,
                    "mae": nonlinear_mae,
                },
            ]
        )

        print(f"\nTarget: {target}")
        print(
            f"  Linear     -> "
            f"R2={linear_r2:.6f}, "
            f"RMSE={linear_rmse:.6f}, "
            f"MAE={linear_mae:.6f}"
        )
        print(
            f"  ExtraTrees -> "
            f"R2={nonlinear_r2:.6f}, "
            f"RMSE={nonlinear_rmse:.6f}, "
            f"MAE={nonlinear_mae:.6f}"
        )

        # Actual vs prediction plots for the structural comparison.
        sample_indices = np.random.default_rng(
            RANDOM_STATE
        ).choice(
            len(yte),
            size=min(10000, len(yte)),
            replace=False,
        )

        actual = yte.to_numpy()[sample_indices]

        fig, ax = plt.subplots(figsize=(7.5, 7.5))

        ax.scatter(
            actual,
            linear_pred[sample_indices],
            s=8,
            alpha=0.18,
            label=f"Linear R²={linear_r2:.4f}",
        )

        ax.scatter(
            actual,
            nonlinear_pred[sample_indices],
            s=8,
            alpha=0.18,
            label=f"Extra Trees R²={nonlinear_r2:.4f}",
        )

        min_value = min(
            actual.min(),
            linear_pred[sample_indices].min(),
            nonlinear_pred[sample_indices].min(),
        )
        max_value = max(
            actual.max(),
            linear_pred[sample_indices].max(),
            nonlinear_pred[sample_indices].max(),
        )

        ax.plot(
            [min_value, max_value],
            [min_value, max_value],
            linestyle="--",
            linewidth=1.5,
        )

        ax.set_xlabel(f"Actual {target}")
        ax.set_ylabel(f"Predicted {target}")
        ax.set_title(
            f"Linear vs Nonlinear Prediction: {target}",
            fontsize=15,
        )
        ax.legend()
        ax.grid(alpha=0.2)

        save_figure(
            fig,
            f"linear_vs_nonlinear_{target}.png",
        )

    results = pd.DataFrame(rows)

    save_table(
        results.round(8),
        "linear_vs_nonlinear_model_comparison.csv",
        index=False,
    )

    return results


# =====================================================================
# 3. One-dimensional nonlinear response diagnostics
# =====================================================================

def plot_binned_response(
    df: pd.DataFrame,
    feature: str,
    target: str,
    filename: str,
    title: str,
) -> None:
    """Plot quantile-binned mean target against a continuous input."""
    work = df[[feature, target]].dropna().copy()

    work["bin"] = pd.qcut(
        work[feature],
        q=N_BINS_1D,
        duplicates="drop",
    )

    grouped = (
        work.groupby("bin", observed=True)
        .agg(
            feature_mean=(feature, "mean"),
            target_mean=(target, "mean"),
            target_std=(target, "std"),
            count=(target, "size"),
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

    # The ±1 standard deviation band gives a sense of the spread at each
    # input region without trying to interpret it as a confidence interval.
    lower = grouped["target_mean"] - grouped["target_std"]
    upper = grouped["target_mean"] + grouped["target_std"]

    ax.fill_between(
        grouped["feature_mean"],
        lower,
        upper,
        alpha=0.12,
        label="±1 standard deviation",
    )

    ax.set_xlabel(feature)
    ax.set_ylabel(f"Mean {target}")
    ax.set_title(title, fontsize=16)
    ax.grid(alpha=0.2)
    ax.legend()

    save_figure(fig, filename)


def generate_key_nonlinear_response_plots(df: pd.DataFrame) -> None:
    """Generate physically motivated one-dimensional response curves."""
    print_section("3. ONE-DIMENSIONAL NONLINEAR RESPONSE")

    selected_relationships = [
        (
            "velocity_ms",
            "energy_drain_kw",
            "binned_velocity_energy.png",
            "Velocity vs Energy Drain",
        ),
        (
            "velocity_ms",
            "cd_total",
            "binned_velocity_cd.png",
            "Velocity vs Drag Coefficient",
        ),
        (
            "velocity_ms",
            "cl_total",
            "binned_velocity_cl.png",
            "Velocity vs Lift Coefficient",
        ),
        (
            "flap_front_angle",
            "cl_total",
            "binned_front_flap_cl.png",
            "Front Flap Angle vs Lift Coefficient",
        ),
        (
            "flap_rear_angle",
            "cl_total",
            "binned_rear_flap_cl.png",
            "Rear Flap Angle vs Lift Coefficient",
        ),
        (
            "flap_front_angle",
            "cd_total",
            "binned_front_flap_cd.png",
            "Front Flap Angle vs Drag Coefficient",
        ),
        (
            "flap_rear_angle",
            "cd_total",
            "binned_rear_flap_cd.png",
            "Rear Flap Angle vs Drag Coefficient",
        ),
        (
            "ride_height_front",
            "cl_total",
            "binned_front_ride_cl.png",
            "Front Ride Height vs Lift Coefficient",
        ),
        (
            "ride_height_rear",
            "cl_total",
            "binned_rear_ride_cl.png",
            "Rear Ride Height vs Lift Coefficient",
        ),
        (
            "flap_rear_angle",
            "cop_shift",
            "binned_rear_flap_cop.png",
            "Rear Flap Angle vs Centre-of-Pressure Shift",
        ),
    ]

    for feature, target, filename, title in selected_relationships:
        plot_binned_response(
            df,
            feature,
            target,
            filename,
            title,
        )


# =====================================================================
# 4. Two-dimensional response surfaces
# =====================================================================

def build_binned_surface(
    df: pd.DataFrame,
    x_feature: str,
    y_feature: str,
    target: str,
    n_bins: int = N_BINS_2D,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build a 2D quantile-binned mean response surface.

    Returns:
        surface values
        x/y bin midpoint information
    """
    work = df[
        [
            x_feature,
            y_feature,
            target,
        ]
    ].dropna().copy()

    work["x_bin"] = pd.qcut(
        work[x_feature],
        q=n_bins,
        duplicates="drop",
    )

    work["y_bin"] = pd.qcut(
        work[y_feature],
        q=n_bins,
        duplicates="drop",
    )

    work["x_mid"] = work["x_bin"].map(
        lambda interval: interval.mid
    )
    work["y_mid"] = work["y_bin"].map(
        lambda interval: interval.mid
    )

    grouped = (
        work.groupby(
            ["y_bin", "x_bin"],
            observed=True,
        )
        .agg(
            target_mean=(target, "mean"),
            count=(target, "size"),
        )
        .reset_index()
    )

    grouped["x_mid"] = grouped["x_bin"].map(
        lambda interval: interval.mid
    )
    grouped["y_mid"] = grouped["y_bin"].map(
        lambda interval: interval.mid
    )

    surface = grouped.pivot(
        index="y_mid",
        columns="x_mid",
        values="target_mean",
    )

    return surface, grouped


def plot_response_surface(
    df: pd.DataFrame,
    x_feature: str,
    y_feature: str,
    target: str,
    filename: str,
    title: str,
) -> None:
    """Plot a 2D binned response surface."""
    surface, grouped = build_binned_surface(
        df,
        x_feature,
        y_feature,
        target,
    )

    fig, ax = plt.subplots(figsize=(11, 8))

    mesh = sns.heatmap(
        surface,
        cmap="viridis",
        ax=ax,
        cbar_kws={"label": target},
    )

    ax.set_xlabel(x_feature)
    ax.set_ylabel(y_feature)
    ax.set_title(title, fontsize=16)

    save_figure(fig, filename)

    # Save the numerical surface as a table so that later modelling or
    # presentation work can reuse the values.
    safe_x = x_feature.replace("/", "_")
    safe_y = y_feature.replace("/", "_")
    safe_target = target.replace("/", "_")

    save_table(
        surface.round(8),
        f"surface_{safe_x}_{safe_y}_to_{safe_target}.csv",
        index=True,
    )

    save_table(
        grouped.round(8),
        f"surface_points_{safe_x}_{safe_y}_to_{safe_target}.csv",
        index=False,
    )


def generate_response_surfaces(df: pd.DataFrame) -> None:
    """Generate selected aerodynamic interaction surfaces."""
    print_section("4. TWO-DIMENSIONAL RESPONSE SURFACES")

    surfaces = [
        (
            "flap_front_angle",
            "flap_rear_angle",
            "cd_total",
            "surface_flaps_to_cd.png",
            "Drag Coefficient Response Surface: Front vs Rear Flap",
        ),
        (
            "flap_front_angle",
            "flap_rear_angle",
            "cl_total",
            "surface_flaps_to_cl.png",
            "Lift Coefficient Response Surface: Front vs Rear Flap",
        ),
        (
            "ride_height_front",
            "ride_height_rear",
            "cd_total",
            "surface_ride_heights_to_cd.png",
            "Drag Coefficient Response Surface: Front vs Rear Ride Height",
        ),
        (
            "ride_height_front",
            "ride_height_rear",
            "cl_total",
            "surface_ride_heights_to_cl.png",
            "Lift Coefficient Response Surface: Front vs Rear Ride Height",
        ),
        (
            "flap_rear_angle",
            "ride_height_rear",
            "cd_total",
            "surface_rear_flap_rear_ride_to_cd.png",
            "Drag Response: Rear Flap vs Rear Ride Height",
        ),
        (
            "flap_rear_angle",
            "ride_height_rear",
            "cl_total",
            "surface_rear_flap_rear_ride_to_cl.png",
            "Lift Response: Rear Flap vs Rear Ride Height",
        ),
    ]

    for (
        x_feature,
        y_feature,
        target,
        filename,
        title,
    ) in surfaces:
        plot_response_surface(
            df,
            x_feature,
            y_feature,
            target,
            filename,
            title,
        )


# =====================================================================
# 5. Pairwise interaction screening
# =====================================================================

def calculate_pairwise_interaction_scores(
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
) -> pd.DataFrame:
    """
    Screen pairs of continuous inputs for interaction effects.

    Approach
    --------
    For each feature pair (x1, x2), create a simple rank-based interaction
    feature:

        interaction = z(x1) * z(x2)

    Then measure how strongly this interaction relates to the target.

    This is an exploratory screening statistic, not a causal estimate and
    not a replacement for a full interaction model.
    """
    rows = []

    X = X_train.copy()

    standardized = (
        X - X.mean()
    ) / X.std(ddof=0)

    for i, feature_a in enumerate(INPUT_COLUMNS):
        for feature_b in INPUT_COLUMNS[i + 1:]:
            interaction = (
                standardized[feature_a]
                * standardized[feature_b]
            )

            for target in TARGET_COLUMNS:
                y = y_train[target]

                pearson_interaction = interaction.corr(
                    y,
                    method="pearson",
                )

                spearman_interaction = interaction.corr(
                    y,
                    method="spearman",
                )

                rows.append(
                    {
                        "target": target,
                        "feature_a": feature_a,
                        "feature_b": feature_b,
                        "interaction_feature":
                            f"{feature_a} * {feature_b}",
                        "interaction_pearson_r":
                            float(pearson_interaction),
                        "interaction_absolute_r":
                            abs(float(pearson_interaction)),
                        "interaction_spearman_r":
                            float(spearman_interaction),
                        "interaction_absolute_spearman":
                            abs(float(spearman_interaction)),
                    }
                )

    result = pd.DataFrame(rows)

    result = result.sort_values(
        [
            "target",
            "interaction_absolute_spearman",
        ],
        ascending=[True, False],
    )

    return result


def plot_top_interactions(
    interaction_table: pd.DataFrame,
) -> None:
    """Plot the strongest interaction-screening terms for each target."""
    print_section("5. INTERACTION SCREENING")

    for target in TARGET_COLUMNS:
        sub = interaction_table[
            interaction_table["target"] == target
        ].head(6).copy()

        sub = sub.sort_values(
            "interaction_absolute_spearman"
        )

        fig, ax = plt.subplots(figsize=(10, 5.5))

        ax.barh(
            sub["interaction_feature"],
            sub["interaction_absolute_spearman"],
        )

        ax.set_xlabel("|Spearman r| of interaction term with target")
        ax.set_ylabel("Continuous feature pair")
        ax.set_title(
            f"Strongest Pairwise Interaction Candidates: {target}",
            fontsize=15,
        )
        ax.grid(axis="x", alpha=0.2)

        save_figure(
            fig,
            f"top_interactions_{target}.png",
        )


# =====================================================================
# 6. Cd-Cl trade-off
# =====================================================================

def analyze_drag_downforce_tradeoff(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Analyze the empirical relationship between Cd and Cl.

    This is particularly important to the eventual simulator because the
    vehicle trades aerodynamic load against drag.
    """
    sample = sample_dataframe(
        df,
        PLOT_SAMPLE_SIZE,
    )

    correlation = df["cl_total"].corr(
        df["cd_total"],
        method="pearson",
    )

    spearman = df["cl_total"].corr(
        df["cd_total"],
        method="spearman",
    )

    print_section("6. DRAG / DOWNFORCE TRADE-OFF")

    print(
        f"Pearson correlation, Cl vs Cd:   {correlation:.6f}"
    )
    print(
        f"Spearman correlation, Cl vs Cd:  {spearman:.6f}"
    )

    fig, ax = plt.subplots(figsize=(10, 7))

    scatter = ax.scatter(
        sample["cd_total"],
        sample["cl_total"],
        c=sample["flap_rear_angle"],
        s=10,
        alpha=0.25,
    )

    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label("Rear flap angle")

    ax.set_xlabel("Drag coefficient, Cd")
    ax.set_ylabel("Lift coefficient, Cl")
    ax.set_title(
        "Aerodynamic Trade-off: Downforce vs Drag",
        fontsize=16,
    )
    ax.grid(alpha=0.2)

    save_figure(
        fig,
        "cl_vs_cd_tradeoff.png",
    )

    # Calculate a simple empirical efficiency indicator.
    enriched = df[["cl_total", "cd_total"]].copy()
    enriched["cl_cd_ratio"] = (
        enriched["cl_total"]
        / enriched["cd_total"]
    )

    summary = enriched["cl_cd_ratio"].describe().to_frame(
        name="cl_cd_ratio"
    )

    save_table(
        summary.round(8),
        "cl_cd_ratio_statistics.csv",
    )

    # Quantile-bin the Cd axis and calculate mean Cl.
    enriched["cd_bin"] = pd.qcut(
        enriched["cd_total"],
        q=40,
        duplicates="drop",
    )

    tradeoff = (
        enriched.groupby(
            "cd_bin",
            observed=True,
        )
        .agg(
            cd_mean=("cd_total", "mean"),
            cl_mean=("cl_total", "mean"),
            cl_std=("cl_total", "std"),
            count=("cl_total", "size"),
        )
        .reset_index(drop=True)
    )

    save_table(
        tradeoff.round(8),
        "cl_cd_tradeoff_binned.csv",
        index=False,
    )

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(
        tradeoff["cd_mean"],
        tradeoff["cl_mean"],
        marker="o",
        markersize=4,
        linewidth=1.8,
    )

    ax.set_xlabel("Mean Cd")
    ax.set_ylabel("Mean Cl")
    ax.set_title(
        "Mean Downforce vs Mean Drag Across Operating Space",
        fontsize=16,
    )
    ax.grid(alpha=0.2)

    save_figure(
        fig,
        "cl_cd_binned_tradeoff.png",
    )

    return tradeoff


# =====================================================================
# 7. Preliminary nonlinear feature importance
# =====================================================================

def calculate_permutation_importance(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.DataFrame,
    y_test: pd.DataFrame,
) -> pd.DataFrame:
    """
    Estimate feature importance using test-set permutation importance.

    Each target receives a separately fitted Extra Trees model.
    """
    print_section("7. NONLINEAR FEATURE IMPORTANCE")

    rows = []

    sample_n = min(PERMUTATION_SAMPLE_SIZE, len(X_test))

    rng = np.random.default_rng(RANDOM_STATE)
    indices = rng.choice(
        len(X_test),
        size=sample_n,
        replace=False,
    )

    X_test_sample = X_test.iloc[indices].copy()

    for target in TARGET_COLUMNS:
        model = ExtraTreesRegressor(
            n_estimators=N_ESTIMATORS,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            max_features=1.0,
        )

        model.fit(
            X_train,
            y_train[target],
        )

        result = permutation_importance(
            model,
            X_test_sample,
            y_test[target].iloc[indices],
            n_repeats=PERMUTATION_REPEATS,
            random_state=RANDOM_STATE,
            scoring="r2",
            n_jobs=1,
        )

        for feature, mean, std in zip(
            INPUT_COLUMNS,
            result.importances_mean,
            result.importances_std,
        ):
            rows.append(
                {
                    "target": target,
                    "feature": feature,
                    "permutation_importance_mean": float(mean),
                    "permutation_importance_std": float(std),
                }
            )

    table = pd.DataFrame(rows)

    table = table.sort_values(
        [
            "target",
            "permutation_importance_mean",
        ],
        ascending=[True, False],
    )

    save_table(
        table.round(8),
        "permutation_importance.csv",
        index=False,
    )

    for target in TARGET_COLUMNS:
        sub = table[
            table["target"] == target
        ].sort_values(
            "permutation_importance_mean"
        )

        fig, ax = plt.subplots(figsize=(9, 5.5))

        ax.barh(
            sub["feature"],
            sub["permutation_importance_mean"],
            xerr=sub["permutation_importance_std"],
            alpha=0.85,
        )

        ax.axvline(0, linewidth=0.8)

        ax.set_xlabel("Decrease in test R² after permutation")
        ax.set_ylabel("Continuous input")
        ax.set_title(
            f"Nonlinear Feature Importance: {target}",
            fontsize=15,
        )
        ax.grid(axis="x", alpha=0.2)

        save_figure(
            fig,
            f"permutation_importance_{target}.png",
        )

    return table


# =====================================================================
# 8. Residual/error structure
# =====================================================================

def analyze_residual_structure(
    X_test: pd.DataFrame,
    y_test: pd.DataFrame,
    y_predictions: dict[str, np.ndarray],
) -> pd.DataFrame:
    """
    Study how preliminary nonlinear-model errors vary across feature ranges.

    The objective is to identify regions where the surrogate may eventually
    need more modelling flexibility.
    """
    print_section("8. RESIDUAL / ERROR STRUCTURE")

    rows = []

    for target in TARGET_COLUMNS:
        y_true = y_test[target].to_numpy(dtype=float)
        pred = y_predictions[target]

        abs_error = np.abs(y_true - pred)

        for feature in INPUT_COLUMNS:
            values = X_test[feature].to_numpy(dtype=float)

            bin_edges = np.quantile(
                values,
                np.linspace(0, 1, 11),
            )

            bin_edges = np.unique(bin_edges)

            if len(bin_edges) < 3:
                continue

            bins = pd.cut(
                values,
                bins=bin_edges,
                include_lowest=True,
                duplicates="drop",
            )

            temp = pd.DataFrame(
                {
                    "bin": bins,
                    "feature": values,
                    "absolute_error": abs_error,
                }
            )

            grouped = (
                temp.groupby(
                    "bin",
                    observed=True,
                )
                .agg(
                    feature_mean=("feature", "mean"),
                    mean_absolute_error=(
                        "absolute_error",
                        "mean",
                    ),
                    median_absolute_error=(
                        "absolute_error",
                        "median",
                    ),
                    count=("absolute_error", "size"),
                )
                .reset_index(drop=True)
            )

            grouped["target"] = target
            grouped["feature_name"] = feature

            rows.append(grouped)

            fig, ax = plt.subplots(figsize=(10, 6))

            ax.plot(
                grouped["feature_mean"],
                grouped["mean_absolute_error"],
                marker="o",
                linewidth=1.8,
            )

            ax.set_xlabel(feature)
            ax.set_ylabel("Mean absolute prediction error")
            ax.set_title(
                f"Model Error vs {feature}: {target}",
                fontsize=15,
            )
            ax.grid(alpha=0.2)

            filename = (
                f"error_vs_{feature}_for_{target}.png"
            )

            save_figure(fig, filename)

    if rows:
        residual_table = pd.concat(
            rows,
            ignore_index=True,
        )

        save_table(
            residual_table.round(8),
            "error_structure_by_feature_region.csv",
            index=False,
        )

        return residual_table

    return pd.DataFrame()


# =====================================================================
# 9. Fit final preliminary models for residual analysis
# =====================================================================

def fit_preliminary_nonlinear_models(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.DataFrame,
) -> dict[str, np.ndarray]:
    """
    Fit one preliminary Extra Trees model per target and return test
    predictions.

    These models are intentionally simple and will NOT be reused as the final
    production models.
    """
    predictions: dict[str, np.ndarray] = {}

    for target in TARGET_COLUMNS:
        model = ExtraTreesRegressor(
            n_estimators=N_ESTIMATORS,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            max_features=1.0,
        )

        model.fit(
            X_train,
            y_train[target],
        )

        predictions[target] = model.predict(X_test)

    return predictions


# =====================================================================
# 10. Report
# =====================================================================

def create_markdown_report(
    df: pd.DataFrame,
    dependence_table: pd.DataFrame,
    model_comparison: pd.DataFrame,
    interaction_table: pd.DataFrame,
    tradeoff: pd.DataFrame,
    permutation_table: pd.DataFrame,
) -> None:
    """Create a detailed Markdown summary."""
    report_path = REPORT_DIR / "nonlinear_structure_summary.md"

    lines = [
        "# F1 2026 Mode-1 Dataset — Nonlinear Structure Analysis",
        "",
        "## Objective",
        "",
        "This phase investigates nonlinear dependence and interactions after "
        "PCA showed that the six continuous input variables do not have a "
        "useful low-dimensional linear representation.",
        "",
        "## Inputs",
        "",
    ]

    for feature in INPUT_COLUMNS:
        lines.append(f"- `{feature}`")

    lines.extend(
        [
            "",
            "## Targets",
            "",
        ]
    )

    for target in TARGET_COLUMNS:
        lines.append(f"- `{target}`")

    lines.extend(
        [
            "",
            "## 1. Dependence measures",
            "",
            "Pearson correlation measures linear association. Spearman "
            "correlation detects monotonic association even when it is "
            "nonlinear. Mutual information is a more general dependence "
            "measure.",
            "",
        ]
    )

    # Top 3 MI relationships per target.
    for target in TARGET_COLUMNS:
        top = dependence_table[
            dependence_table["target"] == target
        ].head(3)

        lines.append(f"### `{target}`")

        for _, row in top.iterrows():
            lines.append(
                f"- `{row['feature']}`: "
                f"Pearson r = {row['pearson_r']:.4f}, "
                f"Spearman r = {row['spearman_r']:.4f}, "
                f"MI = {row['mutual_information']:.4f}"
            )

        lines.append("")

    lines.extend(
        [
            "## 2. Linear vs nonlinear predictive structure",
            "",
        ]
    )

    lines.append(
        dataframe_to_markdown_table(
            model_comparison,
            index=False,
            float_digits=6,
        )
    )

    lines.extend(
        [
            "",
            "The nonlinear model is used here as a structural diagnostic. "
            "The final model-selection stage will perform proper hyperparameter "
            "comparison and model diagnostics.",
            "",
            "## 3. Strongest interaction candidates",
            "",
        ]
    )

    for target in TARGET_COLUMNS:
        top = interaction_table[
            interaction_table["target"] == target
        ].head(5)

        lines.append(f"### `{target}`")

        for _, row in top.iterrows():
            lines.append(
                f"- `{row['feature_a']} × {row['feature_b']}`: "
                f"|Spearman| = "
                f"{row['interaction_absolute_spearman']:.4f}"
            )

        lines.append("")

    lines.extend(
        [
            "## 4. Drag/downforce trade-off",
            "",
        ]
    )

    if not tradeoff.empty:
        lines.append(
            dataframe_to_markdown_table(
                tradeoff.head(12),
                index=False,
                float_digits=5,
            )
        )

    lines.extend(
        [
            "",
            "## 5. Nonlinear feature importance",
            "",
        ]
    )

    for target in TARGET_COLUMNS:
        top = permutation_table[
            permutation_table["target"] == target
        ].head(5)

        lines.append(f"### `{target}`")

        for _, row in top.iterrows():
            lines.append(
                f"- `{row['feature']}`: importance = "
                f"{row['permutation_importance_mean']:.6f} "
                f"± {row['permutation_importance_std']:.6f}"
            )

        lines.append("")

    lines.extend(
        [
            "## Interpretation guidance",
            "",
            "This phase should determine the structure that the final "
            "regression models need to capture. Strong mutual information, "
            "large differences between Pearson and Spearman association, "
            "important pairwise interaction terms, or a large performance "
            "gap between linear and nonlinear models are evidence that a "
            "purely additive linear model is insufficient.",
            "",
            "PCA was not forced into a dimensionality-reduction role because "
            "the input variables are largely independent. The original six "
            "continuous variables should therefore remain available to the "
            "supervised models.",
            "",
            "The next phase should select, train, and compare the final "
            "continuous-to-continuous regression models for Cd and Cl, with "
            "CoP shift and energy drain treated as secondary outputs.",
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
        "F1 2026 AERODYNAMIC SIMULATOR — PHASE 4"
    )

    if not DATA_PATH.exists():
        print(
            "ERROR: Dataset not found.\n"
            f"Expected path: {DATA_PATH}"
        )
        return 1

    ensure_output_directories()

    # ---------------------------------------------------------------
    # Load
    # ---------------------------------------------------------------
    print_section("LOADING DATASET")

    try:
        df = pd.read_csv(DATA_PATH)
    except Exception as exc:
        print(f"ERROR: Could not load dataset: {exc}")
        return 1

    try:
        check_required_columns(df)
    except (ValueError, TypeError) as exc:
        print(f"ERROR: {exc}")
        return 1

    print(f"Dataset: {DATA_PATH.name}")
    print(
        f"Shape:   {df.shape[0]:,} rows x "
        f"{df.shape[1]} columns"
    )

    # ---------------------------------------------------------------
    # Prepare train/test data
    # ---------------------------------------------------------------
    print_section("CREATING TRAIN / TEST SETS")

    X_train, X_test, y_train, y_test = train_test_split(
        df[INPUT_COLUMNS],
        df[TARGET_COLUMNS],
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )

    print(f"Training rows: {len(X_train):,}")
    print(f"Testing rows:  {len(X_test):,}")

    # ---------------------------------------------------------------
    # Dependence analysis
    # ---------------------------------------------------------------
    dependence_table = build_dependence_table(
        X_train,
        y_train,
    )

    save_table(
        dependence_table.round(8),
        "pearson_spearman_mutual_information.csv",
        index=False,
    )

    plot_dependence_heatmaps(
        dependence_table,
    )

    plot_dependence_rank_comparison(
        dependence_table,
    )

    # ---------------------------------------------------------------
    # Linear vs nonlinear prediction
    # ---------------------------------------------------------------
    model_comparison = fit_preliminary_models(
        X_train,
        X_test,
        y_train,
        y_test,
    )

    # ---------------------------------------------------------------
    # One-dimensional nonlinear structure
    # ---------------------------------------------------------------
    generate_key_nonlinear_response_plots(
        df,
    )

    # ---------------------------------------------------------------
    # Two-dimensional response surfaces
    # ---------------------------------------------------------------
    generate_response_surfaces(
        df,
    )

    # ---------------------------------------------------------------
    # Interaction screening
    # ---------------------------------------------------------------
    interaction_table = calculate_pairwise_interaction_scores(
        X_train,
        y_train,
    )

    save_table(
        interaction_table.round(8),
        "pairwise_interaction_screening.csv",
        index=False,
    )

    plot_top_interactions(
        interaction_table,
    )

    # ---------------------------------------------------------------
    # Cd-Cl trade-off
    # ---------------------------------------------------------------
    tradeoff = analyze_drag_downforce_tradeoff(
        df,
    )

    # ---------------------------------------------------------------
    # Permutation importance
    # ---------------------------------------------------------------
    permutation_table = calculate_permutation_importance(
        X_train,
        X_test,
        y_train,
        y_test,
    )

    # ---------------------------------------------------------------
    # Residual analysis
    # ---------------------------------------------------------------
    predictions = fit_preliminary_nonlinear_models(
        X_train,
        X_test,
        y_train,
    )

    analyze_residual_structure(
        X_test,
        y_test,
        predictions,
    )

    # ---------------------------------------------------------------
    # Report
    # ---------------------------------------------------------------
    print_section("WRITING NONLINEAR STRUCTURE REPORT")

    create_markdown_report(
        df=df,
        dependence_table=dependence_table,
        model_comparison=model_comparison,
        interaction_table=interaction_table,
        tradeoff=tradeoff,
        permutation_table=permutation_table,
    )

    # ---------------------------------------------------------------
    # Completion
    # ---------------------------------------------------------------
    print_section("PHASE 4 COMPLETE")

    print(
        f"Figures saved to:\n  {FIGURE_DIR}\n\n"
        f"Tables saved to:\n  {TABLE_DIR}\n\n"
        f"Report saved to:\n  {REPORT_DIR / 'nonlinear_structure_summary.md'}"
    )

    print(
        "\nNext step: Phase 5 — final continuous-to-continuous "
        "regression model selection."
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
