"""
06_surrogate_validation.py

Phase 6 of the F1 2026 Data-Driven Aerodynamic Simulator project.

Purpose
-------
Validate the saved machine-learning surrogate models independently before
they are connected to the aerodynamic/vehicle simulator.

The surrogate models were trained in Phase 5. This script does NOT retrain
them.

The objective is to answer:

    1. Do the saved models reproduce the target outputs on the untouched
       test set?
    2. Are the predictions stable across the full operating envelope?
    3. How large are errors in the derived aerodynamic forces?
    4. Are there particular operating regions where the surrogate fails?
    5. Are predicted outputs physically plausible relative to the training
       domain?
    6. Is the surrogate fast enough for repeated simulation calls?

Primary targets
---------------
    cl_total
    cd_total

Secondary targets
-----------------
    cop_shift
    energy_drain_kw

Continuous inputs
-----------------
    velocity_ms
    flap_front_angle
    flap_rear_angle
    ride_height_front
    ride_height_rear
    battery_soc

Saved models expected
---------------------
models/
├── cd_model.pkl
├── cl_model.pkl
├── cop_shift_model.pkl
└── energy_drain_kw_model.pkl

Each file is expected to contain the metadata bundle saved by
05_regression_models.py.

Important
---------
- The original CSV is NEVER modified.
- No model is retrained.
- The final test set is reproduced using the exact split procedure from
  Phase 5.
- All model inputs remain continuous.
- Aero mode is not used.
- This is validation, NOT simulation.

Derived quantities
------------------
Using the same assumptions used in earlier phases:

    rho = 1.225 kg/m^3
    A   = 1.5 m^2

    F_D = 0.5 * rho * Cd * A * v^2
    F_L = 0.5 * rho * Cl * A * v^2

The force comparison is simply the aerodynamic coefficient comparison
converted into physically interpretable quantities.

Outputs
-------
reports/06_surrogate_validation/
├── figures/
└── tables/
    surrogate_summary.md

Run
---
python 06_surrogate_validation.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

# Always use a non-GUI backend. This avoids Tkinter crashes when the script
# is run from a terminal.
import matplotlib
matplotlib.use("Agg")

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split


# =====================================================================
# Configuration
# =====================================================================

PROJECT_ROOT = Path(__file__).resolve().parent

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "F1_2026_Synthetic_Dataset_Mode1.csv"
)

MODEL_DIR = PROJECT_ROOT / "models"

REPORT_DIR = (
    PROJECT_ROOT
    / "reports"
    / "06_surrogate_validation"
)

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

MODEL_FILES = {
    "cl_total": "cl_model.pkl",
    "cd_total": "cd_model.pkl",
    "cop_shift": "cop_shift_model.pkl",
    "energy_drain_kw": "energy_drain_kw_model.pkl",
}

PRIMARY_TARGETS = [
    "cl_total",
    "cd_total",
]

RANDOM_STATE = 42

TRAIN_FRACTION = 0.70
VALIDATION_FRACTION = 0.15
TEST_FRACTION = 0.15

AIR_DENSITY = 1.225
FRONTAL_AREA = 1.5

PLOT_SAMPLE_SIZE = 12000

# Number of velocity / operating-envelope bins used for stress diagnostics.
N_ENVELOPE_BINS = 10

# Number of calls used when benchmarking prediction speed.
BENCHMARK_CALLS = 10000


# =====================================================================
# Utility functions
# =====================================================================

def print_section(title: str) -> None:
    """Print a clear section heading."""
    print("\n" + "=" * 86)
    print(title)
    print("=" * 86)


def ensure_output_directories() -> None:
    """Create report directories."""
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)


def save_figure(fig: plt.Figure, filename: str) -> None:
    """Save a figure using a consistent DPI and bounding box."""
    path = FIGURE_DIR / filename
    fig.savefig(
        path,
        dpi=220,
        bbox_inches="tight",
    )
    plt.close(fig)
    print(f"Saved figure: {path}")


def save_table(
    table: pd.DataFrame,
    filename: str,
    index: bool = False,
) -> None:
    """Save a DataFrame to CSV."""
    path = TABLE_DIR / filename
    table.to_csv(
        path,
        index=index,
    )
    print(f"Saved table: {path}")


def dataframe_to_markdown_table(
    table: pd.DataFrame,
    index: bool = False,
    float_digits: int = 6,
) -> str:
    """Convert a DataFrame to Markdown without requiring tabulate."""
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

    headers = [str(column) for column in work.columns]

    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]

    for _, row in work.iterrows():
        values = [
            str(value).replace("|", r"\|")
            for value in row
        ]
        lines.append(
            "| " + " | ".join(values) + " |"
        )

    return "\n".join(lines)


def safe_filename(text: str) -> str:
    """Create a filesystem-friendly string."""
    return (
        text.lower()
        .replace(" ", "_")
        .replace("/", "_")
        .replace("-", "_")
    )


def calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, float]:
    """Calculate common continuous regression metrics."""
    mse = mean_squared_error(
        y_true,
        y_pred,
    )

    return {
        "r2": float(r2_score(y_true, y_pred)),
        "rmse": float(np.sqrt(mse)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "max_absolute_error":
            float(np.max(np.abs(y_true - y_pred))),
    }


# =====================================================================
# Dataset checks / exact Phase 5 split
# =====================================================================

def check_required_columns(df: pd.DataFrame) -> None:
    """Ensure all continuous modelling columns exist and are numerical."""
    required = INPUT_COLUMNS + TARGET_COLUMNS

    missing = [
        column
        for column in required
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            "Missing required columns:\n"
            + "\n".join(
                f"  - {column}"
                for column in missing
            )
        )

    non_numeric = [
        column
        for column in required
        if not pd.api.types.is_numeric_dtype(df[column])
    ]

    if non_numeric:
        raise TypeError(
            "All modelling variables must be continuous numerical "
            "variables. Non-numeric columns found:\n"
            + "\n".join(
                f"  - {column}"
                for column in non_numeric
            )
        )

    values = df[required].to_numpy(
        dtype=float,
    )

    if not np.isfinite(values).all():
        raise ValueError(
            "Non-finite values found in modelling columns."
        )


def reproduce_phase5_test_split(
    df: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
]:
    """
    Reproduce exactly the final test set used in Phase 5.

    Phase 5 performed:
        70% train / 30% temporary
        50% of temporary validation / 50% test

    Therefore:
        70% train
        15% validation
        15% test
    """
    X = df[INPUT_COLUMNS].copy()
    y = df[TARGET_COLUMNS].copy()

    (
        X_train,
        X_temp,
        y_train,
        y_temp,
    ) = train_test_split(
        X,
        y,
        test_size=(1.0 - TRAIN_FRACTION),
        random_state=RANDOM_STATE,
    )

    temp_test_share = TEST_FRACTION / (
        VALIDATION_FRACTION + TEST_FRACTION
    )

    (
        X_validation,
        X_test,
        y_validation,
        y_test,
    ) = train_test_split(
        X_temp,
        y_temp,
        test_size=temp_test_share,
        random_state=RANDOM_STATE,
    )

    # Explicitly discard train/validation. They are not needed in this phase.
    del X_train, X_validation, y_train, y_validation

    return X_test, y_test


# =====================================================================
# Model loading
# =====================================================================

def load_saved_models() -> dict[str, object]:
    """
    Load saved Phase-5 model bundles and validate their metadata.
    """
    print_section("1. LOADING SAVED SURROGATE MODELS")

    models: dict[str, object] = {}

    for target, filename in MODEL_FILES.items():
        path = MODEL_DIR / filename

        if not path.exists():
            raise FileNotFoundError(
                f"Saved model not found:\n  {path}\n\n"
                "Run 05_regression_models.py first."
            )

        bundle = joblib.load(path)

        if not isinstance(bundle, dict):
            raise TypeError(
                f"Model file {path.name} does not contain the expected "
                "metadata dictionary."
            )

        for required_key in [
            "model",
            "target",
            "input_columns",
            "model_name",
        ]:
            if required_key not in bundle:
                raise KeyError(
                    f"{path.name} is missing expected metadata key: "
                    f"{required_key}"
                )

        if bundle["target"] != target:
            raise ValueError(
                f"Target mismatch in {path.name}: "
                f"expected {target}, got {bundle['target']}"
            )

        if list(bundle["input_columns"]) != INPUT_COLUMNS:
            raise ValueError(
                f"Input-column mismatch in {path.name}."
            )

        models[target] = bundle["model"]

        print(
            f"{target:20s} -> "
            f"{bundle['model_name']}"
        )

    return models


# =====================================================================
# Prediction
# =====================================================================

def generate_test_predictions(
    models: dict[str, object],
    X_test: pd.DataFrame,
) -> dict[str, np.ndarray]:
    """Generate predictions from saved models."""
    predictions: dict[str, np.ndarray] = {}

    print_section("2. GENERATING TEST-SET PREDICTIONS")

    for target in TARGET_COLUMNS:
        model = models[target]

        start = time.perf_counter()

        prediction = model.predict(
            X_test[INPUT_COLUMNS]
        )

        elapsed = time.perf_counter() - start

        prediction = np.asarray(
            prediction,
            dtype=float,
        )

        if not np.isfinite(prediction).all():
            raise ValueError(
                f"Non-finite predictions produced for {target}."
            )

        predictions[target] = prediction

        print(
            f"{target:20s} -> "
            f"{len(prediction):,} predictions in "
            f"{elapsed:.4f} s"
        )

    return predictions


# =====================================================================
# 1. Overall validation
# =====================================================================

def calculate_overall_validation(
    y_test: pd.DataFrame,
    predictions: dict[str, np.ndarray],
) -> pd.DataFrame:
    """Calculate final test metrics for all surrogate targets."""
    rows = []

    for target in TARGET_COLUMNS:
        y_true = y_test[target].to_numpy(
            dtype=float
        )

        metrics = calculate_metrics(
            y_true,
            predictions[target],
        )

        rows.append(
            {
                "target": target,
                "r2": metrics["r2"],
                "rmse": metrics["rmse"],
                "mae": metrics["mae"],
                "max_absolute_error":
                    metrics["max_absolute_error"],
            }
        )

    result = pd.DataFrame(rows)

    save_table(
        result.round(10),
        "overall_test_validation.csv",
    )

    return result


# =====================================================================
# 2. Actual vs predicted plots
# =====================================================================

def plot_actual_vs_predicted(
    target: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> None:
    """Create an actual-vs-predicted scatter plot."""
    rng = np.random.default_rng(
        RANDOM_STATE
    )

    n = min(
        PLOT_SAMPLE_SIZE,
        len(y_true),
    )

    indices = rng.choice(
        len(y_true),
        size=n,
        replace=False,
    )

    actual = y_true[indices]
    predicted = y_pred[indices]

    lower = min(
        float(actual.min()),
        float(predicted.min()),
    )

    upper = max(
        float(actual.max()),
        float(predicted.max()),
    )

    metrics = calculate_metrics(
        actual,
        predicted,
    )

    fig, ax = plt.subplots(
        figsize=(8, 8)
    )

    ax.scatter(
        actual,
        predicted,
        s=10,
        alpha=0.22,
    )

    ax.plot(
        [lower, upper],
        [lower, upper],
        linestyle="--",
        linewidth=1.5,
    )

    ax.set_xlabel(
        f"Actual {target}"
    )
    ax.set_ylabel(
        f"Predicted {target}"
    )
    ax.set_title(
        f"Surrogate Validation: Actual vs Predicted {target}",
        fontsize=15,
    )

    stats = (
        f"R² = {metrics['r2']:.6f}\n"
        f"RMSE = {metrics['rmse']:.6g}\n"
        f"MAE = {metrics['mae']:.6g}"
    )

    ax.text(
        0.04,
        0.96,
        stats,
        transform=ax.transAxes,
        va="top",
        bbox={
            "boxstyle": "round",
            "facecolor": "white",
            "alpha": 0.85,
        },
    )

    ax.grid(alpha=0.2)

    save_figure(
        fig,
        f"actual_vs_predicted_{safe_filename(target)}.png",
    )


# =====================================================================
# 3. Residual diagnostics
# =====================================================================

def plot_residual_diagnostics(
    target: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> None:
    """Create residual distribution and residual-vs-prediction plots."""
    residuals = (
        y_true
        - y_pred
    )

    # Residual distribution
    fig, ax = plt.subplots(
        figsize=(9, 5.8)
    )

    sns.histplot(
        residuals,
        bins=60,
        kde=True,
        ax=ax,
    )

    ax.axvline(
        0,
        linestyle="--",
        linewidth=1.2,
    )

    ax.set_xlabel(
        "Residual = actual − predicted"
    )
    ax.set_ylabel("Count")
    ax.set_title(
        f"Residual Distribution — {target}",
        fontsize=15,
    )
    ax.grid(alpha=0.2)

    save_figure(
        fig,
        f"residual_distribution_{safe_filename(target)}.png",
    )

    # Residual vs prediction
    rng = np.random.default_rng(
        RANDOM_STATE
    )

    n = min(
        PLOT_SAMPLE_SIZE,
        len(y_true),
    )

    indices = rng.choice(
        len(y_true),
        size=n,
        replace=False,
    )

    fig, ax = plt.subplots(
        figsize=(10, 6)
    )

    ax.scatter(
        y_pred[indices],
        residuals[indices],
        s=10,
        alpha=0.20,
    )

    ax.axhline(
        0,
        linestyle="--",
        linewidth=1.3,
    )

    ax.set_xlabel(
        f"Predicted {target}"
    )
    ax.set_ylabel("Residual")
    ax.set_title(
        f"Residuals vs Prediction — {target}",
        fontsize=15,
    )
    ax.grid(alpha=0.2)

    save_figure(
        fig,
        f"residual_vs_prediction_{safe_filename(target)}.png",
    )


# =====================================================================
# 4. Error by operating envelope
# =====================================================================

def calculate_error_by_feature_region(
    X_test: pd.DataFrame,
    y_test: pd.DataFrame,
    predictions: dict[str, np.ndarray],
) -> pd.DataFrame:
    """
    Measure surrogate error across quantile-defined regions of each
    continuous input.

    This identifies where the simulator surrogate is weakest.
    """
    rows = []

    for target in TARGET_COLUMNS:
        y_true = y_test[target].to_numpy(
            dtype=float
        )

        y_pred = predictions[target]

        absolute_error = np.abs(
            y_true - y_pred
        )

        for feature in INPUT_COLUMNS:
            values = X_test[feature].to_numpy(
                dtype=float
            )

            quantiles = np.quantile(
                values,
                np.linspace(
                    0,
                    1,
                    N_ENVELOPE_BINS + 1,
                ),
            )

            quantiles = np.unique(
                quantiles
            )

            if len(quantiles) < 3:
                continue

            bins = pd.cut(
                values,
                bins=quantiles,
                include_lowest=True,
                duplicates="drop",
            )

            temp = pd.DataFrame(
                {
                    "bin": bins,
                    "feature_value": values,
                    "absolute_error": absolute_error,
                    "squared_error":
                        (y_true - y_pred) ** 2,
                }
            )

            grouped = (
                temp.groupby(
                    "bin",
                    observed=True,
                )
                .agg(
                    feature_mean=(
                        "feature_value",
                        "mean",
                    ),
                    feature_min=(
                        "feature_value",
                        "min",
                    ),
                    feature_max=(
                        "feature_value",
                        "max",
                    ),
                    mae=(
                        "absolute_error",
                        "mean",
                    ),
                    rmse=(
                        "squared_error",
                        lambda values: np.sqrt(
                            np.mean(values)
                        ),
                    ),
                    count=(
                        "absolute_error",
                        "size",
                    ),
                )
                .reset_index(drop=True)
            )

            grouped["target"] = target
            grouped["feature"] = feature

            rows.append(grouped)

    if not rows:
        return pd.DataFrame()

    return pd.concat(
        rows,
        ignore_index=True,
    )


def save_and_plot_error_by_region(
    error_table: pd.DataFrame,
) -> None:
    """Save operating-region error diagnostics and generate plots."""
    if error_table.empty:
        return

    save_table(
        error_table.round(10),
        "error_by_operating_region.csv",
        index=False,
    )

    for target in TARGET_COLUMNS:
        target_data = error_table[
            error_table["target"] == target
        ]

        for feature in INPUT_COLUMNS:
            sub = target_data[
                target_data["feature"] == feature
            ].copy()

            if sub.empty:
                continue

            fig, ax = plt.subplots(
                figsize=(10, 5.8)
            )

            ax.plot(
                sub["feature_mean"],
                sub["mae"],
                marker="o",
                linewidth=1.8,
            )

            ax.set_xlabel(feature)
            ax.set_ylabel("Mean absolute error")
            ax.set_title(
                f"Surrogate Error Across {feature} — {target}",
                fontsize=15,
            )
            ax.grid(alpha=0.2)

            save_figure(
                fig,
                (
                    f"error_by_region_"
                    f"{safe_filename(target)}_"
                    f"{safe_filename(feature)}.png"
                ),
            )


# =====================================================================
# 5. Derived aerodynamic force validation
# =====================================================================

def calculate_aerodynamic_forces(
    velocity: np.ndarray,
    cd: np.ndarray,
    cl: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Calculate drag and downforce."""
    factor = (
        0.5
        * AIR_DENSITY
        * FRONTAL_AREA
        * velocity**2
    )

    drag = factor * cd
    downforce = factor * cl

    return drag, downforce


def validate_derived_forces(
    X_test: pd.DataFrame,
    y_test: pd.DataFrame,
    predictions: dict[str, np.ndarray],
) -> pd.DataFrame:
    """
    Convert actual/predicted Cd and Cl into forces and validate force errors.
    """
    velocity = X_test["velocity_ms"].to_numpy(
        dtype=float
    )

    actual_cd = y_test["cd_total"].to_numpy(
        dtype=float
    )
    predicted_cd = predictions["cd_total"]

    actual_cl = y_test["cl_total"].to_numpy(
        dtype=float
    )
    predicted_cl = predictions["cl_total"]

    actual_drag, actual_downforce = (
        calculate_aerodynamic_forces(
            velocity,
            actual_cd,
            actual_cl,
        )
    )

    predicted_drag, predicted_downforce = (
        calculate_aerodynamic_forces(
            velocity,
            predicted_cd,
            predicted_cl,
        )
    )

    results = []

    for quantity, actual, predicted in [
        (
            "drag_force_N",
            actual_drag,
            predicted_drag,
        ),
        (
            "downforce_N",
            actual_downforce,
            predicted_downforce,
        ),
    ]:
        metrics = calculate_metrics(
            actual,
            predicted,
        )

        results.append(
            {
                "quantity": quantity,
                "r2": metrics["r2"],
                "rmse_N": metrics["rmse"],
                "mae_N": metrics["mae"],
                "max_absolute_error_N":
                    metrics["max_absolute_error"],
            }
        )

    force_results = pd.DataFrame(
        results
    )

    save_table(
        force_results.round(10),
        "derived_force_validation.csv",
    )

    # Force actual vs predicted plots
    force_pairs = [
        (
            "drag_force_N",
            actual_drag,
            predicted_drag,
            "Drag Force",
            "drag_force_actual_vs_predicted.png",
        ),
        (
            "downforce_N",
            actual_downforce,
            predicted_downforce,
            "Downforce",
            "downforce_actual_vs_predicted.png",
        ),
    ]

    rng = np.random.default_rng(
        RANDOM_STATE
    )

    for (
        quantity,
        actual,
        predicted,
        title,
        filename,
    ) in force_pairs:
        n = min(
            PLOT_SAMPLE_SIZE,
            len(actual),
        )

        indices = rng.choice(
            len(actual),
            size=n,
            replace=False,
        )

        a = actual[indices]
        p = predicted[indices]

        lower = min(
            float(a.min()),
            float(p.min()),
        )

        upper = max(
            float(a.max()),
            float(p.max()),
        )

        fig, ax = plt.subplots(
            figsize=(8, 8)
        )

        ax.scatter(
            a,
            p,
            s=10,
            alpha=0.20,
        )

        ax.plot(
            [lower, upper],
            [lower, upper],
            linestyle="--",
            linewidth=1.5,
        )

        ax.set_xlabel(
            f"Actual {title} (N)"
        )
        ax.set_ylabel(
            f"Predicted {title} (N)"
        )
        ax.set_title(
            f"Ground Truth vs Surrogate — {title}",
            fontsize=15,
        )
        ax.grid(alpha=0.2)

        save_figure(
            fig,
            filename,
        )

    return force_results


# =====================================================================
# 6. Output distribution preservation
# =====================================================================

def compare_prediction_distributions(
    y_test: pd.DataFrame,
    predictions: dict[str, np.ndarray],
) -> pd.DataFrame:
    """
    Compare actual and predicted target distributions using key quantiles.
    """
    rows = []

    quantiles = [
        0.01,
        0.05,
        0.25,
        0.50,
        0.75,
        0.95,
        0.99,
    ]

    for target in TARGET_COLUMNS:
        actual = y_test[target].to_numpy(
            dtype=float
        )
        predicted = predictions[target]

        actual_quantiles = np.quantile(
            actual,
            quantiles,
        )

        predicted_quantiles = np.quantile(
            predicted,
            quantiles,
        )

        for q, actual_q, predicted_q in zip(
            quantiles,
            actual_quantiles,
            predicted_quantiles,
        ):
            rows.append(
                {
                    "target": target,
                    "quantile": q,
                    "actual": actual_q,
                    "predicted": predicted_q,
                    "absolute_difference":
                        abs(actual_q - predicted_q),
                }
            )

    result = pd.DataFrame(rows)

    save_table(
        result.round(10),
        "prediction_distribution_quantiles.csv",
        index=False,
    )

    return result


def plot_distribution_comparison(
    target: str,
    actual: np.ndarray,
    predicted: np.ndarray,
) -> None:
    """Plot actual and predicted target distributions."""
    fig, ax = plt.subplots(
        figsize=(10, 6)
    )

    sns.histplot(
        actual,
        bins=60,
        stat="density",
        alpha=0.35,
        label="Actual",
        ax=ax,
    )

    sns.histplot(
        predicted,
        bins=60,
        stat="density",
        alpha=0.35,
        label="Surrogate",
        ax=ax,
    )

    ax.set_xlabel(target)
    ax.set_ylabel("Density")
    ax.set_title(
        f"Actual vs Surrogate Distribution — {target}",
        fontsize=15,
    )
    ax.legend()
    ax.grid(alpha=0.2)

    save_figure(
        fig,
        f"distribution_comparison_{safe_filename(target)}.png",
    )


# =====================================================================
# 7. Operating-envelope coverage / prediction plausibility
# =====================================================================

def check_prediction_ranges(
    df: pd.DataFrame,
    predictions: dict[str, np.ndarray],
) -> pd.DataFrame:
    """
    Check whether predictions remain within the target range observed in
    the full dataset.

    This is a diagnostic only. It does NOT claim that values outside the
    observed target range are physically impossible.
    """
    rows = []

    for target in TARGET_COLUMNS:
        actual_min = float(
            df[target].min()
        )
        actual_max = float(
            df[target].max()
        )

        pred = predictions[target]

        outside_lower = int(
            np.sum(pred < actual_min)
        )
        outside_upper = int(
            np.sum(pred > actual_max)
        )

        rows.append(
            {
                "target": target,
                "observed_min": actual_min,
                "observed_max": actual_max,
                "predicted_min":
                    float(pred.min()),
                "predicted_max":
                    float(pred.max()),
                "predictions_below_observed_min":
                    outside_lower,
                "predictions_above_observed_max":
                    outside_upper,
                "percent_outside_observed_range":
                    (
                        (outside_lower + outside_upper)
                        / len(pred)
                        * 100
                    ),
            }
        )

    result = pd.DataFrame(rows)

    save_table(
        result.round(10),
        "prediction_range_check.csv",
    )

    return result


# =====================================================================
# 8. Prediction-speed benchmark
# =====================================================================

def benchmark_prediction_speed(
    models: dict[str, object],
    X_test: pd.DataFrame,
) -> pd.DataFrame:
    """
    Estimate prediction throughput for repeated surrogate calls.

    This is relevant because the eventual simulator will query the surrogate
    repeatedly.
    """
    print_section("3. SURROGATE PREDICTION SPEED")

    sample_size = min(
        BENCHMARK_CALLS,
        len(X_test),
    )

    sample = X_test.iloc[
        :sample_size
    ][INPUT_COLUMNS]

    rows = []

    for target in TARGET_COLUMNS:
        model = models[target]

        # Warm-up call so one-time internal setup does not dominate timing.
        model.predict(
            sample.iloc[: min(100, len(sample))]
        )

        start = time.perf_counter()

        prediction = model.predict(
            sample
        )

        elapsed = (
            time.perf_counter()
            - start
        )

        predictions_per_second = (
            len(sample) / elapsed
            if elapsed > 0
            else np.nan
        )

        microseconds_per_prediction = (
            elapsed
            / len(sample)
            * 1e6
            if len(sample) > 0
            else np.nan
        )

        rows.append(
            {
                "target": target,
                "n_predictions": len(prediction),
                "elapsed_seconds": elapsed,
                "predictions_per_second":
                    predictions_per_second,
                "microseconds_per_prediction":
                    microseconds_per_prediction,
            }
        )

        print(
            f"{target:20s} -> "
            f"{predictions_per_second:,.0f} predictions/s"
        )

    result = pd.DataFrame(rows)

    save_table(
        result.round(6),
        "prediction_speed_benchmark.csv",
    )

    return result


# =====================================================================
# 9. Markdown report
# =====================================================================

def create_markdown_report(
    overall_results: pd.DataFrame,
    force_results: pd.DataFrame,
    quantile_results: pd.DataFrame,
    range_results: pd.DataFrame,
    speed_results: pd.DataFrame,
    error_region_table: pd.DataFrame,
) -> None:
    """Create the surrogate-validation summary."""
    report_path = REPORT_DIR / "surrogate_summary.md"

    lines = [
        "# F1 2026 Mode-1 Dataset — Surrogate Validation",
        "",
        "## Objective",
        "",
        "The final models from Phase 5 were loaded without retraining and "
        "validated on the same untouched 15% test split used during final "
        "model evaluation.",
        "",
        "The purpose is to determine whether the saved models are reliable "
        "enough to act as aerodynamic surrogate models inside the simulator.",
        "",
        "## Surrogate inputs",
        "",
    ]

    for feature in INPUT_COLUMNS:
        lines.append(f"- `{feature}`")

    lines.extend(
        [
            "",
            "## Surrogate outputs",
            "",
        ]
    )

    for target in TARGET_COLUMNS:
        role = (
            "primary"
            if target in PRIMARY_TARGETS
            else "secondary"
        )
        lines.append(
            f"- `{target}` ({role})"
        )

    lines.extend(
        [
            "",
            "## 1. Overall test-set validation",
            "",
        ]
    )

    lines.append(
        dataframe_to_markdown_table(
            overall_results,
            index=False,
            float_digits=8,
        )
    )

    lines.extend(
        [
            "",
            "## 2. Derived aerodynamic-force validation",
            "",
            "Using:",
            "",
            f"`rho = {AIR_DENSITY} kg/m^3`",
            "",
            f"`A = {FRONTAL_AREA} m^2`",
            "",
            "`F_D = 0.5 * rho * Cd * A * v^2`",
            "",
            "`F_L = 0.5 * rho * Cl * A * v^2`",
            "",
        ]
    )

    lines.append(
        dataframe_to_markdown_table(
            force_results,
            index=False,
            float_digits=8,
        )
    )

    lines.extend(
        [
            "",
            "## 3. Prediction distribution preservation",
            "",
            "The following quantiles compare actual test-set outputs with "
            "surrogate predictions.",
            "",
        ]
    )

    # Keep the report readable by showing a compact subset.
    compact_quantiles = quantile_results.copy()
    compact_quantiles["quantile"] = compact_quantiles[
        "quantile"
    ].map(
        lambda q: f"{q:.2f}"
    )

    lines.append(
        dataframe_to_markdown_table(
            compact_quantiles,
            index=False,
            float_digits=7,
        )
    )

    lines.extend(
        [
            "",
            "## 4. Prediction range check",
            "",
        ]
    )

    lines.append(
        dataframe_to_markdown_table(
            range_results,
            index=False,
            float_digits=8,
        )
    )

    lines.extend(
        [
            "",
            "## 5. Prediction speed",
            "",
        ]
    )

    lines.append(
        dataframe_to_markdown_table(
            speed_results,
            index=False,
            float_digits=5,
        )
    )

    lines.extend(
        [
            "",
            "## 6. Operating-region error analysis",
            "",
        ]
    )

    if error_region_table.empty:
        lines.append(
            "No operating-region error table was generated."
        )
    else:
        # Summarize the largest MAE region for every target/feature pair.
        worst_rows = []

        for (target, feature), group in error_region_table.groupby(
            ["target", "feature"]
        ):
            row = group.loc[
                group["mae"].idxmax()
            ]

            worst_rows.append(
                {
                    "target": target,
                    "feature": feature,
                    "worst_region_mean":
                        row["feature_mean"],
                    "worst_region_mae":
                        row["mae"],
                    "worst_region_rmse":
                        row["rmse"],
                    "count":
                        row["count"],
                }
            )

        worst = pd.DataFrame(
            worst_rows
        ).sort_values(
            [
                "target",
                "worst_region_mae",
            ],
            ascending=[
                True,
                False,
            ],
        )

        lines.append(
            dataframe_to_markdown_table(
                worst,
                index=False,
                float_digits=7,
            )
        )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "A surrogate is considered useful when its prediction errors are "
            "small on unseen data, remain controlled across the operating "
            "envelope, and do not create large errors after conversion from "
            "aerodynamic coefficients into forces.",
            "",
            "For the eventual simulator, Cd and Cl are the critical outputs. "
            "CoP shift and energy drain are validated here so they remain "
            "available for later simulator extensions.",
            "",
            "The next phase can therefore connect the validated Cd and Cl "
            "surrogate models to the aerodynamic force equations and the "
            "vehicle longitudinal-dynamics simulator.",
        ]
    )

    report_path.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )

    print(
        f"Saved report: {report_path}"
    )


# =====================================================================
# Main
# =====================================================================

def main() -> int:
    print_section(
        "F1 2026 AERODYNAMIC SIMULATOR — PHASE 6"
    )

    if not DATA_PATH.exists():
        print(
            "ERROR: Dataset not found.\n"
            f"Expected path: {DATA_PATH}"
        )
        return 1

    ensure_output_directories()

    # ---------------------------------------------------------------
    # Load dataset
    # ---------------------------------------------------------------
    print_section("LOADING DATASET")

    try:
        df = pd.read_csv(
            DATA_PATH
        )
    except Exception as exc:
        print(
            f"ERROR: Could not read dataset: {exc}"
        )
        return 1

    try:
        check_required_columns(df)
    except (ValueError, TypeError) as exc:
        print(
            f"ERROR: {exc}"
        )
        return 1

    print(
        f"Dataset: {DATA_PATH.name}\n"
        f"Rows:    {len(df):,}\n"
        f"Columns: {df.shape[1]}"
    )

    # ---------------------------------------------------------------
    # Load models
    # ---------------------------------------------------------------
    try:
        models = load_saved_models()
    except (
        FileNotFoundError,
        TypeError,
        KeyError,
        ValueError,
    ) as exc:
        print(
            f"\nERROR while loading saved models:\n{exc}"
        )
        return 1

    # ---------------------------------------------------------------
    # Reproduce test set
    # ---------------------------------------------------------------
    print_section(
        "REPRODUCING THE UNTOUCHED PHASE-5 TEST SET"
    )

    try:
        X_test, y_test = reproduce_phase5_test_split(
            df
        )
    except Exception as exc:
        print(
            f"ERROR while reproducing test split: {exc}"
        )
        return 1

    print(
        f"Test observations: {len(X_test):,}"
    )

    # ---------------------------------------------------------------
    # Predictions
    # ---------------------------------------------------------------
    predictions = generate_test_predictions(
        models,
        X_test,
    )

    # ---------------------------------------------------------------
    # Overall metrics
    # ---------------------------------------------------------------
    print_section(
        "OVERALL SURROGATE VALIDATION"
    )

    overall_results = calculate_overall_validation(
        y_test,
        predictions,
    )

    print(
        overall_results.round(8).to_string(
            index=False
        )
    )

    # ---------------------------------------------------------------
    # Actual-vs-predicted + residual plots
    # ---------------------------------------------------------------
    print_section(
        "GENERATING SURROGATE DIAGNOSTICS"
    )

    for target in TARGET_COLUMNS:
        y_true = y_test[target].to_numpy(
            dtype=float
        )

        y_pred = predictions[target]

        plot_actual_vs_predicted(
            target,
            y_true,
            y_pred,
        )

        plot_residual_diagnostics(
            target,
            y_true,
            y_pred,
        )

        plot_distribution_comparison(
            target,
            y_true,
            y_pred,
        )

    # ---------------------------------------------------------------
    # Operating-region errors
    # ---------------------------------------------------------------
    print_section(
        "OPERATING-ENVELOPE ERROR ANALYSIS"
    )

    error_region_table = (
        calculate_error_by_feature_region(
            X_test,
            y_test,
            predictions,
        )
    )

    save_and_plot_error_by_region(
        error_region_table
    )

    # ---------------------------------------------------------------
    # Derived force validation
    # ---------------------------------------------------------------
    print_section(
        "DERIVED AERODYNAMIC-FORCE VALIDATION"
    )

    force_results = validate_derived_forces(
        X_test,
        y_test,
        predictions,
    )

    print(
        force_results.round(8).to_string(
            index=False
        )
    )

    # ---------------------------------------------------------------
    # Distribution and range checks
    # ---------------------------------------------------------------
    print_section(
        "DISTRIBUTION + RANGE VALIDATION"
    )

    quantile_results = (
        compare_prediction_distributions(
            y_test,
            predictions,
        )
    )

    range_results = check_prediction_ranges(
        df,
        predictions,
    )

    print("\nPrediction range check:")
    print(
        range_results.round(6).to_string(
            index=False
        )
    )

    # ---------------------------------------------------------------
    # Prediction speed
    # ---------------------------------------------------------------
    speed_results = benchmark_prediction_speed(
        models,
        X_test,
    )

    # ---------------------------------------------------------------
    # Markdown report
    # ---------------------------------------------------------------
    print_section(
        "WRITING SURROGATE VALIDATION REPORT"
    )

    create_markdown_report(
        overall_results=overall_results,
        force_results=force_results,
        quantile_results=quantile_results,
        range_results=range_results,
        speed_results=speed_results,
        error_region_table=error_region_table,
    )

    # ---------------------------------------------------------------
    # Completion
    # ---------------------------------------------------------------
    print_section(
        "PHASE 6 COMPLETE"
    )

    print(
        f"Figures:\n  {FIGURE_DIR}\n\n"
        f"Tables:\n  {TABLE_DIR}\n\n"
        f"Report:\n  {REPORT_DIR / 'surrogate_summary.md'}"
    )

    print(
        "\nNext step: Phase 7 — build the aerodynamic reference model "
        "and ML-simulation engine."
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
