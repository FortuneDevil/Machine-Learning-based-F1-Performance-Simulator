"""
05_regression_models.py

Phase 5 of the F1 2026 Data-Driven Aerodynamic Simulator project.

Purpose
-------
Select and validate the final continuous-to-continuous regression models
that will later act as the ML aerodynamic surrogate.

Main predictive tasks
---------------------
Primary simulator targets:
    1. cl_total
    2. cd_total

Secondary targets:
    3. cop_shift
    4. energy_drain_kw

Continuous inputs:
    velocity_ms
    flap_front_angle
    flap_rear_angle
    ride_height_front
    ride_height_rear
    battery_soc

Models compared
---------------
1. Multiple Linear Regression
2. Polynomial Ridge Regression (degree 2)
3. Extra Trees Regressor
4. HistGradientBoosting Regressor

Why these four?
----------------
- Linear regression provides the statistical baseline and is easy to interpret.
- Polynomial regression tests whether smooth nonlinear terms improve the
  additive linear model.
- Extra Trees captures nonlinear relationships and interactions.
- HistGradientBoosting provides a second nonlinear model using a different
  learning mechanism.

Evaluation design
-----------------
The dataset is split into:

    70% training
    15% validation
    15% final test

The validation set is used to select the best model for each target.
The final test set is NOT used for model selection.

Metrics:
    R²
    RMSE
    MAE

Additional diagnostics:
    - Actual vs predicted
    - Residual distribution
    - Residual vs predicted
    - Error distribution
    - Model comparison
    - Validation/test performance
    - Tree-based feature importance for nonlinear models

Final model selection
---------------------
The selected model for each target is retrained using the combined
training + validation data and then evaluated once on the final test set.

Models are saved under:

    models/

The primary Cd and Cl models will later be loaded by the simulator.

Important
---------
- The source CSV is NEVER modified.
- Aero mode is not used.
- All predictors and targets are continuous.
- This phase does not build the simulator yet.

Outputs
-------
reports/05_regression_models/
├── figures/
├── tables/
└── regression_summary.md

models/
├── cd_model.pkl
├── cl_model.pkl
├── cop_model.pkl
└── energy_model.pkl

Run
---
python 05_regression_models.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Force a non-GUI backend before importing pyplot. This avoids Tkinter
# crashes when generating figures from a terminal/headless environment.
import matplotlib
matplotlib.use("Agg")

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler


# =====================================================================
# Configuration
# =====================================================================

PROJECT_ROOT = Path(__file__).resolve().parent

DATA_PATH = PROJECT_ROOT / "data" / "F1_2026_Synthetic_Dataset_Mode1.csv"

REPORT_DIR = PROJECT_ROOT / "reports" / "05_regression_models"
FIGURE_DIR = REPORT_DIR / "figures"
TABLE_DIR = REPORT_DIR / "tables"

MODEL_DIR = PROJECT_ROOT / "models"

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

PRIMARY_TARGETS = [
    "cl_total",
    "cd_total",
]

RANDOM_STATE = 42

TRAIN_FRACTION = 0.70
VALIDATION_FRACTION = 0.15
TEST_FRACTION = 0.15

# Nonlinear model settings.
# Each fully grown tree is about 12 MB, so 250 trees is about 3 GB per
# model and 9 GB for the three Extra Trees targets, which does not fit in
# 16 GB of RAM alongside the simulator. 60 trees is about 0.7 GB per model
# and costs roughly 5% in Cl validation RMSE (0.0138 vs 0.0131).
EXTRA_TREES_ESTIMATORS = 60

HGB_MAX_ITER = 300
HGB_LEARNING_RATE = 0.05
HGB_MAX_LEAF_NODES = 31
HGB_MIN_SAMPLES_LEAF = 20
HGB_L2_REGULARIZATION = 1e-4

# Polynomial settings.
POLY_DEGREE = 2
RIDGE_ALPHA = 1e-8

# Plotting.
PLOT_SAMPLE_SIZE = 12000


# =====================================================================
# Utility functions
# =====================================================================

def print_section(title: str) -> None:
    """Print a clear section heading."""
    print("\n" + "=" * 86)
    print(title)
    print("=" * 86)


def ensure_output_directories() -> None:
    """Create all output directories."""
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)


def save_figure(fig: plt.Figure, filename: str) -> None:
    """Save a figure consistently."""
    path = FIGURE_DIR / filename
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure: {path}")


def save_table(
    table: pd.DataFrame,
    filename: str,
    index: bool = False,
) -> None:
    """Save a DataFrame as CSV."""
    path = TABLE_DIR / filename
    table.to_csv(path, index=index)
    print(f"Saved table: {path}")


def dataframe_to_markdown_table(
    table: pd.DataFrame,
    index: bool = False,
    float_digits: int = 5,
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


def check_dataset(df: pd.DataFrame) -> None:
    """Validate required columns and numerical structure."""
    expected = INPUT_COLUMNS + TARGET_COLUMNS

    missing = [column for column in expected if column not in df.columns]

    if missing:
        raise ValueError(
            "Missing required columns:\n"
            + "\n".join(f"  - {column}" for column in missing)
        )

    non_numeric = [
        column
        for column in expected
        if not pd.api.types.is_numeric_dtype(df[column])
    ]

    if non_numeric:
        raise TypeError(
            "The regression project requires continuous numerical "
            "variables. Non-numeric columns found:\n"
            + "\n".join(f"  - {column}" for column in non_numeric)
        )

    values = df[expected].to_numpy(dtype=float)

    if not np.isfinite(values).all():
        raise ValueError(
            "Non-finite values found in the modelling variables."
        )


def calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, float]:
    """Calculate standard regression metrics."""
    mse = mean_squared_error(y_true, y_pred)

    return {
        "r2": float(r2_score(y_true, y_pred)),
        "rmse": float(np.sqrt(mse)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
    }


def safe_filename(text: str) -> str:
    """Create a simple filename-safe string."""
    return (
        text.lower()
        .replace(" ", "_")
        .replace("/", "_")
        .replace("-", "_")
    )


# =====================================================================
# Data splitting
# =====================================================================

def make_three_way_split(
    df: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    """
    Create an honest train/validation/test split.

    First split:
        70% train
        30% temporary

    Second split of temporary:
        50% validation
        50% test

    Result:
        70 / 15 / 15
    """
    X = df[INPUT_COLUMNS].copy()
    y = df[TARGET_COLUMNS].copy()

    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=(1.0 - TRAIN_FRACTION),
        random_state=RANDOM_STATE,
    )

    temp_test_share = TEST_FRACTION / (
        VALIDATION_FRACTION + TEST_FRACTION
    )

    X_validation, X_test, y_validation, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=temp_test_share,
        random_state=RANDOM_STATE,
    )

    return (
        X_train,
        X_validation,
        X_test,
        y_train,
        y_validation,
        y_test,
    )


# =====================================================================
# Model definitions
# =====================================================================

def create_models() -> dict[str, object]:
    """Create all candidate regression models."""
    return {
        "Multiple Linear Regression": LinearRegression(),

        "Polynomial Ridge Regression": Pipeline(
            steps=[
                (
                    "polynomial_features",
                    PolynomialFeatures(
                        degree=POLY_DEGREE,
                        include_bias=False,
                    ),
                ),
                (
                    "scaler",
                    StandardScaler(),
                ),
                (
                    "ridge",
                    Ridge(
                        alpha=RIDGE_ALPHA,
                    ),
                ),
            ]
        ),

        "Extra Trees": ExtraTreesRegressor(
            n_estimators=EXTRA_TREES_ESTIMATORS,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            max_features=1.0,
            min_samples_leaf=1,
        ),

        "HistGradientBoosting": HistGradientBoostingRegressor(
            max_iter=HGB_MAX_ITER,
            learning_rate=HGB_LEARNING_RATE,
            max_leaf_nodes=HGB_MAX_LEAF_NODES,
            min_samples_leaf=HGB_MIN_SAMPLES_LEAF,
            l2_regularization=HGB_L2_REGULARIZATION,
            random_state=RANDOM_STATE,
        ),
    }


# =====================================================================
# Model evaluation
# =====================================================================

def evaluate_candidate_models(
    X_train: pd.DataFrame,
    X_validation: pd.DataFrame,
    y_train: pd.DataFrame,
    y_validation: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[tuple[str, str], object]]:
    """
    Fit every candidate model on training data and evaluate on validation data.

    Returns
    -------
    results
        One row per target/model.
    fitted_models
        Kept for interface compatibility; always empty (see below).
    """
    print_section("1. VALIDATION-SET MODEL COMPARISON")

    results: list[dict[str, object]] = []
    fitted_models: dict[tuple[str, str], object] = {}

    for target in TARGET_COLUMNS:
        print(f"\n--- TARGET: {target} ---")

        ytr = y_train[target]
        yval = y_validation[target]

        for model_name, model in create_models().items():
            print(f"Training: {model_name}")

            model.fit(
                X_train,
                ytr,
            )

            prediction = model.predict(
                X_validation
            )

            metrics = calculate_metrics(
                yval.to_numpy(dtype=float),
                prediction,
            )

            results.append(
                {
                    "target": target,
                    "model": model_name,
                    "r2": metrics["r2"],
                    "rmse": metrics["rmse"],
                    "mae": metrics["mae"],
                    "n_train": len(X_train),
                    "n_validation": len(X_validation),
                }
            )

            # The validation-fit models are not kept: the selected models
            # are refitted from scratch on train + validation data, and each
            # fully grown Extra Trees forest occupies gigabytes of memory.
            del model

            print(
                f"  R²   = {metrics['r2']:.8f}\n"
                f"  RMSE = {metrics['rmse']:.8f}\n"
                f"  MAE  = {metrics['mae']:.8f}"
            )

    results_df = pd.DataFrame(results)

    save_table(
        results_df.round(10),
        "validation_model_comparison.csv",
    )

    return results_df, fitted_models


# =====================================================================
# Model selection
# =====================================================================

def select_best_models(
    validation_results: pd.DataFrame,
) -> pd.DataFrame:
    """
    Select one candidate model per target.

    Selection criterion:
        lowest validation RMSE.

    R² and MAE remain available for interpretation.
    """
    selected_rows = []

    for target in TARGET_COLUMNS:
        candidates = validation_results[
            validation_results["target"] == target
        ].copy()

        best = candidates.loc[
            candidates["rmse"].idxmin()
        ]

        selected_rows.append(
            best.to_dict()
        )

    selected = pd.DataFrame(selected_rows)

    save_table(
        selected.round(10),
        "selected_models_from_validation.csv",
    )

    return selected


# =====================================================================
# Visualization: model comparison
# =====================================================================

def plot_model_comparison(
    validation_results: pd.DataFrame,
) -> None:
    """Plot validation RMSE comparison for every target."""
    for target in TARGET_COLUMNS:
        sub = validation_results[
            validation_results["target"] == target
        ].sort_values("rmse")

        fig, ax = plt.subplots(figsize=(10, 5.8))

        bars = ax.bar(
            sub["model"],
            sub["rmse"],
        )

        ax.set_title(
            f"Validation RMSE Comparison — {target}",
            fontsize=16,
        )
        ax.set_xlabel("Model")
        ax.set_ylabel("RMSE")
        ax.tick_params(axis="x", rotation=15)
        ax.grid(axis="y", alpha=0.2)

        # Values are shown only when they fit cleanly.
        ymax = float(sub["rmse"].max())
        offset = ymax * 0.015 if ymax > 0 else 0.001

        for bar, value in zip(
            bars,
            sub["rmse"],
        ):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + offset,
                f"{value:.5g}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

        save_figure(
            fig,
            f"validation_rmse_comparison_{safe_filename(target)}.png",
        )


def plot_model_r2_comparison(
    validation_results: pd.DataFrame,
) -> None:
    """Plot validation R² comparison."""
    for target in TARGET_COLUMNS:
        sub = validation_results[
            validation_results["target"] == target
        ].sort_values("r2")

        fig, ax = plt.subplots(figsize=(10, 5.8))

        bars = ax.bar(
            sub["model"],
            sub["r2"],
        )

        ax.set_title(
            f"Validation R² Comparison — {target}",
            fontsize=16,
        )
        ax.set_xlabel("Model")
        ax.set_ylabel("R²")
        ax.tick_params(axis="x", rotation=15)
        ax.grid(axis="y", alpha=0.2)

        for bar, value in zip(
            bars,
            sub["r2"],
        ):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.002,
                f"{value:.6f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

        save_figure(
            fig,
            f"validation_r2_comparison_{safe_filename(target)}.png",
        )


# =====================================================================
# Final training and test evaluation
# =====================================================================

def train_final_models_and_evaluate(
    df: pd.DataFrame,
    selected_models: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, object], dict[str, dict[str, object]]]:
    """
    Retrain selected models on train + validation data and evaluate once
    on the final untouched test set.

    Returns
    -------
    test_results
        Final test metrics.
    final_models
        Final fitted models.
    final_predictions
        Test predictions and true values for diagnostics.
    """
    print_section("2. FINAL TRAINING + TEST EVALUATION")

    X = df[INPUT_COLUMNS].copy()
    y = df[TARGET_COLUMNS].copy()

    # Recreate exactly the same split used earlier.
    (
        X_train,
        X_validation,
        X_test,
        y_train,
        y_validation,
        y_test,
    ) = make_three_way_split(df)

    X_train_full = pd.concat(
        [X_train, X_validation],
        ignore_index=True,
    )

    y_train_full = pd.concat(
        [y_train, y_validation],
        ignore_index=True,
    )

    test_results = []
    final_models: dict[str, object] = {}
    final_predictions: dict[str, dict[str, object]] = {}

    for _, selected_row in selected_models.iterrows():
        target = str(selected_row["target"])
        model_name = str(selected_row["model"])

        print(
            f"\nFinal model for {target}: {model_name}"
        )

        # Create a fresh model instance rather than reusing the validation-fit
        # object.
        model = create_models()[model_name]

        model.fit(
            X_train_full,
            y_train_full[target],
        )

        prediction = model.predict(
            X_test
        )

        metrics = calculate_metrics(
            y_test[target].to_numpy(dtype=float),
            prediction,
        )

        final_models[target] = model

        final_predictions[target] = {
            "y_true": y_test[target].to_numpy(dtype=float),
            "y_pred": np.asarray(prediction, dtype=float),
            "X_test": X_test.copy(),
        }

        test_results.append(
            {
                "target": target,
                "selected_model": model_name,
                "r2": metrics["r2"],
                "rmse": metrics["rmse"],
                "mae": metrics["mae"],
                "n_train_plus_validation": len(X_train_full),
                "n_test": len(X_test),
            }
        )

        print(
            f"  Final test R²   = {metrics['r2']:.8f}\n"
            f"  Final test RMSE = {metrics['rmse']:.8f}\n"
            f"  Final test MAE  = {metrics['mae']:.8f}"
        )

        model_filename = (
            f"{safe_filename(target).replace('_total', '')}_model.pkl"
        )

        model_path = MODEL_DIR / model_filename

        # Save a small metadata bundle together with the model. This makes
        # the later simulator safer because it knows exactly which features
        # and target the model expects.
        bundle = {
            "model": model,
            "target": target,
            "input_columns": INPUT_COLUMNS,
            "model_name": model_name,
            "random_state": RANDOM_STATE,
        }

        joblib.dump(
            bundle,
            model_path,
        )

        print(f"Saved model: {model_path}")

    test_results_df = pd.DataFrame(test_results)

    save_table(
        test_results_df.round(10),
        "final_test_results.csv",
    )

    return (
        test_results_df,
        final_models,
        final_predictions,
    )


# =====================================================================
# Diagnostic plots
# =====================================================================

def plot_actual_vs_predicted(
    target: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> None:
    """Plot actual versus predicted values."""
    rng = np.random.default_rng(RANDOM_STATE)

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

    lo = min(
        actual.min(),
        predicted.min(),
    )
    hi = max(
        actual.max(),
        predicted.max(),
    )

    metrics = calculate_metrics(
        actual,
        predicted,
    )

    fig, ax = plt.subplots(figsize=(8, 8))

    ax.scatter(
        actual,
        predicted,
        s=10,
        alpha=0.20,
    )

    ax.plot(
        [lo, hi],
        [lo, hi],
        linestyle="--",
        linewidth=1.6,
    )

    ax.set_xlabel(f"Actual {target}")
    ax.set_ylabel(f"Predicted {target}")
    ax.set_title(
        f"Actual vs Predicted — {target}",
        fontsize=16,
    )

    text = (
        f"R² = {metrics['r2']:.6f}\n"
        f"RMSE = {metrics['rmse']:.6g}\n"
        f"MAE = {metrics['mae']:.6g}"
    )

    ax.text(
        0.04,
        0.96,
        text,
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


def plot_residual_distribution(
    target: str,
    residuals: np.ndarray,
) -> None:
    """Plot residual distribution."""
    fig, ax = plt.subplots(figsize=(9, 5.8))

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

    ax.set_xlabel("Residual = actual − predicted")
    ax.set_ylabel("Count")
    ax.set_title(
        f"Residual Distribution — {target}",
        fontsize=16,
    )
    ax.grid(alpha=0.2)

    save_figure(
        fig,
        f"residual_distribution_{safe_filename(target)}.png",
    )


def plot_residual_vs_prediction(
    target: str,
    y_pred: np.ndarray,
    residuals: np.ndarray,
) -> None:
    """Plot residuals against predicted target."""
    rng = np.random.default_rng(RANDOM_STATE)

    n = min(
        PLOT_SAMPLE_SIZE,
        len(y_pred),
    )

    indices = rng.choice(
        len(y_pred),
        size=n,
        replace=False,
    )

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.scatter(
        y_pred[indices],
        residuals[indices],
        s=10,
        alpha=0.18,
    )

    ax.axhline(
        0,
        linestyle="--",
        linewidth=1.3,
    )

    ax.set_xlabel(f"Predicted {target}")
    ax.set_ylabel("Residual")
    ax.set_title(
        f"Residuals vs Predictions — {target}",
        fontsize=16,
    )
    ax.grid(alpha=0.2)

    save_figure(
        fig,
        f"residual_vs_prediction_{safe_filename(target)}.png",
    )


def save_test_prediction_sample(
    final_predictions: dict[str, dict[str, object]],
) -> None:
    """Save a manageable sample of final test predictions."""
    combined = []

    for target, content in final_predictions.items():
        y_true = content["y_true"]
        y_pred = content["y_pred"]

        frame = pd.DataFrame(
            {
                f"{target}_actual": y_true,
                f"{target}_predicted": y_pred,
                f"{target}_residual":
                    y_true - y_pred,
            }
        )

        combined.append(frame)

    result = pd.concat(
        combined,
        axis=1,
    )

    sample = result.sample(
        n=min(10000, len(result)),
        random_state=RANDOM_STATE,
    )

    save_table(
        sample.round(8),
        "final_test_predictions_sample.csv",
        index=False,
    )


# =====================================================================
# Feature importance
# =====================================================================

def extract_tree_feature_importance(
    final_models: dict[str, object],
) -> pd.DataFrame:
    """
    Extract built-in feature importance from tree-based final models.

    Only models exposing feature_importances_ are included.
    """
    rows = []

    for target, model in final_models.items():
        if not hasattr(
            model,
            "feature_importances_",
        ):
            continue

        values = np.asarray(
            model.feature_importances_,
            dtype=float,
        )

        for feature, importance in zip(
            INPUT_COLUMNS,
            values,
        ):
            rows.append(
                {
                    "target": target,
                    "feature": feature,
                    "importance": importance,
                }
            )

    result = pd.DataFrame(rows)

    if result.empty:
        return result

    result = result.sort_values(
        ["target", "importance"],
        ascending=[True, False],
    )

    save_table(
        result.round(8),
        "final_tree_feature_importance.csv",
    )

    for target in result["target"].unique():
        sub = result[
            result["target"] == target
        ].sort_values(
            "importance"
        )

        fig, ax = plt.subplots(figsize=(9, 5.5))

        ax.barh(
            sub["feature"],
            sub["importance"],
        )

        ax.set_xlabel("Tree-based feature importance")
        ax.set_ylabel("Continuous input")
        ax.set_title(
            f"Final Model Feature Importance — {target}",
            fontsize=15,
        )
        ax.grid(axis="x", alpha=0.2)

        save_figure(
            fig,
            f"final_feature_importance_{safe_filename(target)}.png",
        )

    return result


# =====================================================================
# Markdown report
# =====================================================================

def create_markdown_report(
    validation_results: pd.DataFrame,
    selected_models: pd.DataFrame,
    test_results: pd.DataFrame,
    tree_importance: pd.DataFrame,
) -> None:
    """Write a detailed modelling summary."""
    report_path = REPORT_DIR / "regression_summary.md"

    lines = [
        "# F1 2026 Mode-1 Dataset — Final Regression Analysis",
        "",
        "## Objective",
        "",
        "This phase selects continuous-to-continuous regression models for "
        "the four continuous aerodynamic outputs. Cd and Cl are treated as "
        "the primary simulator targets; CoP shift and energy drain are "
        "secondary outputs.",
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
            "## Model candidates",
            "",
            "- Multiple Linear Regression",
            "- Polynomial Ridge Regression (degree 2)",
            "- Extra Trees Regressor",
            "- HistGradientBoosting Regressor",
            "",
            "## Data split",
            "",
            "- 70% training",
            "- 15% validation",
            "- 15% final test",
            "",
            "The validation set was used for model selection. The final test "
            "set was reserved for the final performance estimate.",
            "",
            "## Validation results",
            "",
        ]
    )

    lines.append(
        dataframe_to_markdown_table(
            validation_results.sort_values(
                ["target", "rmse"]
            ),
            index=False,
            float_digits=8,
        )
    )

    lines.extend(
        [
            "",
            "## Selected models",
            "",
        ]
    )

    lines.append(
        dataframe_to_markdown_table(
            selected_models,
            index=False,
            float_digits=8,
        )
    )

    lines.extend(
        [
            "",
            "## Final untouched-test results",
            "",
        ]
    )

    lines.append(
        dataframe_to_markdown_table(
            test_results,
            index=False,
            float_digits=8,
        )
    )

    lines.extend(
        [
            "",
            "## Tree-model feature importance",
            "",
        ]
    )

    if tree_importance.empty:
        lines.append(
            "No final selected model exposed a native tree feature-importance "
            "attribute."
        )
    else:
        for target in tree_importance["target"].unique():
            lines.append(f"### `{target}`")

            sub = tree_importance[
                tree_importance["target"] == target
            ].head(6)

            lines.append(
                dataframe_to_markdown_table(
                    sub[
                        ["feature", "importance"]
                    ],
                    index=False,
                    float_digits=8,
                )
            )

            lines.append("")

    lines.extend(
        [
            "## Interpretation",
            "",
            "Model selection is based on predictive performance but should "
            "also consider simulator behaviour. The final surrogate should "
            "not be chosen solely because it has the highest R² if a simpler "
            "model produces nearly identical simulation behaviour.",
            "",
            "The primary simulator will use the final Cd and Cl models to "
            "calculate aerodynamic drag and downforce. The CoP and energy "
            "models can be incorporated into later extensions of the "
            "simulator.",
            "",
            "## Next phase",
            "",
            "Build the aerodynamic surrogate interface and then construct the "
            "reference-vs-ML simulator. The simulator will use the saved "
            "Cd/Cl models and the same continuous inputs used during training.",
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
        "F1 2026 AERODYNAMIC SIMULATOR — PHASE 5"
    )

    if not DATA_PATH.exists():
        print(
            "ERROR: Dataset not found.\n"
            f"Expected path: {DATA_PATH}"
        )
        return 1

    ensure_output_directories()

    # ---------------------------------------------------------------
    # Load and validate
    # ---------------------------------------------------------------
    print_section("LOADING DATASET")

    try:
        df = pd.read_csv(DATA_PATH)
    except Exception as exc:
        print(f"ERROR: Could not read dataset: {exc}")
        return 1

    try:
        check_dataset(df)
    except (ValueError, TypeError) as exc:
        print(f"ERROR: {exc}")
        return 1

    print(
        f"Dataset: {DATA_PATH.name}\n"
        f"Rows:    {len(df):,}\n"
        f"Inputs:  {len(INPUT_COLUMNS)} continuous\n"
        f"Targets: {len(TARGET_COLUMNS)} continuous"
    )

    # ---------------------------------------------------------------
    # Split
    # ---------------------------------------------------------------
    print_section("CREATING TRAIN / VALIDATION / TEST SPLIT")

    (
        X_train,
        X_validation,
        X_test,
        y_train,
        y_validation,
        y_test,
    ) = make_three_way_split(df)

    print(f"Training rows:    {len(X_train):,}")
    print(f"Validation rows:  {len(X_validation):,}")
    print(f"Final test rows:  {len(X_test):,}")

    # ---------------------------------------------------------------
    # Candidate models
    # ---------------------------------------------------------------
    validation_results, _ = evaluate_candidate_models(
        X_train,
        X_validation,
        y_train,
        y_validation,
    )

    # ---------------------------------------------------------------
    # Model selection
    # ---------------------------------------------------------------
    selected_models = select_best_models(
        validation_results,
    )

    print_section("SELECTED MODELS")

    for _, row in selected_models.iterrows():
        print(
            f"{row['target']:20s} -> {row['model']} "
            f"(validation RMSE = {row['rmse']:.8f})"
        )

    plot_model_comparison(
        validation_results,
    )

    plot_model_r2_comparison(
        validation_results,
    )

    # ---------------------------------------------------------------
    # Final models + untouched test
    # ---------------------------------------------------------------
    (
        test_results,
        final_models,
        final_predictions,
    ) = train_final_models_and_evaluate(
        df,
        selected_models,
    )

    print_section("FINAL TEST RESULTS")

    print(
        test_results.round(8).to_string(index=False)
    )

    # ---------------------------------------------------------------
    # Diagnostics
    # ---------------------------------------------------------------
    print_section("GENERATING FINAL MODEL DIAGNOSTICS")

    for target, content in final_predictions.items():
        y_true = np.asarray(
            content["y_true"],
            dtype=float,
        )
        y_pred = np.asarray(
            content["y_pred"],
            dtype=float,
        )

        residuals = y_true - y_pred

        plot_actual_vs_predicted(
            target,
            y_true,
            y_pred,
        )

        plot_residual_distribution(
            target,
            residuals,
        )

        plot_residual_vs_prediction(
            target,
            y_pred,
            residuals,
        )

    save_test_prediction_sample(
        final_predictions,
    )

    # ---------------------------------------------------------------
    # Feature importance
    # ---------------------------------------------------------------
    tree_importance = extract_tree_feature_importance(
        final_models,
    )

    # ---------------------------------------------------------------
    # Report
    # ---------------------------------------------------------------
    print_section("WRITING REGRESSION REPORT")

    create_markdown_report(
        validation_results=validation_results,
        selected_models=selected_models,
        test_results=test_results,
        tree_importance=tree_importance,
    )

    # ---------------------------------------------------------------
    # Completion
    # ---------------------------------------------------------------
    print_section("PHASE 5 COMPLETE")

    print(
        f"Figures: {FIGURE_DIR}\n"
        f"Tables:  {TABLE_DIR}\n"
        f"Report:  {REPORT_DIR / 'regression_summary.md'}\n"
        f"Models:  {MODEL_DIR}"
    )

    print(
        "\nNext step: Phase 6 — build the aerodynamic surrogate and "
        "reference/ML simulation engine."
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
