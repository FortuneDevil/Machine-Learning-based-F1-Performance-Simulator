"""
07_reference_model.py

Phase 7 of the F1 2026 Data-Driven Aerodynamic Simulator project.

Purpose
-------
Build the DATA-GROUNDED REFERENCE AERODYNAMIC MODEL that will later be
compared against the machine-learning surrogate inside the simulator.

Important conceptual distinction
---------------------------------
The supplied F1 dataset is a static synthetic input-output dataset. It does
not contain a time-dependent "real F1 trajectory" and it does not expose the
underlying physics-engine equations.

Therefore, this script does NOT claim to reconstruct the hidden physics
equations.

Instead, it creates a high-resolution empirical reference response directly
from the full dataset using local interpolation:

    continuous operating point
            ↓
    nearest observations in dataset
            ↓
      inverse-distance
       weighted average
            ↓
    reference aerodynamic response

This is the "dataset-grounded reference" used by the later simulator.

The ML surrogate from Phase 5 uses only the training portion of the data.
The reference interpolator uses the entire available dataset so that the
reference side represents the best available response contained in the CSV.

The reference model is therefore appropriate for:

    1. Generating reference Cd / Cl values at arbitrary points inside the
       observed operating envelope.
    2. Comparing reference responses against ML-surrogate responses.
    3. Generating a reference trajectory inside the later simulator.

It must NOT be described as an independent real-world measurement model.

Continuous inputs
-----------------
velocity_ms
flap_front_angle
flap_rear_angle
ride_height_front
ride_height_rear
battery_soc

Reference outputs
-----------------
cl_total
cd_total
cop_shift
energy_drain_kw

Interpolation method
--------------------
- Standardize all six continuous input dimensions.
- Build a KD-tree through sklearn NearestNeighbors.
- Find K nearest observations.
- Use inverse-distance weighting.
- Exact dataset points return the exact observed response.
- Queries outside the observed input envelope are rejected by default.

Outputs
-------
models/
└── reference_aero_model.pkl

reports/07_reference_model/
├── figures/
│   ├── reference_vs_actual_cd.png
│   ├── reference_vs_actual_cl.png
│   ├── reference_interpolation_error.png
│   └── reference_distance_distribution.png
│
├── tables/
│   ├── reference_leave_one_out_metrics.csv
│   ├── reference_query_distance_statistics.csv
│   └── reference_test_predictions_sample.csv
│
└── reference_model_summary.md

Run
---
python 07_reference_model.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

# Force a headless Matplotlib backend so the script never starts Tkinter.
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
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


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
    / "07_reference_model"
)

FIGURE_DIR = REPORT_DIR / "figures"
TABLE_DIR = REPORT_DIR / "tables"

REFERENCE_MODEL_PATH = (
    MODEL_DIR / "reference_aero_model.pkl"
)

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

# Number of neighbours used by the reference interpolator.
K_NEIGHBORS = 12

# Exponent used in inverse-distance weighting:
#
#     weight = 1 / distance^p
#
# p=2 gives stronger emphasis to the closest observations.
DISTANCE_POWER = 2.0

# Number of rows used when estimating leave-one-out interpolation quality.
# We do not evaluate all 100k observations because this is only a diagnostic.
VALIDATION_SAMPLE_SIZE = 5000

RANDOM_STATE = 42

# Use a small epsilon to prevent division by zero.
DISTANCE_EPSILON = 1e-12

# How far into the boundary can a diagnostic query be before it is
# considered outside. The actual reference model checks the exact min/max.
BOUNDARY_TOLERANCE = 1e-10


# =====================================================================
# Utility functions
# =====================================================================

def print_section(title: str) -> None:
    """Print a clear section heading."""
    print("\n" + "=" * 86)
    print(title)
    print("=" * 86)


def ensure_output_directories() -> None:
    """Create output directories."""
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)


def save_figure(
    fig: plt.Figure,
    filename: str,
) -> None:
    """Save a figure with consistent resolution."""
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

    headers = [
        str(column)
        for column in work.columns
    ]

    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(
            "---"
            for _ in headers
        ) + " |",
    ]

    for _, row in work.iterrows():
        values = [
            str(value).replace(
                "|",
                r"\|",
            )
            for value in row
        ]

        lines.append(
            "| " + " | ".join(values) + " |"
        )

    return "\n".join(lines)


def calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, float]:
    """Calculate standard regression metrics."""
    mse = mean_squared_error(
        y_true,
        y_pred,
    )

    return {
        "r2": float(
            r2_score(
                y_true,
                y_pred,
            )
        ),
        "rmse": float(
            np.sqrt(mse)
        ),
        "mae": float(
            mean_absolute_error(
                y_true,
                y_pred,
            )
        ),
        "max_absolute_error": float(
            np.max(
                np.abs(
                    y_true - y_pred
                )
            )
        ),
    }


def safe_filename(text: str) -> str:
    """Create a filesystem-friendly name."""
    return (
        text.lower()
        .replace(" ", "_")
        .replace("/", "_")
        .replace("-", "_")
    )


# =====================================================================
# Dataset validation
# =====================================================================

def check_dataset(
    df: pd.DataFrame,
) -> None:
    """Validate reference-model inputs and outputs."""
    required = (
        INPUT_COLUMNS
        + TARGET_COLUMNS
    )

    missing = [
        column
        for column in required
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            "The following required columns are missing:\n"
            + "\n".join(
                f"  - {column}"
                for column in missing
            )
        )

    non_numeric = [
        column
        for column in required
        if not pd.api.types.is_numeric_dtype(
            df[column]
        )
    ]

    if non_numeric:
        raise TypeError(
            "All reference-model variables must be numerical.\n"
            "Non-numeric columns:\n"
            + "\n".join(
                f"  - {column}"
                for column in non_numeric
            )
        )

    values = df[required].to_numpy(
        dtype=float
    )

    if not np.isfinite(values).all():
        raise ValueError(
            "Non-finite values were found in the dataset."
        )

    if df[INPUT_COLUMNS].duplicated().any():
        duplicate_count = int(
            df[INPUT_COLUMNS].duplicated().sum()
        )

        raise ValueError(
            "Duplicate input operating points were found: "
            f"{duplicate_count:,}"
        )


# =====================================================================
# Reference model
# =====================================================================

class ReferenceAerodynamicModel:
    """
    Dataset-grounded local interpolator for aerodynamic responses.

    The model operates entirely inside the observed six-dimensional
    continuous input envelope.

    Parameters
    ----------
    X:
        Full dataset input matrix.
    y:
        Full dataset target matrix.
    input_columns:
        Names of continuous input variables.
    target_columns:
        Names of continuous targets.
    k_neighbors:
        Number of neighbours used for interpolation.
    distance_power:
        Exponent p in inverse-distance weighting.
    """

    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        input_columns: list[str],
        target_columns: list[str],
        k_neighbors: int = K_NEIGHBORS,
        distance_power: float = DISTANCE_POWER,
    ) -> None:
        X = np.asarray(
            X,
            dtype=float,
        )

        y = np.asarray(
            y,
            dtype=float,
        )

        if X.ndim != 2:
            raise ValueError(
                "X must be a 2D array."
            )

        if y.ndim != 2:
            raise ValueError(
                "y must be a 2D array."
            )

        if X.shape[0] != y.shape[0]:
            raise ValueError(
                "X and y must contain the same number of rows."
            )

        if X.shape[1] != len(input_columns):
            raise ValueError(
                "Number of X columns does not match input_columns."
            )

        if y.shape[1] != len(target_columns):
            raise ValueError(
                "Number of y columns does not match target_columns."
            )

        if not np.isfinite(X).all():
            raise ValueError(
                "X contains non-finite values."
            )

        if not np.isfinite(y).all():
            raise ValueError(
                "y contains non-finite values."
            )

        if k_neighbors < 1:
            raise ValueError(
                "k_neighbors must be >= 1."
            )

        if k_neighbors >= len(X):
            raise ValueError(
                "k_neighbors must be smaller than the number of observations."
            )

        if distance_power <= 0:
            raise ValueError(
                "distance_power must be positive."
            )

        self.X_raw = X
        self.y = y

        self.input_columns = list(
            input_columns
        )
        self.target_columns = list(
            target_columns
        )

        self.k_neighbors = int(
            k_neighbors
        )

        self.distance_power = float(
            distance_power
        )

        self.scaler = StandardScaler()

        X_scaled = self.scaler.fit_transform(
            self.X_raw
        )

        self.X_scaled = X_scaled

        self.neighbor_search = NearestNeighbors(
            n_neighbors=self.k_neighbors,
            algorithm="auto",
            metric="euclidean",
            n_jobs=1,
        )

        self.neighbor_search.fit(
            self.X_scaled
        )

        self.feature_min = self.X_raw.min(
            axis=0
        )

        self.feature_max = self.X_raw.max(
            axis=0
        )

    # -----------------------------------------------------------------
    # Input handling
    # -----------------------------------------------------------------

    def _prepare_query(
        self,
        X_query: pd.DataFrame | np.ndarray,
    ) -> np.ndarray:
        """
        Convert a query to an array with correct feature ordering.
        """
        if isinstance(
            X_query,
            pd.DataFrame,
        ):
            missing = [
                col
                for col in self.input_columns
                if col not in X_query.columns
            ]

            if missing:
                raise ValueError(
                    "Query is missing input columns:\n"
                    + "\n".join(
                        f"  - {col}"
                        for col in missing
                    )
                )

            query = X_query[
                self.input_columns
            ].to_numpy(
                dtype=float
            )
        else:
            query = np.asarray(
                X_query,
                dtype=float,
            )

        if query.ndim == 1:
            query = query.reshape(
                1,
                -1,
            )

        if query.ndim != 2:
            raise ValueError(
                "Query must be a 1D or 2D numerical array."
            )

        if query.shape[1] != len(
            self.input_columns
        ):
            raise ValueError(
                "Query has the wrong number of input features. "
                f"Expected {len(self.input_columns)}."
            )

        if not np.isfinite(query).all():
            raise ValueError(
                "Query contains non-finite values."
            )

        return query

    def check_bounds(
        self,
        X_query: pd.DataFrame | np.ndarray,
    ) -> pd.DataFrame:
        """
        Check whether each query lies inside the observed input envelope.

        Returns a row-level report. This does not raise an exception.
        """
        query = self._prepare_query(
            X_query
        )

        rows = []

        for row_index, row in enumerate(
            query
        ):
            below = (
                row
                < (
                    self.feature_min
                    - BOUNDARY_TOLERANCE
                )
            )

            above = (
                row
                > (
                    self.feature_max
                    + BOUNDARY_TOLERANCE
                )
            )

            outside_columns = [
                self.input_columns[i]
                for i in range(
                    len(self.input_columns)
                )
                if below[i] or above[i]
            ]

            rows.append(
                {
                    "query_index": row_index,
                    "inside_observed_envelope":
                        len(outside_columns) == 0,
                    "outside_columns":
                        ", ".join(
                            outside_columns
                        ),
                }
            )

        return pd.DataFrame(
            rows
        )

    # -----------------------------------------------------------------
    # Prediction
    # -----------------------------------------------------------------

    def predict(
        self,
        X_query: pd.DataFrame | np.ndarray,
        reject_outside: bool = True,
        return_distance: bool = False,
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        """
        Predict reference outputs using inverse-distance weighting.

        Parameters
        ----------
        X_query:
            One or more continuous operating points.

        reject_outside:
            If True, reject points outside the observed six-dimensional
            feature envelope. Extrapolation is intentionally avoided.

        return_distance:
            If True, also return the distance to the nearest neighbour.

        Returns
        -------
        predictions:
            Array of shape (n_queries, n_targets).

        nearest_distance:
            Returned only when return_distance=True.
        """
        query = self._prepare_query(
            X_query
        )

        bounds = self.check_bounds(
            query
        )

        if reject_outside:
            invalid = ~bounds[
                "inside_observed_envelope"
            ].to_numpy()

            if invalid.any():
                bad_indices = np.where(
                    invalid
                )[0]

                raise ValueError(
                    "Reference-model query contains points outside the "
                    "observed continuous operating envelope. "
                    f"Invalid query indices: {bad_indices.tolist()}"
                )

        query_scaled = self.scaler.transform(
            query
        )

        distances, indices = (
            self.neighbor_search.kneighbors(
                query_scaled,
                n_neighbors=self.k_neighbors,
                return_distance=True,
            )
        )

        neighbor_values = self.y[
            indices
        ]

        predictions = np.empty(
            (
                len(query),
                self.y.shape[1],
            ),
            dtype=float,
        )

        for i in range(
            len(query)
        ):
            d = distances[i]

            # If the query exactly matches a dataset point, return that
            # observation exactly. This is important for reference runs
            # initialized from an observed operating point.
            exact = (
                d
                <= DISTANCE_EPSILON
            )

            if exact.any():
                first_exact = int(
                    np.where(
                        exact
                    )[0][0]
                )

                predictions[i] = (
                    neighbor_values[
                        i,
                        first_exact
                    ]
                )

                continue

            weights = (
                1.0
                / np.power(
                    d,
                    self.distance_power,
                )
            )

            weights /= weights.sum()

            predictions[i] = np.sum(
                neighbor_values[i]
                * weights[:, None],
                axis=0,
            )

        nearest_distance = distances[
            :, 0
        ]

        if return_distance:
            return (
                predictions,
                nearest_distance,
            )

        return predictions

    def predict_dataframe(
        self,
        X_query: pd.DataFrame,
        reject_outside: bool = True,
    ) -> pd.DataFrame:
        """Predict and return named target columns."""
        prediction = self.predict(
            X_query,
            reject_outside=reject_outside,
        )

        return pd.DataFrame(
            prediction,
            columns=self.target_columns,
            index=X_query.index,
        )


# =====================================================================
# Leave-one-out local interpolation diagnostic
# =====================================================================

def leave_one_out_reference_validation(
    model: ReferenceAerodynamicModel,
    df: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    np.ndarray,
]:
    """
    Estimate local interpolation quality.

    For a sampled set of observations, query the k+1 nearest neighbours so
    that the point itself can be removed. This prevents the interpolation
    diagnostic from trivially returning the exact target due to self-match.

    Note:
        This diagnostic evaluates how well local interpolation reconstructs
        an observed point from surrounding observations. It is NOT a model
        selection metric for the ML surrogate.
    """
    print_section(
        "2. LEAVE-ONE-OUT REFERENCE INTERPOLATION DIAGNOSTIC"
    )

    sample_n = min(
        VALIDATION_SAMPLE_SIZE,
        len(df),
    )

    rng = np.random.default_rng(
        RANDOM_STATE
    )

    sample_indices = rng.choice(
        len(df),
        size=sample_n,
        replace=False,
    )

    X_sample = df.iloc[
        sample_indices
    ][INPUT_COLUMNS]

    y_sample = df.iloc[
        sample_indices
    ][TARGET_COLUMNS]

    # We need one extra neighbour because the first neighbour is the point
    # itself when querying an observation from the reference dataset.
    distances, indices = (
        model.neighbor_search.kneighbors(
            model.scaler.transform(
                X_sample
            ),
            n_neighbors=model.k_neighbors + 1,
            return_distance=True,
        )
    )

    predictions = np.empty(
        (
            sample_n,
            len(TARGET_COLUMNS),
        ),
        dtype=float,
    )

    nearest_nonself_distance = (
        np.empty(
            sample_n,
            dtype=float,
        )
    )

    for i in range(
        sample_n
    ):
        d = distances[i]
        idx = indices[i]

        # Remove the exact self-neighbour.
        nonself_mask = (
            d > DISTANCE_EPSILON
        )

        d = d[
            nonself_mask
        ][: model.k_neighbors]

        idx = idx[
            nonself_mask
        ][: model.k_neighbors]

        if len(d) == 0:
            raise RuntimeError(
                "Unable to identify non-self neighbours."
            )

        weights = (
            1.0
            / np.power(
                d,
                model.distance_power,
            )
        )

        weights /= weights.sum()

        predictions[i] = np.sum(
            model.y[idx]
            * weights[:, None],
            axis=0,
        )

        nearest_nonself_distance[i] = d[0]

    result_rows = []

    for target_index, target in enumerate(
        TARGET_COLUMNS
    ):
        actual = y_sample[
            target
        ].to_numpy(
            dtype=float
        )

        predicted = predictions[
            :,
            target_index,
        ]

        metrics = calculate_metrics(
            actual,
            predicted,
        )

        result_rows.append(
            {
                "target": target,
                "r2": metrics["r2"],
                "rmse": metrics["rmse"],
                "mae": metrics["mae"],
                "max_absolute_error":
                    metrics[
                        "max_absolute_error"
                    ],
                "n_validation":
                    sample_n,
            }
        )

    metrics_table = pd.DataFrame(
        result_rows
    )

    distance_table = pd.DataFrame(
        {
            "nearest_nonself_distance":
                nearest_nonself_distance,
        }
    )

    # Add descriptive distance statistics.
    distance_summary = pd.DataFrame(
        {
            "statistic": [
                "min",
                "1%",
                "5%",
                "25%",
                "50%",
                "75%",
                "95%",
                "99%",
                "max",
                "mean",
                "std",
            ],
            "distance": [
                np.min(
                    nearest_nonself_distance
                ),
                np.quantile(
                    nearest_nonself_distance,
                    0.01,
                ),
                np.quantile(
                    nearest_nonself_distance,
                    0.05,
                ),
                np.quantile(
                    nearest_nonself_distance,
                    0.25,
                ),
                np.quantile(
                    nearest_nonself_distance,
                    0.50,
                ),
                np.quantile(
                    nearest_nonself_distance,
                    0.75,
                ),
                np.quantile(
                    nearest_nonself_distance,
                    0.95,
                ),
                np.quantile(
                    nearest_nonself_distance,
                    0.99,
                ),
                np.max(
                    nearest_nonself_distance
                ),
                np.mean(
                    nearest_nonself_distance
                ),
                np.std(
                    nearest_nonself_distance
                ),
            ],
        }
    )

    save_table(
        metrics_table.round(10),
        "reference_leave_one_out_metrics.csv",
    )

    save_table(
        distance_summary.round(10),
        "reference_query_distance_statistics.csv",
    )

    prediction_table = X_sample.copy()

    for target in TARGET_COLUMNS:
        prediction_table[
            f"{target}_actual"
        ] = y_sample[
            target
        ].to_numpy(
            dtype=float
        )

        prediction_table[
            f"{target}_reference"
        ] = predictions[
            :,
            TARGET_COLUMNS.index(
                target
            ),
        ]

        prediction_table[
            f"{target}_residual"
        ] = (
            prediction_table[
                f"{target}_actual"
            ]
            - prediction_table[
                f"{target}_reference"
            ]
        )

    prediction_table[
        "nearest_nonself_distance"
    ] = nearest_nonself_distance

    save_table(
        prediction_table.round(10),
        "reference_test_predictions_sample.csv",
        index=False,
    )

    return (
        metrics_table,
        distance_summary,
        nearest_nonself_distance,
    )


# =====================================================================
# Diagnostic plots
# =====================================================================

def plot_reference_vs_actual(
    target: str,
    actual: np.ndarray,
    predicted: np.ndarray,
) -> None:
    """Plot actual observations against local-reference reconstruction."""
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
        alpha=0.20,
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
        f"Reference {target}"
    )

    ax.set_title(
        f"Local Reference Reconstruction — {target}",
        fontsize=15,
    )

    annotation = (
        f"R² = {metrics['r2']:.6f}\n"
        f"RMSE = {metrics['rmse']:.6g}\n"
        f"MAE = {metrics['mae']:.6g}"
    )

    ax.text(
        0.04,
        0.96,
        annotation,
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
        (
            "reference_vs_actual_"
            f"{safe_filename(target)}.png"
        ),
    )


def plot_reference_interpolation_error(
    predictions: np.ndarray,
    actual: np.ndarray,
) -> None:
    """Plot the absolute error distribution of local reference reconstruction."""
    errors = np.abs(
        actual - predictions
    )

    combined = errors.ravel()

    fig, ax = plt.subplots(
        figsize=(10, 6)
    )

    sns.histplot(
        combined,
        bins=60,
        kde=True,
        ax=ax,
    )

    ax.set_xlabel(
        "Absolute interpolation error"
    )
    ax.set_ylabel("Count")
    ax.set_title(
        "Local Reference Interpolation Error",
        fontsize=16,
    )
    ax.grid(alpha=0.2)

    save_figure(
        fig,
        "reference_interpolation_error.png",
    )


def plot_distance_distribution(
    nearest_nonself_distance: np.ndarray,
) -> None:
    """Plot nearest-neighbour distances for the reference diagnostic."""
    fig, ax = plt.subplots(
        figsize=(10, 6)
    )

    sns.histplot(
        nearest_nonself_distance,
        bins=60,
        kde=True,
        ax=ax,
    )

    ax.set_xlabel(
        "Standardized distance to nearest non-self observation"
    )
    ax.set_ylabel("Count")
    ax.set_title(
        "Local Data Density — Nearest-Neighbour Distance",
        fontsize=16,
    )
    ax.grid(alpha=0.2)

    save_figure(
        fig,
        "reference_distance_distribution.png",
    )


# =====================================================================
# Benchmark
# =====================================================================

def benchmark_reference_speed(
    model: ReferenceAerodynamicModel,
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Benchmark repeated reference predictions."""
    print_section(
        "3. REFERENCE INTERPOLATION SPEED"
    )

    n = min(
        10000,
        len(df),
    )

    query = df.iloc[
        :n
    ][INPUT_COLUMNS]

    # Warm-up.
    model.predict(
        query.iloc[:100]
    )

    start = time.perf_counter()

    _ = model.predict(
        query
    )

    elapsed = (
        time.perf_counter()
        - start
    )

    predictions_per_second = (
        n / elapsed
        if elapsed > 0
        else np.nan
    )

    result = pd.DataFrame(
        [
            {
                "n_predictions":
                    n,
                "elapsed_seconds":
                    elapsed,
                "predictions_per_second":
                    predictions_per_second,
                "milliseconds_per_prediction":
                    elapsed / n * 1000,
            }
        ]
    )

    save_table(
        result.round(8),
        "reference_prediction_speed.csv",
    )

    print(
        f"Predictions: {n:,}"
    )
    print(
        f"Time: {elapsed:.4f} s"
    )
    print(
        f"Rate: {predictions_per_second:,.0f} predictions/s"
    )

    return result


# =====================================================================
# Markdown report
# =====================================================================

def create_markdown_report(
    df: pd.DataFrame,
    metrics_table: pd.DataFrame,
    distance_summary: pd.DataFrame,
    speed_table: pd.DataFrame,
) -> None:
    """Write a detailed reference-model report."""
    report_path = (
        REPORT_DIR
        / "reference_model_summary.md"
    )

    lines = [
        "# F1 2026 Mode-1 Dataset — Reference Aerodynamic Model",
        "",
        "## Purpose",
        "",
        "The reference model is a data-grounded interpolation layer used to "
        "provide a reference aerodynamic response to the simulator. It does "
        "not reconstruct the hidden equations of the synthetic physics "
        "engine and must not be interpreted as independent real-world data.",
        "",
        "## Dataset",
        "",
        f"- Observations: {len(df):,}",
        "- Continuous inputs: 6",
        "- Continuous outputs: 4",
        "- Aero mode: fixed to Mode 1",
        "",
        "## Inputs",
        "",
    ]

    for feature in INPUT_COLUMNS:
        lines.append(
            f"- `{feature}`"
        )

    lines.extend(
        [
            "",
            "## Outputs",
            "",
        ]
    )

    for target in TARGET_COLUMNS:
        lines.append(
            f"- `{target}`"
        )

    lines.extend(
        [
            "",
            "## Reference interpolation method",
            "",
            f"- Standardization: `StandardScaler`",
            f"- Neighbours: `{K_NEIGHBORS}`",
            f"- Metric: Euclidean distance in standardized input space",
            f"- Weighting: inverse distance^{DISTANCE_POWER:.1f}",
            "- Exact dataset points return their exact stored response",
            "- Queries outside the observed feature envelope are rejected",
            "",
            "The purpose is to provide a smooth, local, data-grounded "
            "reference response for arbitrary simulator states within the "
            "observed operating domain.",
            "",
            "## Leave-one-out local interpolation diagnostic",
            "",
        ]
    )

    lines.append(
        dataframe_to_markdown_table(
            metrics_table,
            index=False,
            float_digits=8,
        )
    )

    lines.extend(
        [
            "",
            "The leave-one-out diagnostic excludes each observation itself "
            "from its local neighbourhood. It therefore measures how well "
            "the response can be reconstructed from nearby observations, "
            "rather than trivially returning the observation itself.",
            "",
            "## Nearest-neighbour distance statistics",
            "",
        ]
    )

    lines.append(
        dataframe_to_markdown_table(
            distance_summary,
            index=False,
            float_digits=8,
        )
    )

    lines.extend(
        [
            "",
            "## Reference prediction speed",
            "",
        ]
    )

    lines.append(
        dataframe_to_markdown_table(
            speed_table,
            index=False,
            float_digits=6,
        )
    )

    lines.extend(
        [
            "",
            "## Role in the simulator",
            "",
            "The later simulator will use two aerodynamic response paths:",
            "",
            "1. **Reference path:** local interpolation from the complete "
            "dataset.",
            "2. **ML surrogate path:** the saved Phase-5 regression models.",
            "",
            "Both paths receive the same continuous operating state. Their "
            "predicted Cd/Cl values are then passed through the same "
            "aerodynamic force equations and vehicle-dynamics equations. "
            "This isolates the effect of replacing the data-grounded "
            "response with the learned ML surrogate.",
            "",
            "## Important limitation",
            "",
            "Because the source dataset is synthetic and static, the reference "
            "model is not an independent measurement of a real F1 car and "
            "should not be presented as such.",
            "",
            "## Next phase",
            "",
            "Connect this reference response and the Phase-5 ML surrogate "
            "to the aerodynamic force equations and construct the vehicle "
            "longitudinal-dynamics simulation.",
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
        "F1 2026 AERODYNAMIC SIMULATOR — PHASE 7"
    )

    if not DATA_PATH.exists():
        print(
            "ERROR: Dataset not found.\n"
            f"Expected: {DATA_PATH}"
        )
        return 1

    ensure_output_directories()

    # ---------------------------------------------------------------
    # Load dataset
    # ---------------------------------------------------------------
    print_section(
        "1. LOADING FULL REFERENCE DATASET"
    )

    try:
        df = pd.read_csv(
            DATA_PATH
        )
    except Exception as exc:
        print(
            f"ERROR: Could not load dataset: {exc}"
        )
        return 1

    try:
        check_dataset(df)
    except (ValueError, TypeError) as exc:
        print(
            f"ERROR: {exc}"
        )
        return 1

    print(
        f"Loaded: {DATA_PATH.name}\n"
        f"Rows:   {len(df):,}\n"
        f"Inputs: {len(INPUT_COLUMNS)}\n"
        f"Targets:{len(TARGET_COLUMNS)}"
    )

    # ---------------------------------------------------------------
    # Create reference model
    # ---------------------------------------------------------------
    print_section(
        "BUILDING DATA-GROUNDED REFERENCE MODEL"
    )

    X = df[
        INPUT_COLUMNS
    ].to_numpy(
        dtype=float
    )

    y = df[
        TARGET_COLUMNS
    ].to_numpy(
        dtype=float
    )

    start = time.perf_counter()

    reference_model = (
        ReferenceAerodynamicModel(
            X=X,
            y=y,
            input_columns=INPUT_COLUMNS,
            target_columns=TARGET_COLUMNS,
            k_neighbors=K_NEIGHBORS,
            distance_power=DISTANCE_POWER,
        )
    )

    build_time = (
        time.perf_counter()
        - start
    )

    print(
        f"Reference model built in "
        f"{build_time:.4f} s"
    )

    print(
        f"K = {K_NEIGHBORS}"
    )

    print(
        f"Distance power = "
        f"{DISTANCE_POWER:.1f}"
    )

    # ---------------------------------------------------------------
    # Save reference model
    # ---------------------------------------------------------------
    bundle = {
        "model": reference_model,
        "description": (
            "Dataset-grounded inverse-distance weighted KNN "
            "reference aerodynamic model."
        ),
        "input_columns": INPUT_COLUMNS,
        "target_columns": TARGET_COLUMNS,
        "k_neighbors": K_NEIGHBORS,
        "distance_power": DISTANCE_POWER,
        "dataset_name": DATA_PATH.name,
        "dataset_rows": len(df),
    }

    joblib.dump(
        bundle,
        REFERENCE_MODEL_PATH,
    )

    print(
        f"Saved reference model:\n  "
        f"{REFERENCE_MODEL_PATH}"
    )

    # ---------------------------------------------------------------
    # Leave-one-out interpolation diagnostic
    # ---------------------------------------------------------------
    (
        metrics_table,
        distance_summary,
        nearest_nonself_distance,
    ) = leave_one_out_reference_validation(
        reference_model,
        df,
    )

    print(
        metrics_table.round(8).to_string(
            index=False
        )
    )

    # ---------------------------------------------------------------
    # Reconstruct sampled predictions for plots
    # ---------------------------------------------------------------
    rng = np.random.default_rng(
        RANDOM_STATE
    )

    sample_n = min(
        VALIDATION_SAMPLE_SIZE,
        len(df),
    )

    sample_indices = rng.choice(
        len(df),
        size=sample_n,
        replace=False,
    )

    X_sample = df.iloc[
        sample_indices
    ][INPUT_COLUMNS]

    y_sample = df.iloc[
        sample_indices
    ][TARGET_COLUMNS]

    # Get reference predictions directly. The sampled observations are
    # exact data points, so this prediction is exact. For the error plots
    # we instead need the leave-one-out predictions recreated below.
    distances, indices = (
        reference_model.neighbor_search.kneighbors(
            reference_model.scaler.transform(
                X_sample
            ),
            n_neighbors=K_NEIGHBORS + 1,
            return_distance=True,
        )
    )

    loo_predictions = np.empty(
        (
            sample_n,
            len(TARGET_COLUMNS),
        ),
        dtype=float,
    )

    for i in range(
        sample_n
    ):
        d = distances[i]
        idx = indices[i]

        nonself = (
            d > DISTANCE_EPSILON
        )

        d = d[
            nonself
        ][:K_NEIGHBORS]

        idx = idx[
            nonself
        ][:K_NEIGHBORS]

        weights = (
            1.0
            / np.power(
                d,
                DISTANCE_POWER,
            )
        )

        weights /= weights.sum()

        loo_predictions[i] = np.sum(
            reference_model.y[idx]
            * weights[:, None],
            axis=0,
        )

    # ---------------------------------------------------------------
    # Plots
    # ---------------------------------------------------------------
    print_section(
        "GENERATING REFERENCE-MODEL DIAGNOSTICS"
    )

    for target_index, target in enumerate(
        TARGET_COLUMNS
    ):
        actual = y_sample[
            target
        ].to_numpy(
            dtype=float
        )

        predicted = loo_predictions[
            :,
            target_index,
        ]

        plot_reference_vs_actual(
            target,
            actual,
            predicted,
        )

    plot_reference_interpolation_error(
        loo_predictions,
        y_sample[
            TARGET_COLUMNS
        ].to_numpy(
            dtype=float
        ),
    )

    plot_distance_distribution(
        nearest_nonself_distance
    )

    # ---------------------------------------------------------------
    # Speed benchmark
    # ---------------------------------------------------------------
    speed_table = benchmark_reference_speed(
        reference_model,
        df,
    )

    # ---------------------------------------------------------------
    # Input envelope report
    # ---------------------------------------------------------------
    print_section(
        "REFERENCE OPERATING ENVELOPE"
    )

    envelope = pd.DataFrame(
        {
            "feature": INPUT_COLUMNS,
            "minimum": reference_model.feature_min,
            "maximum": reference_model.feature_max,
        }
    )

    print(
        envelope.round(6).to_string(
            index=False
        )
    )

    save_table(
        envelope.round(10),
        "reference_input_envelope.csv",
        index=False,
    )

    # ---------------------------------------------------------------
    # Final report
    # ---------------------------------------------------------------
    print_section(
        "WRITING REFERENCE MODEL REPORT"
    )

    create_markdown_report(
        df=df,
        metrics_table=metrics_table,
        distance_summary=distance_summary,
        speed_table=speed_table,
    )

    # ---------------------------------------------------------------
    # Completion
    # ---------------------------------------------------------------
    print_section(
        "PHASE 7 COMPLETE"
    )

    print(
        f"Reference model:\n  "
        f"{REFERENCE_MODEL_PATH}\n\n"
        f"Figures:\n  "
        f"{FIGURE_DIR}\n\n"
        f"Tables:\n  "
        f"{TABLE_DIR}\n\n"
        f"Report:\n  "
        f"{REPORT_DIR / 'reference_model_summary.md'}"
    )

    print(
        "\nNext step: Phase 8 — aerodynamic force and vehicle "
        "longitudinal-dynamics engine."
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
