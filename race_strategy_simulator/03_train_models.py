"""
03_train_models.py

Train and compare lap-time models.

Target
------
delta_pct: how much slower or faster a lap is than the driver's own baseline
pace, in percent. See 02_build_dataset.py.

Split
-----
Temporal, by season. Models are compared on the validation season, the
winner is refitted on training + validation, and the test season is touched
exactly once. A random split would put laps from the same race on both
sides and overstate accuracy.

Outputs
-------
models/lap_time_model.pkl
reports/03_models/validation_comparison.csv
reports/03_models/test_results.csv
reports/03_models/*.png

Run
---
python 03_train_models.py
"""

from __future__ import annotations

import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    OneHotEncoder,
    OrdinalEncoder,
    PolynomialFeatures,
    StandardScaler,
)

from common import (
    DRY_COMPOUNDS,
    MODEL_DIR,
    MODEL_LAPS_PATH,
    MODEL_PATH,
    PIT_LOSS_PATH,
    RANDOM_STATE,
    REPORT_DIR,
)
from plot_style import (
    ACTUAL_COLOR,
    COMPOUND_COLORS,
    NEUTRAL_COLOR,
    TEXT_SECONDARY,
    apply_matplotlib_style,
)
from strategy import CATEGORICAL_FEATURES, FEATURES, NUMERIC_FEATURES, TARGET

apply_matplotlib_style()

import matplotlib.pyplot as plt  # noqa: E402

OUT_DIR = REPORT_DIR / "03_models"

# Stints are planned only up to this quantile of observed tyre age at each
# circuit. The very longest stints are rare and come from cars whose tyres
# happened to last, so the model is over-optimistic there: allowing them
# (quantile 1.0) moved the recommended first stop further from what teams
# do (10.5 laps away on average instead of 8.0).
MAX_TYRE_LIFE_QUANTILE = 0.98

RACE_DRIVER_KEYS = ["Year", "Round", "Driver"]
STINT_KEYS = RACE_DRIVER_KEYS + ["Stint"]


# =====================================================================
# Candidate models
# =====================================================================


def one_hot() -> OneHotEncoder:
    return OneHotEncoder(handle_unknown="ignore", sparse_output=False)


def gradient_boosting(monotonic_tyre_wear: bool) -> Pipeline:
    """Gradient-boosted trees with native categorical handling."""
    encoder = ColumnTransformer(
        [
            ("numeric", "passthrough", NUMERIC_FEATURES),
            (
                "categorical",
                OrdinalEncoder(
                    handle_unknown="use_encoded_value",
                    unknown_value=np.nan,
                ),
                CATEGORICAL_FEATURES,
            ),
        ]
    )

    n_numeric = len(NUMERIC_FEATURES)
    n_features = n_numeric + len(CATEGORICAL_FEATURES)

    # Optionally force predicted lap time to never improve as a tyre ages.
    monotonic = [0] * n_features
    if monotonic_tyre_wear:
        monotonic[NUMERIC_FEATURES.index("tyre_life")] = 1

    # Deliberately small trees. Lap times are noisy and there are only a
    # few races per circuit, so larger trees (31 leaves) fitted race-specific
    # noise and did worse on the validation season than plain linear
    # regression.
    model = HistGradientBoostingRegressor(
        max_iter=300,
        learning_rate=0.03,
        max_leaf_nodes=4,
        min_samples_leaf=300,
        l2_regularization=5.0,
        categorical_features=list(range(n_numeric, n_features)),
        monotonic_cst=monotonic,
        random_state=RANDOM_STATE,
    )

    return Pipeline([("encode", encoder), ("model", model)])


def create_models() -> dict[str, Pipeline]:
    linear_inputs = ColumnTransformer(
        [
            ("numeric", StandardScaler(), NUMERIC_FEATURES),
            ("compound", one_hot(), ["compound", "baseline_compound"]),
        ]
    )

    # Same linear model, but tyre age interacts with the compound, so each
    # compound gets its own wear rate and curvature.
    other_numeric = [f for f in NUMERIC_FEATURES if f != "tyre_life"]

    wear_inputs = ColumnTransformer(
        [
            (
                "wear",
                Pipeline(
                    [
                        (
                            "inputs",
                            ColumnTransformer(
                                [
                                    ("age", StandardScaler(), ["tyre_life"]),
                                    ("compound", one_hot(), ["compound"]),
                                ]
                            ),
                        ),
                        (
                            "poly",
                            PolynomialFeatures(degree=2, include_bias=False),
                        ),
                    ]
                ),
                ["tyre_life", "compound"],
            ),
            ("numeric", StandardScaler(), other_numeric),
            ("baseline", one_hot(), ["baseline_compound"]),
        ]
    )

    return {
        "No change from baseline": Pipeline(
            [("model", DummyRegressor(strategy="constant", constant=0.0))]
        ),
        "Mean change": Pipeline(
            [("model", DummyRegressor(strategy="mean"))]
        ),
        "Linear regression": Pipeline(
            [("inputs", linear_inputs), ("model", LinearRegression())]
        ),
        "Linear, wear per compound": Pipeline(
            [("inputs", wear_inputs), ("model", Ridge(alpha=1.0))]
        ),
        "Gradient boosting": gradient_boosting(monotonic_tyre_wear=False),
        "Gradient boosting (monotonic tyre wear)": gradient_boosting(
            monotonic_tyre_wear=True
        ),
    }


# =====================================================================
# Metrics
# =====================================================================


def calculate_metrics(
    data: pd.DataFrame,
    prediction: np.ndarray,
) -> dict[str, float]:
    """Errors in percent of baseline pace and converted to seconds."""
    actual = data[TARGET].to_numpy(dtype=float)
    error_pct = prediction - actual
    error_s = error_pct / 100.0 * data["baseline_time"].to_numpy(dtype=float)

    return {
        "rmse_pct": float(np.sqrt(np.mean(error_pct**2))),
        "mae_pct": float(mean_absolute_error(actual, prediction)),
        "rmse_s": float(np.sqrt(np.mean(error_s**2))),
        "mae_s": float(np.mean(np.abs(error_s))),
        "r2": float(r2_score(actual, prediction)),
    }


def residual_components(
    data: pd.DataFrame,
    prediction: np.ndarray,
) -> dict[str, float]:
    """
    Split out-of-sample error into three nested parts.

    race_sd_pct: spread of a driver's average error over a whole race.
        Mostly a mis-estimated baseline pace. It shifts every strategy for
        that driver equally, so it does not change which strategy is best.
    stint_sd_pct: spread of a stint's average error around the driver's
        race average. This is what separates strategies.
    lap_sd_pct: spread of single laps around their own stint's average.
    """
    residual = pd.Series(
        data[TARGET].to_numpy(dtype=float) - prediction,
        index=data.index,
    )
    race_groups = [data[k] for k in RACE_DRIVER_KEYS]
    stint_groups = [data[k] for k in STINT_KEYS]

    race_mean = residual.groupby(race_groups).transform("mean")
    stint_mean = residual.groupby(stint_groups).transform("mean")

    return {
        "race_sd_pct": float(residual.groupby(race_groups).mean().std()),
        "stint_sd_pct": float(
            (stint_mean - race_mean).groupby(stint_groups).first().std()
        ),
        "lap_sd_pct": float((residual - stint_mean).std()),
    }


# =====================================================================
# Figures
# =====================================================================


def plot_validation_comparison(results: pd.DataFrame) -> None:
    ordered = results.sort_values("rmse_s", ascending=False)

    fig, ax = plt.subplots(figsize=(8, 3.6))
    bars = ax.barh(ordered["model"], ordered["rmse_s"], color=ACTUAL_COLOR,
                   height=0.6)
    ax.bar_label(bars, fmt="%.3f s", padding=4, color=TEXT_SECONDARY)

    ax.set_xlabel("Validation RMSE (seconds per lap)")
    ax.set_title("Lap-time error by model, 2024 validation season")
    ax.grid(axis="y", visible=False)
    ax.set_xlim(0, ordered["rmse_s"].max() * 1.15)

    fig.savefig(OUT_DIR / "validation_rmse_by_model.png")
    plt.close(fig)


def plot_predicted_vs_actual(test: pd.DataFrame, prediction: np.ndarray) -> None:
    sample = test.assign(prediction=prediction).sample(
        n=min(6000, len(test)), random_state=RANDOM_STATE
    )

    low = float(np.quantile(test[TARGET], 0.002))
    high = float(np.quantile(test[TARGET], 0.998))

    fig, ax = plt.subplots(figsize=(5.6, 5.6))
    ax.scatter(sample[TARGET], sample["prediction"], s=6, alpha=0.25,
               color=ACTUAL_COLOR, linewidths=0)
    ax.plot([low, high], [low, high], color=NEUTRAL_COLOR, linewidth=1.2,
            linestyle="--")

    ax.set_xlim(low, high)
    ax.set_ylim(low, high)
    ax.set_xlabel("Actual change from baseline pace (%)")
    ax.set_ylabel("Predicted change from baseline pace (%)")
    ax.set_title("Predicted vs actual lap-time change, 2025 test season")

    fig.savefig(OUT_DIR / "test_predicted_vs_actual.png")
    plt.close(fig)


def plot_degradation_curves(model: Pipeline, history: pd.DataFrame) -> None:
    """
    Predicted lap-time change as tyres age, one line per compound.

    Fuel is held fixed at mid-race so the curve shows tyre wear alone, for
    the circuit with the most data.
    """
    circuit = history["circuit"].value_counts().index[0]
    at_circuit = history[history["circuit"] == circuit]

    fig, ax = plt.subplots(figsize=(8, 4.4))

    for compound in DRY_COMPOUNDS:
        observed = at_circuit[at_circuit["compound"] == compound]
        if observed.empty:
            continue

        ages = np.arange(2, int(observed["tyre_life"].quantile(0.98)) + 1)

        query = pd.DataFrame(
            {
                "tyre_life": ages.astype(float),
                "laps_since_baseline_frac": 0.5,
                "lap_frac": 0.5,
                "total_laps": float(at_circuit["total_laps"].median()),
                "baseline_tyre_life": 5.0,
                "baseline_track_temp": float(
                    at_circuit["baseline_track_temp"].median()
                ),
                "compound": compound,
                "baseline_compound": "MEDIUM",
                "circuit": circuit,
            }
        )

        curve = model.predict(query[FEATURES])
        ax.plot(ages, curve, color=COMPOUND_COLORS[compound],
                label=compound.title())

    ax.set_xlabel("Tyre age (laps)")
    ax.set_ylabel("Predicted change from baseline pace (%)")
    ax.set_title(f"Learned tyre degradation at {circuit} (fuel held at mid-race)")
    # The curves run close together, so end-of-line labels would collide;
    # the legend sits in the empty lower-right corner instead.
    ax.legend(loc="lower right")

    fig.savefig(OUT_DIR / "degradation_curves.png")
    plt.close(fig)


def plot_residuals(test: pd.DataFrame, prediction: np.ndarray) -> None:
    error_s = (prediction - test[TARGET]) / 100.0 * test["baseline_time"]
    limit = float(np.quantile(np.abs(error_s), 0.995))

    fig, ax = plt.subplots(figsize=(8, 3.8))
    ax.hist(error_s, bins=80, range=(-limit, limit), color=ACTUAL_COLOR,
            edgecolor="#fcfcfb", linewidth=0.4)
    ax.axvline(0, color=NEUTRAL_COLOR, linewidth=1.2)

    ax.set_xlabel("Prediction error (seconds per lap, predicted minus actual)")
    ax.set_ylabel("Laps")
    ax.set_title("Lap-time prediction error, 2025 test season")

    fig.savefig(OUT_DIR / "test_error_distribution.png")
    plt.close(fig)


# =====================================================================
# Main
# =====================================================================


def main() -> int:
    if not MODEL_LAPS_PATH.exists():
        print(f"ERROR: {MODEL_LAPS_PATH} not found. "
              "Run 02_build_dataset.py first.")
        return 1

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    laps = pd.read_csv(MODEL_LAPS_PATH)

    train = laps[laps["Split"] == "train"]
    validation = laps[laps["Split"] == "validation"]
    test = laps[laps["Split"] == "test"]
    history = pd.concat([train, validation])

    print(f"Train laps:      {len(train):,}")
    print(f"Validation laps: {len(validation):,}")
    print(f"Test laps:       {len(test):,}")

    # -----------------------------------------------------------------
    # Compare candidates on the validation season
    # -----------------------------------------------------------------
    rows = []
    validation_predictions = {}

    for name, model in create_models().items():
        model.fit(train[FEATURES], train[TARGET])
        prediction = model.predict(validation[FEATURES])
        validation_predictions[name] = prediction
        rows.append({"model": name, **calculate_metrics(validation, prediction)})

    validation_results = pd.DataFrame(rows).sort_values("rmse_s")
    validation_results.to_csv(OUT_DIR / "validation_comparison.csv", index=False)

    print("\nValidation comparison (2024):\n")
    print(validation_results.round(4).to_string(index=False))

    best_name = validation_results.iloc[0]["model"]
    print(f"\nSelected model: {best_name}")

    residual = residual_components(
        validation, validation_predictions[best_name]
    )

    # -----------------------------------------------------------------
    # Refit on train + validation, evaluate once on the test season
    # -----------------------------------------------------------------
    final_model = create_models()[best_name]
    final_model.fit(history[FEATURES], history[TARGET])

    test_prediction = final_model.predict(test[FEATURES])

    reference = create_models()["No change from baseline"]
    reference.fit(history[FEATURES], history[TARGET])

    test_results = pd.DataFrame(
        [
            {"model": best_name, **calculate_metrics(test, test_prediction)},
            {
                "model": "No change from baseline",
                **calculate_metrics(test, reference.predict(test[FEATURES])),
            },
        ]
    )
    test_results.to_csv(OUT_DIR / "test_results.csv", index=False)

    print("\nTest results (2025, untouched until now):\n")
    print(test_results.round(4).to_string(index=False))

    # -----------------------------------------------------------------
    # Supporting tables for the simulator
    # -----------------------------------------------------------------
    max_tyre_life = (
        history.groupby(["circuit", "compound"])["tyre_life"]
        .quantile(MAX_TYRE_LIFE_QUANTILE)
        .to_dict()
    )
    for compound, value in (
        history.groupby("compound")["tyre_life"]
        .quantile(MAX_TYRE_LIFE_QUANTILE)
        .items()
    ):
        max_tyre_life[("ALL", compound)] = value

    pit_loss = (
        pd.read_csv(PIT_LOSS_PATH).set_index("Circuit")["pit_loss_s"].to_dict()
    )

    bundle = {
        "model": final_model,
        "model_name": best_name,
        "features": FEATURES,
        "target": TARGET,
        "residual": residual,
        "pit_loss": pit_loss,
        "max_tyre_life": max_tyre_life,
        "validation_results": validation_results,
        "test_results": test_results,
    }
    joblib.dump(bundle, MODEL_PATH)

    print(f"\nError components from validation: "
          f"driver-race {residual['race_sd_pct']:.3f}%, "
          f"stint {residual['stint_sd_pct']:.3f}%, "
          f"lap {residual['lap_sd_pct']:.3f}%")
    print(f"Saved model: {MODEL_PATH}")

    # -----------------------------------------------------------------
    # Figures
    # -----------------------------------------------------------------
    plot_validation_comparison(validation_results)
    plot_predicted_vs_actual(test, test_prediction)
    plot_degradation_curves(final_model, history)
    plot_residuals(test, test_prediction)

    print(f"Saved figures: {OUT_DIR}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
