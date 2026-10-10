"""
08_vehicle_longitudinal_simulator.py

Phase 8 of the F1 2026 Data-Driven Aerodynamic Simulator project.

Purpose
-------
Connect the Phase-7 dataset-grounded reference aerodynamic model and the
Phase-5 machine-learning surrogate to the SAME longitudinal vehicle-dynamics
model.

The simulator is intentionally a COAST-DOWN / ACTIVE-AERO demonstration,
not a claim of a complete F1 race-car simulator. The supplied dataset is a
static synthetic aerodynamic dataset and does not contain throttle, engine
torque, tyre, steering, track, or measured vehicle-trajectory data.

Therefore this phase asks a controlled question:

    Given the same initial vehicle state and the same aero-control schedule,
    how does the ML surrogate reproduce the reference aerodynamic response
    when both are embedded in identical longitudinal dynamics?

Two paths are simulated:

    reference path
        operating state
          -> dataset-grounded interpolation
          -> Cd / Cl / CoP / energy
          -> aerodynamic forces
          -> longitudinal dynamics
          -> next vehicle state

    ML-surrogate path
        operating state
          -> saved Phase-5 regressors
          -> Cd / Cl / CoP / energy
          -> aerodynamic forces
          -> longitudinal dynamics
          -> next vehicle state

The two paths use exactly the same vehicle parameters and equations. The
only intended difference is the aerodynamic response source.

Important scientific limitation
--------------------------------
The reference path is a local interpolation through the complete synthetic
CSV. It is NOT real telemetry and it does not reconstruct the hidden physics
engine equations.

Vehicle-dynamics assumptions
----------------------------
The following values are explicit demonstration assumptions and are NOT
claimed to be supplied by the dataset:

    air density           rho = 1.225 kg/m^3
    frontal area          A   = 1.50 m^2
    vehicle mass          m   = 800 kg
    rolling coefficient   Crr = 0.015
    gravity               g   = 9.81 m/s^2

Longitudinal equations
----------------------
Aerodynamic drag:

    F_D = 0.5 * rho * A * Cd * v^2

Downforce magnitude:

    F_L = 0.5 * rho * A * Cl * v^2

Rolling resistance is increased by downforce:

    F_RR = Crr * (m*g + F_L)

For the default coast-down experiment there is no propulsion or braking:

    a = -(F_D + F_RR) / m

Velocity is advanced with an explicit (forward) Euler step and distance
with the trapezoidal rule over the same step.

If the vehicle slows below the lowest velocity in the dataset, the run stops
cleanly at that point instead of extrapolating outside the observed envelope.

Active-aero scenario
--------------------
To make the experiment visibly dynamic, the rear flap follows a simple
illustrative schedule:

    0-5 s       rear flap = 5 degrees
    5-10 s      rear flap = 20 degrees
    10+ s       rear flap = 5 degrees

This is an experimental control schedule, not a learned controller and not
an official F1 regulation strategy.

Inputs used by both aerodynamic paths
--------------------------------------
    velocity_ms
    flap_front_angle
    flap_rear_angle
    ride_height_front
    ride_height_rear
    battery_soc

Outputs tracked
---------------
    cl_total
    cd_total
    cop_shift
    energy_drain_kw

Outputs saved
-------------
models/
    (no new trained model is created in this phase)

reports/08_vehicle_longitudinal_simulator/
    figures/
        speed_comparison.png
        distance_comparison.png
        drag_comparison.png
        downforce_comparison.png
        coefficients_comparison.png
        speed_error.png
    tables/
        reference_trajectory.csv
        surrogate_trajectory.csv
        trajectory_comparison.csv
        simulation_summary.csv
        model_difference_vs_reference_uncertainty.csv
        control_schedule.csv
    vehicle_simulator_summary.md

Run
---
python 08_vehicle_longitudinal_simulator.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

# Force a headless Matplotlib backend. This avoids Tkinter crashes on systems
# where a GUI backend is unavailable.
import matplotlib
matplotlib.use("Agg")

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from reference_model import BOUNDARY_TOLERANCE, ReferenceAerodynamicModel


# =====================================================================
# Configuration
# =====================================================================

PROJECT_ROOT = Path(__file__).resolve().parent

DATA_FILENAME = "F1_2026_Synthetic_Dataset_Mode1.csv"

DATA_CANDIDATES = [
    PROJECT_ROOT / "data" / DATA_FILENAME,
    PROJECT_ROOT / DATA_FILENAME,
]

MODEL_DIR = PROJECT_ROOT / "models"

REFERENCE_MODEL_PATH = MODEL_DIR / "reference_aero_model.pkl"

SURROGATE_MODEL_FILES = {
    "cl_total": "cl_model.pkl",
    "cd_total": "cd_model.pkl",
    "cop_shift": "cop_shift_model.pkl",
    "energy_drain_kw": "energy_drain_kw_model.pkl",
}

REPORT_DIR = (
    PROJECT_ROOT
    / "reports"
    / "08_vehicle_longitudinal_simulator"
)

FIGURE_DIR = REPORT_DIR / "figures"
TABLE_DIR = REPORT_DIR / "tables"
SUMMARY_PATH = REPORT_DIR / "vehicle_simulator_summary.md"

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

# ---------------------------------------------------------------------
# Demonstration vehicle parameters.
# These are explicit assumptions, not dataset values.
# ---------------------------------------------------------------------

AIR_DENSITY_KG_M3 = 1.225
FRONTAL_AREA_M2 = 1.50
VEHICLE_MASS_KG = 800.0
ROLLING_RESISTANCE_COEFFICIENT = 0.015
GRAVITY_M_S2 = 9.81

# ---------------------------------------------------------------------
# Simulation controls.
# ---------------------------------------------------------------------

TIME_STEP_S = 0.05
SIMULATION_DURATION_S = 15.0

# Initial state is deliberately inside the observed dataset envelope.
INITIAL_VELOCITY_MS = 80.0
INITIAL_FRONT_FLAP_DEG = 15.0
INITIAL_REAR_FLAP_DEG = 5.0
INITIAL_RIDE_HEIGHT_FRONT = 30.0
INITIAL_RIDE_HEIGHT_REAR = 52.5
INITIAL_BATTERY_SOC = 50.0

# Small numerical/termination settings.
MIN_VELOCITY_MS = 0.10

RANDOM_STATE = 42

# Phase-7 leave-one-out table. It gives the reference model's own
# interpolation error, which bounds how finely the two paths can be compared.
REFERENCE_UNCERTAINTY_PATH = (
    PROJECT_ROOT
    / "reports"
    / "07_reference_model"
    / "tables"
    / "reference_leave_one_out_metrics.csv"
)


# =====================================================================
# Console / filesystem utilities
# =====================================================================


def print_section(title: str) -> None:
    """Print a clear section heading."""
    print("\n" + "=" * 88)
    print(title)
    print("=" * 88)


def ensure_output_directories() -> None:
    """Create required report directories."""
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)


def resolve_data_path() -> Path:
    """Locate the cleaned Mode-1 dataset in the usual project locations."""
    for path in DATA_CANDIDATES:
        if path.exists():
            return path

    candidates_text = "\n".join(
        f"  - {path}"
        for path in DATA_CANDIDATES
    )

    raise FileNotFoundError(
        "Could not find the cleaned Mode-1 dataset. Checked:\n"
        + candidates_text
    )


def save_figure(fig: plt.Figure, filename: str) -> None:
    """Save a figure and close it."""
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
    """Save a DataFrame as CSV."""
    path = TABLE_DIR / filename
    table.to_csv(path, index=index)
    print(f"Saved table: {path}")


def dataframe_to_markdown_table(
    table: pd.DataFrame,
    float_digits: int = 6,
) -> str:
    """Create a Markdown table without requiring the tabulate package."""
    work = table.copy()

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


# =====================================================================
# Model loading
# =====================================================================


def load_reference_model() -> ReferenceAerodynamicModel:
    """Load and validate the Phase-7 reference interpolator."""
    if not REFERENCE_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Reference model not found:\n  {REFERENCE_MODEL_PATH}\n\n"
            "Run 07_reference_model.py first."
        )

    bundle = joblib.load(REFERENCE_MODEL_PATH)

    if not isinstance(bundle, dict) or "model" not in bundle:
        raise TypeError(
            "reference_aero_model.pkl does not contain the expected "
            "Phase-7 metadata bundle."
        )

    model = bundle["model"]

    if not hasattr(model, "predict"):
        raise TypeError("Loaded reference model does not expose predict().")

    if list(bundle.get("input_columns", [])) != INPUT_COLUMNS:
        raise ValueError(
            "Reference-model input columns do not match the simulator."
        )

    if list(bundle.get("target_columns", [])) != TARGET_COLUMNS:
        raise ValueError(
            "Reference-model target columns do not match the simulator."
        )

    return model


def load_surrogate_models() -> dict[str, object]:
    """Load the four saved Phase-5 regression model bundles."""
    models: dict[str, object] = {}

    for target, filename in SURROGATE_MODEL_FILES.items():
        path = MODEL_DIR / filename

        if not path.exists():
            raise FileNotFoundError(
                f"Saved surrogate model not found:\n  {path}\n\n"
                "Run 05_regression_models.py first."
            )

        bundle = joblib.load(path)

        if not isinstance(bundle, dict):
            raise TypeError(
                f"{filename} does not contain the expected model bundle."
            )

        for key in [
            "model",
            "target",
            "input_columns",
            "model_name",
        ]:
            if key not in bundle:
                raise KeyError(
                    f"{filename} is missing required metadata key '{key}'."
                )

        if bundle["target"] != target:
            raise ValueError(
                f"Model metadata mismatch in {filename}: "
                f"expected target '{target}', got '{bundle['target']}'."
            )

        if list(bundle["input_columns"]) != INPUT_COLUMNS:
            raise ValueError(
                f"Input-feature mismatch in {filename}."
            )

        model = bundle["model"]

        # The simulator predicts one state per time step. A parallel
        # forest pays thread-pool overhead on every single-row call, so
        # run the saved models serially here.
        if hasattr(model, "n_jobs"):
            model.n_jobs = 1

        models[target] = model

    return models


# =====================================================================
# Aerodynamics and vehicle dynamics
# =====================================================================


def aerodynamic_forces(
    cd: float,
    cl: float,
    velocity_ms: float,
) -> tuple[float, float]:
    """Return drag and downforce magnitudes in Newtons."""
    dynamic_term = (
        0.5
        * AIR_DENSITY_KG_M3
        * FRONTAL_AREA_M2
        * velocity_ms**2
    )

    drag_n = dynamic_term * cd
    downforce_n = dynamic_term * cl

    return float(drag_n), float(downforce_n)


def longitudinal_step(
    velocity_ms: float,
    drag_n: float,
    downforce_n: float,
) -> tuple[float, float, float]:
    """
    Advance one coast-down step.

    Returns
    -------
    acceleration_m_s2:
        Instantaneous longitudinal acceleration (negative during coast-down).
    new_velocity_ms:
        Explicit (forward) Euler updated velocity.
    rolling_resistance_n:
        Longitudinal rolling-resistance force.
    """
    rolling_resistance_n = (
        ROLLING_RESISTANCE_COEFFICIENT
        * (
            VEHICLE_MASS_KG * GRAVITY_M_S2
            + downforce_n
        )
    )

    acceleration_m_s2 = -(
        drag_n + rolling_resistance_n
    ) / VEHICLE_MASS_KG

    new_velocity_ms = max(
        0.0,
        velocity_ms + acceleration_m_s2 * TIME_STEP_S,
    )

    return (
        float(acceleration_m_s2),
        float(new_velocity_ms),
        float(rolling_resistance_n),
    )


def rear_flap_schedule(time_s: float) -> float:
    """
    Illustrative active-aero schedule used to make the simulator dynamic.

    0-5 s   : low rear flap angle
    5-10 s  : high rear flap angle
    10+ s   : low rear flap angle again
    """
    if time_s < 5.0:
        return 5.0

    if time_s < 10.0:
        return 20.0

    return 5.0


def build_control_schedule(
    time_grid: np.ndarray,
) -> pd.DataFrame:
    """Create and save the exact control schedule used by both paths."""
    schedule = pd.DataFrame(
        {
            "time_s": time_grid,
            "flap_front_angle": np.full(
                len(time_grid),
                INITIAL_FRONT_FLAP_DEG,
                dtype=float,
            ),
            "flap_rear_angle": np.array(
                [rear_flap_schedule(t) for t in time_grid],
                dtype=float,
            ),
            "ride_height_front": np.full(
                len(time_grid),
                INITIAL_RIDE_HEIGHT_FRONT,
                dtype=float,
            ),
            "ride_height_rear": np.full(
                len(time_grid),
                INITIAL_RIDE_HEIGHT_REAR,
                dtype=float,
            ),
            "battery_soc": np.full(
                len(time_grid),
                INITIAL_BATTERY_SOC,
                dtype=float,
            ),
        }
    )

    return schedule


# =====================================================================
# Aerodynamic prediction wrappers
# =====================================================================


def reference_predict(
    reference_model: ReferenceAerodynamicModel,
    state: dict[str, float],
) -> dict[str, float]:
    """Predict all four reference outputs for one simulator state."""
    query = pd.DataFrame(
        [{column: state[column] for column in INPUT_COLUMNS}]
    )

    prediction = reference_model.predict_dataframe(
        query,
        reject_outside=True,
    ).iloc[0]

    return {
        target: float(prediction[target])
        for target in TARGET_COLUMNS
    }


def surrogate_predict(
    surrogate_models: dict[str, object],
    state: dict[str, float],
) -> dict[str, float]:
    """Predict all four outputs using the saved Phase-5 regressors."""
    query = pd.DataFrame(
        [{column: state[column] for column in INPUT_COLUMNS}]
    )

    outputs: dict[str, float] = {}

    for target in TARGET_COLUMNS:
        prediction = surrogate_models[target].predict(query)[0]
        outputs[target] = float(prediction)

    return outputs


# =====================================================================
# Simulation engine
# =====================================================================


def validate_simulation_state(
    state: dict[str, float],
    reference_model: ReferenceAerodynamicModel,
) -> None:
    """Raise a clear error before querying the reference model."""
    query = pd.DataFrame(
        [{column: state[column] for column in INPUT_COLUMNS}]
    )

    bounds = reference_model.check_bounds(query).iloc[0]

    if not bool(bounds["inside_observed_envelope"]):
        raise ValueError(
            "Simulator state left the observed operating envelope. "
            f"Outside variables: {bounds['outside_columns']}\n"
            f"State: {state}"
        )


def simulate_path(
    path_name: str,
    reference_model: ReferenceAerodynamicModel,
    surrogate_models: dict[str, object],
    use_reference: bool,
    control_schedule: pd.DataFrame,
    initial_velocity_ms: float = INITIAL_VELOCITY_MS,
) -> pd.DataFrame:
    """
    Run one simulator path using the supplied aerodynamic response source.

    Both paths are fed the same control schedule and use identical vehicle
    equations. Only the Cd/Cl/CoP/Energy response source changes.

    If the run ends before the schedule does, the reason is stored in
    ``result.attrs["stop_reason"]`` (an empty string otherwise).
    """
    rows: list[dict[str, float | str]] = []
    stop_reason = ""

    distance_m = 0.0
    velocity_ms = float(initial_velocity_ms)

    observed_min_velocity = float(
        reference_model.feature_min[
            INPUT_COLUMNS.index("velocity_ms")
        ]
    )

    for step_index, control in control_schedule.iterrows():
        time_s = float(control["time_s"])

        state = {
            "velocity_ms": velocity_ms,
            "flap_front_angle": float(control["flap_front_angle"]),
            "flap_rear_angle": float(control["flap_rear_angle"]),
            "ride_height_front": float(control["ride_height_front"]),
            "ride_height_rear": float(control["ride_height_rear"]),
            "battery_soc": float(control["battery_soc"]),
        }

        validate_simulation_state(
            state,
            reference_model,
        )

        if use_reference:
            aero = reference_predict(
                reference_model,
                state,
            )
        else:
            aero = surrogate_predict(
                surrogate_models,
                state,
            )

        drag_n, downforce_n = aerodynamic_forces(
            cd=aero["cd_total"],
            cl=aero["cl_total"],
            velocity_ms=velocity_ms,
        )

        acceleration, new_velocity, rolling_resistance = longitudinal_step(
            velocity_ms=velocity_ms,
            drag_n=drag_n,
            downforce_n=downforce_n,
        )

        rows.append(
            {
                "path": path_name,
                "step": int(step_index),
                "time_s": time_s,
                "velocity_ms": velocity_ms,
                "distance_m": distance_m,
                "acceleration_m_s2": acceleration,
                "flap_front_angle": state["flap_front_angle"],
                "flap_rear_angle": state["flap_rear_angle"],
                "ride_height_front": state["ride_height_front"],
                "ride_height_rear": state["ride_height_rear"],
                "battery_soc": state["battery_soc"],
                "cl_total": aero["cl_total"],
                "cd_total": aero["cd_total"],
                "cop_shift": aero["cop_shift"],
                "energy_drain_kw": aero["energy_drain_kw"],
                "drag_force_N": drag_n,
                "downforce_N": downforce_n,
                "rolling_resistance_N": rolling_resistance,
            }
        )

        # The row above holds the state AT time_s. Distance covered during
        # this step therefore belongs to the next recorded time.
        distance_m += 0.5 * (
            velocity_ms + new_velocity
        ) * TIME_STEP_S

        velocity_ms = new_velocity

        if velocity_ms < MIN_VELOCITY_MS:
            stop_reason = (
                f"vehicle reached near-zero speed after t={time_s:.2f} s"
            )
            break

        # Both aerodynamic sources are only valid inside the observed
        # envelope. Stop before the next query would leave it.
        if velocity_ms < observed_min_velocity - BOUNDARY_TOLERANCE:
            stop_reason = (
                f"speed fell below the lowest velocity in the dataset "
                f"({observed_min_velocity:.2f} m/s) after t={time_s:.2f} s"
            )
            break

    if stop_reason:
        print(f"{path_name}: {stop_reason}. Ending simulation early.")

    result = pd.DataFrame(rows)

    if result.empty:
        raise RuntimeError(f"Simulation path '{path_name}' produced no rows.")

    result.attrs["stop_reason"] = stop_reason

    return result


def open_loop_surrogate_response(
    surrogate_models: dict[str, object],
    reference: pd.DataFrame,
) -> pd.DataFrame:
    """
    Evaluate the ML surrogate on the reference path's own states.

    In the closed-loop runs the two paths drift to different speeds, so
    their coefficient differences mix model error with state divergence.
    Feeding the surrogate exactly the states the reference visited isolates
    the pure model-versus-model difference.
    """
    states = reference[INPUT_COLUMNS]

    return pd.DataFrame(
        {
            target: np.asarray(
                surrogate_models[target].predict(states),
                dtype=float,
            )
            for target in TARGET_COLUMNS
        },
        index=reference.index,
    )


# =====================================================================
# Comparison metrics
# =====================================================================


def rmse(
    a: np.ndarray,
    b: np.ndarray,
) -> float:
    """Root mean squared error."""
    return float(np.sqrt(np.mean((a - b) ** 2)))


def compare_trajectories(
    reference: pd.DataFrame,
    surrogate: pd.DataFrame,
) -> pd.DataFrame:
    """Align both paths by time and calculate surrogate-reference errors."""
    common = reference[[
        "time_s",
        "velocity_ms",
        "distance_m",
        "acceleration_m_s2",
        "cl_total",
        "cd_total",
        "cop_shift",
        "energy_drain_kw",
        "drag_force_N",
        "downforce_N",
    ]].merge(
        surrogate[[
            "time_s",
            "velocity_ms",
            "distance_m",
            "acceleration_m_s2",
            "cl_total",
            "cd_total",
            "cop_shift",
            "energy_drain_kw",
            "drag_force_N",
            "downforce_N",
        ]],
        on="time_s",
        how="inner",
        suffixes=("_reference", "_surrogate"),
    )

    # One path may stop a step earlier than the other at the edge of the
    # observed envelope; the comparison then covers their common times.
    if common.empty:
        raise ValueError(
            "Reference and surrogate trajectories have no common times."
        )

    error_columns = [
        "velocity_ms",
        "distance_m",
        "acceleration_m_s2",
        "cl_total",
        "cd_total",
        "cop_shift",
        "energy_drain_kw",
        "drag_force_N",
        "downforce_N",
    ]

    for column in error_columns:
        common[f"{column}_error"] = (
            common[f"{column}_surrogate"]
            - common[f"{column}_reference"]
        )

    return common


def build_summary_metrics(
    comparison: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize overall reference-versus-surrogate simulation error."""
    metric_rows: list[dict[str, float | str]] = []

    metrics_to_report = {
        "velocity_ms": "Speed (m/s)",
        "distance_m": "Distance (m)",
        "acceleration_m_s2": "Acceleration (m/s^2)",
        "cl_total": "Lift coefficient Cl",
        "cd_total": "Drag coefficient Cd",
        "cop_shift": "CoP shift",
        "energy_drain_kw": "Energy drain (kW)",
        "drag_force_N": "Drag force (N)",
        "downforce_N": "Downforce (N)",
    }

    for column, label in metrics_to_report.items():
        ref = comparison[f"{column}_reference"].to_numpy(dtype=float)
        pred = comparison[f"{column}_surrogate"].to_numpy(dtype=float)
        error = pred - ref

        metric_rows.append(
            {
                "quantity": label,
                "rmse": rmse(ref, pred),
                "mae": float(np.mean(np.abs(error))),
                "max_absolute_error": float(np.max(np.abs(error))),
                "final_reference": float(ref[-1]),
                "final_surrogate": float(pred[-1]),
                "final_difference": float(error[-1]),
            }
        )

    return pd.DataFrame(metric_rows)


def build_model_difference_table(
    reference: pd.DataFrame,
    open_loop: pd.DataFrame,
    comparison: pd.DataFrame,
) -> pd.DataFrame:
    """
    Put the surrogate-reference differences next to the reference's own error.

    closed_loop_rmse
        Difference along the two separately simulated trajectories.
    open_loop_rmse
        Difference when the surrogate is evaluated on the reference states.
    reference_loo_rmse
        Phase-7 leave-one-out interpolation error of the reference itself.
        Differences at or below this level cannot be attributed to the
        surrogate.
    """
    reference_loo: dict[str, float] = {}

    if REFERENCE_UNCERTAINTY_PATH.exists():
        loo_table = pd.read_csv(REFERENCE_UNCERTAINTY_PATH)
        reference_loo = dict(
            zip(loo_table["target"], loo_table["rmse"])
        )

    rows = []

    for target in TARGET_COLUMNS:
        rows.append(
            {
                "target": target,
                "closed_loop_rmse": rmse(
                    comparison[f"{target}_reference"].to_numpy(dtype=float),
                    comparison[f"{target}_surrogate"].to_numpy(dtype=float),
                ),
                "open_loop_rmse": rmse(
                    reference[target].to_numpy(dtype=float),
                    open_loop[target].to_numpy(dtype=float),
                ),
                "reference_loo_rmse": reference_loo.get(target, np.nan),
            }
        )

    return pd.DataFrame(rows)


# =====================================================================
# Figures
# =====================================================================


def plot_line_comparison(
    comparison: pd.DataFrame,
    column: str,
    ylabel: str,
    filename: str,
    title: str,
) -> None:
    """Plot reference and surrogate trajectories for one quantity."""
    fig, ax = plt.subplots(figsize=(9, 5.5))

    ax.plot(
        comparison["time_s"],
        comparison[f"{column}_reference"],
        linewidth=2.0,
        label="Reference",
    )

    ax.plot(
        comparison["time_s"],
        comparison[f"{column}_surrogate"],
        linewidth=1.8,
        linestyle="--",
        label="ML surrogate",
    )

    ax.set_xlabel("Time (s)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    ax.legend()

    save_figure(fig, filename)


def plot_coefficients_comparison(
    comparison: pd.DataFrame,
) -> None:
    """Plot Cd and Cl comparisons in two separate panels within one figure."""
    fig, axes = plt.subplots(
        2,
        1,
        figsize=(9, 8),
        sharex=True,
    )

    axes[0].plot(
        comparison["time_s"],
        comparison["cd_total_reference"],
        linewidth=2.0,
        label="Reference Cd",
    )
    axes[0].plot(
        comparison["time_s"],
        comparison["cd_total_surrogate"],
        linewidth=1.8,
        linestyle="--",
        label="ML surrogate Cd",
    )
    axes[0].set_ylabel("Cd")
    axes[0].set_title("Drag coefficient")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend()

    axes[1].plot(
        comparison["time_s"],
        comparison["cl_total_reference"],
        linewidth=2.0,
        label="Reference Cl",
    )
    axes[1].plot(
        comparison["time_s"],
        comparison["cl_total_surrogate"],
        linewidth=1.8,
        linestyle="--",
        label="ML surrogate Cl",
    )
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Cl")
    axes[1].set_title("Lift / downforce coefficient")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend()

    fig.suptitle("Aerodynamic coefficient comparison")
    fig.tight_layout()

    save_figure(fig, "coefficients_comparison.png")


def plot_speed_error(comparison: pd.DataFrame) -> None:
    """Plot surrogate speed error relative to the reference."""
    fig, ax = plt.subplots(figsize=(9, 5.5))

    ax.axhline(
        0.0,
        linewidth=1.0,
        linestyle=":",
    )

    ax.plot(
        comparison["time_s"],
        comparison["velocity_ms_error"],
        linewidth=2.0,
    )

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Surrogate - reference speed (m/s)")
    ax.set_title("Trajectory error in vehicle speed")
    ax.grid(True, alpha=0.25)

    save_figure(fig, "speed_error.png")


# =====================================================================
# Markdown report
# =====================================================================


def write_summary_report(
    data_path: Path,
    reference: pd.DataFrame,
    surrogate: pd.DataFrame,
    comparison: pd.DataFrame,
    summary_metrics: pd.DataFrame,
    model_difference: pd.DataFrame,
    build_elapsed_s: float,
    reference_load_s: float,
    surrogate_load_s: float,
) -> None:
    """Write a concise Phase-8 summary report."""
    final_reference_speed = float(reference["velocity_ms"].iloc[-1])
    final_surrogate_speed = float(surrogate["velocity_ms"].iloc[-1])

    final_reference_distance = float(reference["distance_m"].iloc[-1])
    final_surrogate_distance = float(surrogate["distance_m"].iloc[-1])

    speed_rmse = float(
        summary_metrics.loc[
            summary_metrics["quantity"] == "Speed (m/s)",
            "rmse",
        ].iloc[0]
    )

    downforce_rmse = float(
        summary_metrics.loc[
            summary_metrics["quantity"] == "Downforce (N)",
            "rmse",
        ].iloc[0]
    )

    cd_rmse = float(
        summary_metrics.loc[
            summary_metrics["quantity"] == "Drag coefficient Cd",
            "rmse",
        ].iloc[0]
    )

    cl_rmse = float(
        summary_metrics.loc[
            summary_metrics["quantity"] == "Lift coefficient Cl",
            "rmse",
        ].iloc[0]
    )

    content = f"""# F1 2026 Mode-1 Dataset — Vehicle Longitudinal Simulator

## Purpose

Phase 8 connects the Phase-7 data-grounded reference aerodynamic response
and the Phase-5 ML surrogate to the same longitudinal vehicle-dynamics model.
The simulator uses identical controls, equations, and initial conditions for
both paths.

The reference path is a local interpolation through the complete synthetic
dataset. It is not independent real-world telemetry and it does not reconstruct
the hidden physics engine equations.

## Dataset

- Dataset: `{data_path.name}`
- Aero mode: fixed to Mode 1
- Continuous inputs: 6
- Continuous outputs: 4
- No categorical modelling variables are introduced.

## Simulation assumptions

| parameter | value |
| --- | ---: |
| air_density_kg_m3 | {AIR_DENSITY_KG_M3:.3f} |
| frontal_area_m2 | {FRONTAL_AREA_M2:.3f} |
| vehicle_mass_kg | {VEHICLE_MASS_KG:.1f} |
| rolling_resistance_coefficient | {ROLLING_RESISTANCE_COEFFICIENT:.4f} |
| gravity_m_s2 | {GRAVITY_M_S2:.2f} |
| time_step_s | {TIME_STEP_S:.3f} |
| nominal_duration_s | {SIMULATION_DURATION_S:.2f} |
| initial_velocity_ms | {INITIAL_VELOCITY_MS:.2f} |
| initial_front_flap_deg | {INITIAL_FRONT_FLAP_DEG:.2f} |
| initial_rear_flap_deg | {INITIAL_REAR_FLAP_DEG:.2f} |
| initial_front_ride_height | {INITIAL_RIDE_HEIGHT_FRONT:.2f} |
| initial_rear_ride_height | {INITIAL_RIDE_HEIGHT_REAR:.2f} |
| initial_battery_soc | {INITIAL_BATTERY_SOC:.2f} |

These values are explicit demonstration assumptions, not measured F1 vehicle
parameters.

## Active-aero control schedule

The rear flap is deliberately changed during the experiment to make the
simulation dynamic:

- 0–5 s: rear flap = 5 degrees
- 5–10 s: rear flap = 20 degrees
- 10–15 s: rear flap = 5 degrees

Both reference and ML paths receive exactly this same schedule.

## Longitudinal model

Aerodynamic drag:

`F_D = 0.5 * rho * A * Cd * v^2`

Downforce:

`F_L = 0.5 * rho * A * Cl * v^2`

Rolling resistance:

`F_RR = Crr * (m*g + F_L)`

Coast-down acceleration:

`a = -(F_D + F_RR) / m`

Velocity is integrated with an explicit Euler step and distance with the
trapezoidal rule.

## Simulation result

| quantity | reference final | surrogate final | difference |
| --- | ---: | ---: | ---: |
| final speed (m/s) | {final_reference_speed:.6f} | {final_surrogate_speed:.6f} | {final_surrogate_speed - final_reference_speed:.6f} |
| final distance (m) | {final_reference_distance:.6f} | {final_surrogate_distance:.6f} | {final_surrogate_distance - final_reference_distance:.6f} |

### Trajectory-level errors

{dataframe_to_markdown_table(summary_metrics.round(8), float_digits=8)}

## Key aerodynamic comparison

- Cd trajectory RMSE: **{cd_rmse:.8f}**
- Cl trajectory RMSE: **{cl_rmse:.8f}**
- Downforce trajectory RMSE: **{downforce_rmse:.6f} N**
- Vehicle-speed trajectory RMSE: **{speed_rmse:.8f} m/s**

The speed error is the most important end-to-end quantity in this phase:
it measures how much the learned aerodynamic surrogate changes the simulated
vehicle trajectory relative to the data-grounded reference path.

## How much of the difference is the reference's own error?

{dataframe_to_markdown_table(model_difference.round(8), float_digits=8)}

- `closed_loop_rmse`: difference along the two separately simulated paths.
- `open_loop_rmse`: difference when the surrogate is evaluated on exactly the
  states the reference path visited, which removes state divergence.
- `reference_loo_rmse`: the reference interpolator's own leave-one-out error
  from Phase 7.

A surrogate-reference difference at or below `reference_loo_rmse` is within
the reference's own uncertainty and should not be read as surrogate error.

## Computational timing

- Reference model load time: {reference_load_s:.6f} s
- Surrogate model load time: {surrogate_load_s:.6f} s
- Two-path simulation time: {build_elapsed_s:.6f} s
- Simulated reference steps: {len(reference):,}
- Simulated surrogate steps: {len(surrogate):,}

## Interpretation

The central experiment is not “real F1 telemetry versus simulation”. It is:

**data-grounded reference response versus learned ML surrogate response under
the same vehicle-dynamics equations.**

This isolates the modelling approximation introduced by replacing the
reference aerodynamic map with the machine-learned models.

## Important limitation

Because the source dataset is synthetic and static, this simulator is a
controlled surrogate experiment rather than a validated race-car simulation.
The next interface layer can expose the six continuous operating variables,
show live Cd/Cl/drag/downforce values, and allow the user to run alternative
what-if aero schedules.
"""

    SUMMARY_PATH.write_text(
        content,
        encoding="utf-8",
    )

    print(f"Saved summary report: {SUMMARY_PATH}")


# =====================================================================
# Main pipeline
# =====================================================================


def main() -> int:
    """Run Phase 8 end to end."""
    print_section("PHASE 8 — VEHICLE LONGITUDINAL-DYNAMICS SIMULATOR")

    ensure_output_directories()

    try:
        data_path = resolve_data_path()
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}")
        return 1

    print_section("1. LOADING DATASET")

    try:
        df = pd.read_csv(data_path)
    except Exception as exc:
        print(f"ERROR while loading dataset: {exc}")
        return 1

    required_columns = INPUT_COLUMNS + TARGET_COLUMNS
    missing = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing:
        print(
            "ERROR: Dataset is missing required columns:\n"
            + "\n".join(f"  - {column}" for column in missing)
        )
        return 1

    if not np.isfinite(
        df[required_columns].to_numpy(dtype=float)
    ).all():
        print("ERROR: Non-finite values found in required dataset columns.")
        return 1

    print(
        f"Loaded dataset: {data_path}\n"
        f"Rows: {len(df):,}\n"
        f"Continuous inputs: {len(INPUT_COLUMNS)}\n"
        f"Continuous outputs: {len(TARGET_COLUMNS)}"
    )

    # -----------------------------------------------------------------
    # Load Phase-7 reference model.
    # -----------------------------------------------------------------
    print_section("2. LOADING DATA-GROUNDED REFERENCE MODEL")

    try:
        start = time.perf_counter()
        reference_model = load_reference_model()
        reference_load_s = time.perf_counter() - start
    except Exception as exc:
        print(f"ERROR while loading reference model: {exc}")
        return 1

    print(
        "Reference model loaded successfully.\n"
        f"Load time: {reference_load_s:.6f} s"
    )

    # -----------------------------------------------------------------
    # Load Phase-5 surrogate models.
    # -----------------------------------------------------------------
    print_section("3. LOADING ML SURROGATE MODELS")

    try:
        start = time.perf_counter()
        surrogate_models = load_surrogate_models()
        surrogate_load_s = time.perf_counter() - start
    except Exception as exc:
        print(f"ERROR while loading surrogate models: {exc}")
        return 1

    for target, filename in SURROGATE_MODEL_FILES.items():
        print(
            f"Loaded {target}: {filename}"
        )

    print(
        f"Total surrogate load time: {surrogate_load_s:.6f} s"
    )

    # -----------------------------------------------------------------
    # Build the common control schedule.
    # -----------------------------------------------------------------
    print_section("4. BUILDING COMMON ACTIVE-AERO CONTROL SCHEDULE")

    time_grid = np.arange(
        0.0,
        SIMULATION_DURATION_S + 0.5 * TIME_STEP_S,
        TIME_STEP_S,
    )

    control_schedule = build_control_schedule(
        time_grid
    )

    save_table(
        control_schedule.round(8),
        "control_schedule.csv",
    )

    print(
        f"Simulation steps: {len(control_schedule):,}\n"
        f"Time step: {TIME_STEP_S:.3f} s\n"
        f"Nominal duration: {SIMULATION_DURATION_S:.2f} s"
    )

    # -----------------------------------------------------------------
    # Run both paths.
    # -----------------------------------------------------------------
    print_section("5. RUNNING REFERENCE + ML-SURROGATE SIMULATIONS")

    simulation_start = time.perf_counter()

    try:
        reference = simulate_path(
            path_name="reference",
            reference_model=reference_model,
            surrogate_models=surrogate_models,
            use_reference=True,
            control_schedule=control_schedule,
        )

        surrogate = simulate_path(
            path_name="ml_surrogate",
            reference_model=reference_model,
            surrogate_models=surrogate_models,
            use_reference=False,
            control_schedule=control_schedule,
        )
    except Exception as exc:
        print(f"ERROR during simulation: {exc}")
        return 1

    simulation_elapsed_s = time.perf_counter() - simulation_start

    print(
        f"Reference steps: {len(reference):,}\n"
        f"ML-surrogate steps: {len(surrogate):,}\n"
        f"Two-path simulation time: {simulation_elapsed_s:.6f} s"
    )

    # -----------------------------------------------------------------
    # Comparison and outputs.
    # -----------------------------------------------------------------
    print_section("6. COMPARING TRAJECTORIES")

    try:
        comparison = compare_trajectories(
            reference,
            surrogate,
        )

        summary_metrics = build_summary_metrics(
            comparison
        )

        model_difference = build_model_difference_table(
            reference,
            open_loop_surrogate_response(
                surrogate_models,
                reference,
            ),
            comparison,
        )
    except Exception as exc:
        print(f"ERROR while comparing trajectories: {exc}")
        return 1

    save_table(
        reference.round(10),
        "reference_trajectory.csv",
    )

    save_table(
        surrogate.round(10),
        "surrogate_trajectory.csv",
    )

    save_table(
        comparison.round(10),
        "trajectory_comparison.csv",
    )

    save_table(
        summary_metrics.round(10),
        "simulation_summary.csv",
    )

    save_table(
        model_difference.round(10),
        "model_difference_vs_reference_uncertainty.csv",
    )

    print("\nTrajectory error summary:\n")
    print(summary_metrics.round(8).to_string(index=False))

    print("\nSurrogate-reference difference vs reference uncertainty:\n")
    print(model_difference.round(8).to_string(index=False))

    # -----------------------------------------------------------------
    # Figures.
    # -----------------------------------------------------------------
    print_section("7. GENERATING SIMULATION FIGURES")

    plot_line_comparison(
        comparison,
        column="velocity_ms",
        ylabel="Vehicle speed (m/s)",
        filename="speed_comparison.png",
        title="Reference vs ML-surrogate vehicle speed",
    )

    plot_line_comparison(
        comparison,
        column="distance_m",
        ylabel="Distance travelled (m)",
        filename="distance_comparison.png",
        title="Reference vs ML-surrogate distance travelled",
    )

    plot_line_comparison(
        comparison,
        column="drag_force_N",
        ylabel="Drag force (N)",
        filename="drag_comparison.png",
        title="Reference vs ML-surrogate aerodynamic drag",
    )

    plot_line_comparison(
        comparison,
        column="downforce_N",
        ylabel="Downforce (N)",
        filename="downforce_comparison.png",
        title="Reference vs ML-surrogate aerodynamic downforce",
    )

    plot_coefficients_comparison(
        comparison
    )

    plot_speed_error(
        comparison
    )

    # -----------------------------------------------------------------
    # Summary report.
    # -----------------------------------------------------------------
    print_section("8. WRITING PHASE-8 SUMMARY")

    write_summary_report(
        data_path=data_path,
        reference=reference,
        surrogate=surrogate,
        comparison=comparison,
        summary_metrics=summary_metrics,
        model_difference=model_difference,
        build_elapsed_s=simulation_elapsed_s,
        reference_load_s=reference_load_s,
        surrogate_load_s=surrogate_load_s,
    )

    print_section("PHASE 8 COMPLETE")

    print(
        "The simulator now provides two end-to-end paths:\n"
        "  Reference data -> aero coefficients -> forces -> vehicle dynamics\n"
        "  ML surrogate -> aero coefficients -> forces -> vehicle dynamics\n\n"
        "Next phase: connect this simulation engine to the interactive GUI."
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
