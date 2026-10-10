"""
09_interactive_simulator.py

Phase 9 — Interactive F1 aerodynamic surrogate simulator.

Purpose
-------
Provide a lightweight Streamlit interface for the completed modelling pipeline:

    six continuous operating variables
        -> reference aerodynamic response OR ML surrogate
        -> Cd / Cl / CoP / energy drain
        -> aerodynamic forces
        -> longitudinal coast-down dynamics
        -> vehicle trajectory
        -> reference vs ML comparison

The app deliberately does NOT claim to be a complete F1 race-car simulator.
The source dataset is synthetic and static, and it does not contain throttle,
engine torque, tyre, steering, track, or measured vehicle-trajectory data.

How to run
----------
From the project root:

    streamlit run 09_interactive_simulator.py

Required project files
----------------------
models/
    reference_aero_model.pkl
    cd_model.pkl
    cl_model.pkl
    cop_shift_model.pkl
    energy_drain_kw_model.pkl

data/
    F1_2026_Synthetic_Dataset_Mode1.csv

The app loads the saved models once using Streamlit's resource cache so that
interactive reruns do not repeatedly pay the large model-loading cost observed
in Phase 8.

No new model is trained in this phase.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    import streamlit as st
except ImportError:
    print(
        "Streamlit is not installed. Install it with:\n"
        "  pip install streamlit\n"
        "Then run:\n"
        "  streamlit run 09_interactive_simulator.py"
    )
    sys.exit(1)

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
except ImportError:
    st.error("Plotly is required. Install it with: pip install plotly")
    st.stop()


# =====================================================================
# Project paths and constants
# =====================================================================

PROJECT_ROOT = Path(__file__).resolve().parent

# Vehicle parameters, equations, model loading and the simulation engine all
# come from the Phase-8 module (loaded below), so the app cannot drift away
# from the scripted simulator.
DEFAULT_DURATION_S = 15.0

DEFAULTS = {
    "velocity_ms": 80.0,
    "flap_front_angle": 15.0,
    "flap_rear_angle": 5.0,
    "ride_height_front": 30.0,
    "ride_height_rear": 52.5,
    "battery_soc": 50.0,
}


# =====================================================================
# Page configuration / styling
# =====================================================================

st.set_page_config(
    page_title="F1 Aerodynamic Surrogate Simulator",
    page_icon="🏎️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .block-container {
        padding-top: 1.2rem;
        padding-bottom: 2rem;
    }
    .metric-card {
        border: 1px solid rgba(128,128,128,0.22);
        border-radius: 12px;
        padding: 12px 14px;
        background: rgba(128,128,128,0.05);
    }
    .small-note {
        color: rgba(120,120,120,0.95);
        font-size: 0.86rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# =====================================================================
# Phase-8 simulation engine
# =====================================================================

@st.cache_resource(show_spinner=False)
def load_core_module():
    """Load Phase-8 definitions without executing its main pipeline."""
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

    try:
        return importlib.import_module("08_vehicle_longitudinal_simulator")
    except Exception as exc:
        raise RuntimeError(
            "Could not import 08_vehicle_longitudinal_simulator.py. "
            "Keep the Phase-8 script in the same project directory as this app."
        ) from exc


try:
    core = load_core_module()
except Exception as exc:
    st.error(str(exc))
    st.stop()

INPUT_COLUMNS = core.INPUT_COLUMNS
TARGET_COLUMNS = core.TARGET_COLUMNS


# =====================================================================
# Cached model loading
# =====================================================================

@st.cache_resource(show_spinner="Loading aerodynamic models…")
def load_models() -> tuple[Any, dict[str, Any]]:
    """
    Load Phase-7 reference model and Phase-5 surrogate models exactly once.

    Returns
    -------
    reference_model, surrogate_models
    """
    reference_model = core.load_reference_model()
    surrogate_models = core.load_surrogate_models()
    return reference_model, surrogate_models


# =====================================================================
# Utility functions
# =====================================================================

def clamp_default(value: float, low: float, high: float) -> float:
    """Keep a default slider value inside an observed range."""
    return float(np.clip(value, low, high))


def format_value(value: float, digits: int = 3) -> str:
    """Human-readable numerical formatting."""
    return f"{value:,.{digits}f}"


def make_query(state: dict[str, float]) -> pd.DataFrame:
    """Create a one-row model query in the correct feature order."""
    return pd.DataFrame(
        [{column: float(state[column]) for column in INPUT_COLUMNS}]
    )


def inside_observed_envelope(
    reference_model: Any,
    state: dict[str, float],
) -> tuple[bool, list[str]]:
    """Check whether a simulator state is inside the Phase-7 input domain."""
    query = make_query(state)
    bounds = reference_model.check_bounds(query).iloc[0]
    outside = []
    if not bool(bounds["inside_observed_envelope"]):
        raw = str(bounds["outside_columns"])
        if raw:
            outside = [part.strip() for part in raw.split(",")]
    return bool(bounds["inside_observed_envelope"]), outside


def build_svg_car(
    front_flap_deg: float,
    rear_flap_deg: float,
    active_pulse: bool,
) -> str:
    """Return a self-contained vector-style F1 top-view schematic."""
    rear_label = f"Rear flap: {rear_flap_deg:.1f}°"
    front_label = f"Front flap: {front_flap_deg:.1f}°"
    pulse_label = "ACTIVE-AERO PULSE ON" if active_pulse else "FIXED AERO STATE"

    return f"""
    <svg viewBox="0 0 900 280" width="100%" role="img" aria-label="F1 car schematic">
      <rect x="0" y="0" width="900" height="280" rx="20" fill="none"/>
      <line x1="40" y1="140" x2="860" y2="140" stroke="currentColor" stroke-opacity="0.15" stroke-width="2" stroke-dasharray="7 7"/>

      <!-- wheels -->
      <rect x="210" y="57" width="88" height="44" rx="13" fill="currentColor" fill-opacity="0.22" transform="rotate(-8 254 79)"/>
      <rect x="210" y="179" width="88" height="44" rx="13" fill="currentColor" fill-opacity="0.22" transform="rotate(8 254 201)"/>
      <rect x="610" y="57" width="88" height="44" rx="13" fill="currentColor" fill-opacity="0.22" transform="rotate(8 654 79)"/>
      <rect x="610" y="179" width="88" height="44" rx="13" fill="currentColor" fill-opacity="0.22" transform="rotate(-8 654 201)"/>

      <!-- front wing -->
      <path d="M125 110 Q155 140 125 170" fill="none" stroke="currentColor" stroke-width="11" stroke-linecap="round"/>
      <line x1="92" y1="111" x2="158" y2="111" stroke="currentColor" stroke-width="7" stroke-linecap="round"/>
      <line x1="92" y1="169" x2="158" y2="169" stroke="currentColor" stroke-width="7" stroke-linecap="round"/>

      <!-- central body -->
      <path d="M158 140 C208 108, 270 96, 378 105 L494 87 C566 76, 624 101, 680 140 C624 179, 566 204, 494 193 L378 175 C270 184, 208 172, 158 140 Z"
            fill="currentColor" fill-opacity="0.12" stroke="currentColor" stroke-width="3"/>
      <path d="M300 126 Q392 91 510 116 L578 140 L510 164 Q392 189 300 154 Z"
            fill="currentColor" fill-opacity="0.10" stroke="currentColor" stroke-width="2"/>

      <!-- cockpit -->
      <ellipse cx="395" cy="140" rx="48" ry="28" fill="currentColor" fill-opacity="0.18" stroke="currentColor" stroke-width="2"/>
      <circle cx="395" cy="136" r="10" fill="currentColor" fill-opacity="0.22"/>

      <!-- rear wing -->
      <line x1="713" y1="99" x2="713" y2="181" stroke="currentColor" stroke-width="9" stroke-linecap="round"/>
      <line x1="695" y1="99" x2="755" y2="99" stroke="currentColor" stroke-width="7" stroke-linecap="round"/>
      <line x1="695" y1="181" x2="755" y2="181" stroke="currentColor" stroke-width="7" stroke-linecap="round"/>

      <!-- arrows -->
      <line x1="54" y1="140" x2="116" y2="140" stroke="currentColor" stroke-width="3"/>
      <polygon points="116,140 104,133 104,147" fill="currentColor"/>
      <text x="45" y="118" font-size="17" fill="currentColor">Vehicle motion</text>

      <line x1="800" y1="82" x2="800" y2="122" stroke="currentColor" stroke-width="3"/>
      <polygon points="800,122 793,110 807,110" fill="currentColor"/>
      <text x="732" y="58" font-size="17" fill="currentColor">Downforce</text>

      <text x="38" y="250" font-size="17" fill="currentColor">{front_label}</text>
      <text x="690" y="250" font-size="17" fill="currentColor">{rear_label}</text>
      <text x="343" y="251" font-size="16" fill="currentColor" opacity="0.72">{pulse_label}</text>
    </svg>
    """


def build_control_schedule(
    reference_model: Any,
    initial_state: dict[str, float],
    duration_s: float,
    active_pulse: bool,
    high_flap_delta_deg: float = 15.0,
) -> pd.DataFrame:
    """
    Build the control schedule for one interactive run.

    The schedule has the same columns as the Phase-8 schedule, so the
    Phase-8 simulation engine can run it unchanged.
    """
    n_steps = int(round(duration_s / core.TIME_STEP_S)) + 1
    times = np.arange(n_steps, dtype=float) * core.TIME_STEP_S

    low_rear = float(initial_state["flap_rear_angle"])
    observed_max_rear = float(
        reference_model.feature_max[INPUT_COLUMNS.index("flap_rear_angle")]
    )

    high_rear = min(
        low_rear + high_flap_delta_deg,
        observed_max_rear,
    )

    if active_pulse:
        rear_flap = np.where(
            (times >= 5.0) & (times < 10.0),
            high_rear,
            low_rear,
        )
    else:
        rear_flap = np.full(n_steps, low_rear)

    schedule = pd.DataFrame(
        {
            "time_s": times,
            "flap_rear_angle": rear_flap,
        }
    )

    for column in (
        "flap_front_angle",
        "ride_height_front",
        "ride_height_rear",
        "battery_soc",
    ):
        schedule[column] = float(initial_state[column])

    return schedule


def initial_dashboard_rows(
    reference_model: Any,
    surrogate_models: dict[str, Any],
    state: dict[str, float],
) -> tuple[dict[str, float], dict[str, float]]:
    """Get reference and ML values for the selected operating point."""
    reference = core.reference_predict(reference_model, state)
    surrogate = core.surrogate_predict(surrogate_models, state)

    for values in (reference, surrogate):
        drag, downforce = core.aerodynamic_forces(
            cd=values["cd_total"],
            cl=values["cl_total"],
            velocity_ms=state["velocity_ms"],
        )
        values["drag_force_N"] = drag
        values["downforce_N"] = downforce

    return reference, surrogate


# =====================================================================
# Load resources and establish slider domains
# =====================================================================

try:
    reference_model, surrogate_models = load_models()
except Exception as exc:
    st.error(str(exc))
    st.info(
        "Make sure Phase-8 and all saved model files are present, then start "
        "the app from the project root."
    )
    st.stop()

feature_min = dict(zip(INPUT_COLUMNS, reference_model.feature_min))
feature_max = dict(zip(INPUT_COLUMNS, reference_model.feature_max))


# =====================================================================
# Header
# =====================================================================

st.title("🏎️ F1 Aerodynamic Performance Simulator")
st.caption(
    "Data-driven surrogate simulator • Mode-1 synthetic dataset • "
    "Reference interpolation vs ML surrogate"
)

st.markdown(
    """
    **What this simulator does:** the six continuous operating variables are fed
    into two aerodynamic response paths. The **Reference** path interpolates the
    source dataset, while the **ML Surrogate** uses the saved regression models.
    Both then use the same aerodynamic equations and longitudinal coast-down
    dynamics.
    """
)

with st.expander("Modeling scope and scientific limitation", expanded=False):
    st.write(
        "This is a controlled surrogate experiment, not a validated real F1 "
        "race-car simulation. The source dataset is synthetic and static. "
        "The reference path is data-grounded interpolation rather than real "
        "telemetry or recovered hidden physics."
    )


# =====================================================================
# Sidebar — six continuous controls
# =====================================================================

st.sidebar.header("⚙️ Operating state")
st.sidebar.caption(
    "All six model inputs remain continuous. Slider ranges are taken directly "
    "from the observed reference-model domain."
)


def slider_value(column: str, label: str, step: float, fmt: str) -> float:
    low = float(feature_min[column])
    high = float(feature_max[column])
    default = clamp_default(float(DEFAULTS[column]), low, high)
    return float(
        st.sidebar.slider(
            label,
            min_value=low,
            max_value=high,
            value=default,
            step=step,
            format=fmt,
            key=f"slider_{column}",
        )
    )


velocity = slider_value("velocity_ms", "Vehicle velocity (m/s)", 0.1, "%.1f")
front_flap = slider_value("flap_front_angle", "Front flap angle (°)", 0.1, "%.1f")
rear_flap = slider_value("flap_rear_angle", "Rear flap angle (°)", 0.1, "%.1f")
ride_front = slider_value("ride_height_front", "Front ride height", 0.1, "%.1f")
ride_rear = slider_value("ride_height_rear", "Rear ride height", 0.1, "%.1f")
battery_soc = slider_value("battery_soc", "Battery SOC (%)", 0.1, "%.1f")

st.sidebar.divider()
st.sidebar.header("🎛️ Simulation")

active_pulse = st.sidebar.checkbox(
    "Enable 5–10 s rear-flap pulse",
    value=True,
    help=(
        "During 5–10 s the rear flap is increased by up to 15°. "
        "The value is clipped at the observed dataset maximum."
    ),
)

duration = st.sidebar.slider(
    "Simulation duration (s)",
    min_value=5.0,
    max_value=15.0,
    value=DEFAULT_DURATION_S,
    step=0.5,
)

run_simulation = st.sidebar.button(
    "▶ Run simulation",
    type="primary",
    width="stretch",
)


# =====================================================================
# Current state and initial aero response
# =====================================================================

current_state = {
    "velocity_ms": velocity,
    "flap_front_angle": front_flap,
    "flap_rear_angle": rear_flap,
    "ride_height_front": ride_front,
    "ride_height_rear": ride_rear,
    "battery_soc": battery_soc,
}

inside, outside = inside_observed_envelope(reference_model, current_state)

if not inside:
    st.error(
        "The selected state is outside the observed dataset envelope. "
        f"Outside variable(s): {', '.join(outside)}"
    )
    st.stop()

try:
    reference_now, surrogate_now = initial_dashboard_rows(
        reference_model,
        surrogate_models,
        current_state,
    )
except Exception as exc:
    st.error(f"Could not evaluate the selected operating point: {exc}")
    st.stop()


# =====================================================================
# Vehicle schematic + live metrics
# =====================================================================

left, right = st.columns([1.2, 1.8], gap="large")

with left:
    st.subheader("Vehicle state")
    st.markdown(
        build_svg_car(front_flap, rear_flap, active_pulse),
        unsafe_allow_html=True,
    )
    st.caption(
        f"Selected state: {velocity:.1f} m/s • "
        f"front flap {front_flap:.1f}° • rear flap {rear_flap:.1f}°"
    )

with right:
    st.subheader("Live aerodynamic response")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Reference Cd", format_value(reference_now["cd_total"], 4))
    m2.metric("ML Cd", format_value(surrogate_now["cd_total"], 4))
    m3.metric("Reference Cl", format_value(reference_now["cl_total"], 4))
    m4.metric("ML Cl", format_value(surrogate_now["cl_total"], 4))

    m5, m6, m7, m8 = st.columns(4)
    m5.metric("Reference drag", f"{format_value(reference_now['drag_force_N'], 0)} N")
    m6.metric("ML drag", f"{format_value(surrogate_now['drag_force_N'], 0)} N")
    m7.metric("Reference downforce", f"{format_value(reference_now['downforce_N'], 0)} N")
    m8.metric("ML downforce", f"{format_value(surrogate_now['downforce_N'], 0)} N")

    response_table = pd.DataFrame(
        {
            "Quantity": [
                "Cd",
                "Cl",
                "CoP shift",
                "Energy drain (kW)",
                "Drag (N)",
                "Downforce (N)",
            ],
            "Reference": [
                reference_now["cd_total"],
                reference_now["cl_total"],
                reference_now["cop_shift"],
                reference_now["energy_drain_kw"],
                reference_now["drag_force_N"],
                reference_now["downforce_N"],
            ],
            "ML surrogate": [
                surrogate_now["cd_total"],
                surrogate_now["cl_total"],
                surrogate_now["cop_shift"],
                surrogate_now["energy_drain_kw"],
                surrogate_now["drag_force_N"],
                surrogate_now["downforce_N"],
            ],
        }
    )

    response_table["ML - Reference"] = (
        response_table["ML surrogate"] - response_table["Reference"]
    )

    st.dataframe(
        response_table.style.format(
            {
                "Reference": "{:.4f}",
                "ML surrogate": "{:.4f}",
                "ML - Reference": "{:+.4f}",
            }
        ),
        width="stretch",
        hide_index=True,
    )


# =====================================================================
# Simulation execution
# =====================================================================

st.divider()
st.subheader("Reference vs ML vehicle simulation")

if run_simulation:
    # Snapshot the controls at the moment the user explicitly starts the run.
    # Later slider movements update the live aero panel but do not silently
    # replace the plotted trajectory.
    st.session_state["simulation_state"] = current_state.copy()
    st.session_state["simulation_duration"] = float(duration)
    st.session_state["simulation_active_pulse"] = bool(active_pulse)

simulation_state = st.session_state.get("simulation_state")
simulation_duration = st.session_state.get("simulation_duration")
simulation_active_pulse = st.session_state.get("simulation_active_pulse")

if simulation_state is None:
    st.info(
        "Adjust the six continuous controls and press **Run simulation** in "
        "the sidebar to generate the trajectory comparison."
    )
else:
    # The expensive model-loading phase is cached. Trajectory data are also
    # stored in session state so moving sliders does not silently rerun the
    # simulation; only the explicit Run button starts a new trajectory.
    if run_simulation:
        with st.spinner("Running reference and ML-surrogate coast-down simulations…"):
            try:
                control_schedule = build_control_schedule(
                    reference_model,
                    simulation_state,
                    simulation_duration,
                    simulation_active_pulse,
                )

                reference_trajectory = core.simulate_path(
                    path_name="reference",
                    reference_model=reference_model,
                    surrogate_models=surrogate_models,
                    use_reference=True,
                    control_schedule=control_schedule,
                    initial_velocity_ms=simulation_state["velocity_ms"],
                )

                surrogate_trajectory = core.simulate_path(
                    path_name="ml_surrogate",
                    reference_model=reference_model,
                    surrogate_models=surrogate_models,
                    use_reference=False,
                    control_schedule=control_schedule,
                    initial_velocity_ms=simulation_state["velocity_ms"],
                )

                comparison = core.compare_trajectories(
                    reference_trajectory,
                    surrogate_trajectory,
                )

                st.session_state["reference_trajectory"] = reference_trajectory
                st.session_state["surrogate_trajectory"] = surrogate_trajectory
                st.session_state["comparison"] = comparison
                st.session_state["stop_reasons"] = [
                    f"{label}: {trajectory.attrs['stop_reason']}"
                    for label, trajectory in (
                        ("Reference", reference_trajectory),
                        ("ML surrogate", surrogate_trajectory),
                    )
                    if trajectory.attrs.get("stop_reason")
                ]
            except Exception as exc:
                st.error(f"Simulation could not be completed: {exc}")
                st.stop()

    reference_trajectory = st.session_state.get("reference_trajectory")
    surrogate_trajectory = st.session_state.get("surrogate_trajectory")
    comparison = st.session_state.get("comparison")

    if reference_trajectory is None or surrogate_trajectory is None or comparison is None:
        st.info("Press **Run simulation** to generate the trajectory comparison.")
        st.stop()

    stop_reasons = st.session_state.get("stop_reasons", [])

    if stop_reasons:
        st.warning(
            "The run ended before the requested duration. "
            + " ".join(f"{reason}." for reason in stop_reasons)
        )

    # -----------------------------------------------------------------
    # Headline trajectory metrics
    # -----------------------------------------------------------------
    speed_rmse = core.rmse(
        comparison["velocity_ms_reference"].to_numpy(),
        comparison["velocity_ms_surrogate"].to_numpy(),
    )
    distance_rmse = core.rmse(
        comparison["distance_m_reference"].to_numpy(),
        comparison["distance_m_surrogate"].to_numpy(),
    )
    cl_rmse = core.rmse(
        comparison["cl_total_reference"].to_numpy(),
        comparison["cl_total_surrogate"].to_numpy(),
    )
    cd_rmse = core.rmse(
        comparison["cd_total_reference"].to_numpy(),
        comparison["cd_total_surrogate"].to_numpy(),
    )
    downforce_rmse = core.rmse(
        comparison["downforce_N_reference"].to_numpy(),
        comparison["downforce_N_surrogate"].to_numpy(),
    )

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Speed RMSE", f"{speed_rmse:.3f} m/s")
    k2.metric("Distance RMSE", f"{distance_rmse:.3f} m")
    k3.metric("Cd RMSE", f"{cd_rmse:.4f}")
    k4.metric("Cl RMSE", f"{cl_rmse:.4f}")
    k5.metric("Downforce RMSE", f"{downforce_rmse:.1f} N")

    # -----------------------------------------------------------------
    # Trajectory plots
    # -----------------------------------------------------------------
    tab1, tab2, tab3, tab4 = st.tabs(
        [
            "🏁 Vehicle trajectory",
            "🌬️ Aerodynamics",
            "📉 Surrogate error",
            "📋 Data",
        ]
    )

    with tab1:
        fig = make_subplots(
            rows=2,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.12,
            subplot_titles=("Vehicle speed", "Distance travelled"),
        )

        fig.add_trace(
            go.Scatter(
                x=comparison["time_s"],
                y=comparison["velocity_ms_reference"],
                name="Reference",
                mode="lines",
                line=dict(width=3),
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=comparison["time_s"],
                y=comparison["velocity_ms_surrogate"],
                name="ML surrogate",
                mode="lines",
                line=dict(width=2, dash="dash"),
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=comparison["time_s"],
                y=comparison["distance_m_reference"],
                name="Reference distance",
                mode="lines",
                line=dict(width=3),
                showlegend=False,
            ),
            row=2,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=comparison["time_s"],
                y=comparison["distance_m_surrogate"],
                name="ML surrogate distance",
                mode="lines",
                line=dict(width=2, dash="dash"),
                showlegend=False,
            ),
            row=2,
            col=1,
        )

        fig.update_yaxes(title_text="Speed (m/s)", row=1, col=1)
        fig.update_yaxes(title_text="Distance (m)", row=2, col=1)
        fig.update_xaxes(title_text="Time (s)", row=2, col=1)
        fig.update_layout(height=620, hovermode="x unified")
        st.plotly_chart(fig, width="stretch")

        final_speed_ref = float(comparison["velocity_ms_reference"].iloc[-1])
        final_speed_ml = float(comparison["velocity_ms_surrogate"].iloc[-1])
        final_speed_diff = final_speed_ml - final_speed_ref

        st.write(
            f"**Final speed:** reference = {final_speed_ref:.3f} m/s, "
            f"ML = {final_speed_ml:.3f} m/s, "
            f"difference = {final_speed_diff:+.3f} m/s"
        )

    with tab2:
        fig = make_subplots(
            rows=2,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.12,
            subplot_titles=("Cd / Cl", "Drag / downforce"),
        )

        fig.add_trace(
            go.Scatter(
                x=comparison["time_s"],
                y=comparison["cd_total_reference"],
                name="Reference Cd",
                mode="lines",
                line=dict(width=2),
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=comparison["time_s"],
                y=comparison["cd_total_surrogate"],
                name="ML Cd",
                mode="lines",
                line=dict(width=2, dash="dash"),
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=comparison["time_s"],
                y=comparison["cl_total_reference"],
                name="Reference Cl",
                mode="lines",
                line=dict(width=2),
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=comparison["time_s"],
                y=comparison["cl_total_surrogate"],
                name="ML Cl",
                mode="lines",
                line=dict(width=2, dash="dash"),
            ),
            row=1,
            col=1,
        )

        fig.add_trace(
            go.Scatter(
                x=comparison["time_s"],
                y=comparison["drag_force_N_reference"],
                name="Reference drag",
                mode="lines",
                line=dict(width=2),
            ),
            row=2,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=comparison["time_s"],
                y=comparison["drag_force_N_surrogate"],
                name="ML drag",
                mode="lines",
                line=dict(width=2, dash="dash"),
            ),
            row=2,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=comparison["time_s"],
                y=comparison["downforce_N_reference"],
                name="Reference downforce",
                mode="lines",
                line=dict(width=2),
            ),
            row=2,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=comparison["time_s"],
                y=comparison["downforce_N_surrogate"],
                name="ML downforce",
                mode="lines",
                line=dict(width=2, dash="dash"),
            ),
            row=2,
            col=1,
        )

        fig.update_yaxes(title_text="Coefficient", row=1, col=1)
        fig.update_yaxes(title_text="Force (N)", row=2, col=1)
        fig.update_xaxes(title_text="Time (s)", row=2, col=1)
        fig.update_layout(height=680, hovermode="x unified")
        st.plotly_chart(fig, width="stretch")

        st.markdown("**Cl–Cd trade-off during the simulated run**")
        fig_trade = go.Figure()
        fig_trade.add_trace(
            go.Scatter(
                x=comparison["cd_total_reference"],
                y=comparison["cl_total_reference"],
                mode="lines+markers",
                marker=dict(size=4),
                name="Reference",
            )
        )
        fig_trade.add_trace(
            go.Scatter(
                x=comparison["cd_total_surrogate"],
                y=comparison["cl_total_surrogate"],
                mode="lines+markers",
                marker=dict(size=4),
                name="ML surrogate",
            )
        )
        fig_trade.update_layout(
            height=430,
            xaxis_title="Cd",
            yaxis_title="Cl",
            hovermode="closest",
        )
        st.plotly_chart(fig_trade, width="stretch")

    with tab3:
        error_fig = go.Figure()
        error_fig.add_trace(
            go.Scatter(
                x=comparison["time_s"],
                y=comparison["velocity_ms_error"],
                name="Speed error",
                mode="lines",
            )
        )
        error_fig.add_hline(y=0, line_dash="dot")
        error_fig.update_layout(
            height=400,
            xaxis_title="Time (s)",
            yaxis_title="ML − Reference speed (m/s)",
            hovermode="x unified",
        )
        st.plotly_chart(error_fig, width="stretch")

        error_table = pd.DataFrame(
            {
                "Quantity": [
                    "Speed",
                    "Distance",
                    "Acceleration",
                    "Cl",
                    "Cd",
                    "CoP shift",
                    "Energy drain",
                    "Drag force",
                    "Downforce",
                ],
                "RMSE": [
                    core.rmse(comparison["velocity_ms_reference"].to_numpy(), comparison["velocity_ms_surrogate"].to_numpy()),
                    core.rmse(comparison["distance_m_reference"].to_numpy(), comparison["distance_m_surrogate"].to_numpy()),
                    core.rmse(comparison["acceleration_m_s2_reference"].to_numpy(), comparison["acceleration_m_s2_surrogate"].to_numpy()),
                    core.rmse(comparison["cl_total_reference"].to_numpy(), comparison["cl_total_surrogate"].to_numpy()),
                    core.rmse(comparison["cd_total_reference"].to_numpy(), comparison["cd_total_surrogate"].to_numpy()),
                    core.rmse(comparison["cop_shift_reference"].to_numpy(), comparison["cop_shift_surrogate"].to_numpy()),
                    core.rmse(comparison["energy_drain_kw_reference"].to_numpy(), comparison["energy_drain_kw_surrogate"].to_numpy()),
                    core.rmse(comparison["drag_force_N_reference"].to_numpy(), comparison["drag_force_N_surrogate"].to_numpy()),
                    core.rmse(comparison["downforce_N_reference"].to_numpy(), comparison["downforce_N_surrogate"].to_numpy()),
                ],
            }
        )
        st.dataframe(
            error_table.style.format({"RMSE": "{:.6f}"}),
            width="stretch",
            hide_index=True,
        )

    with tab4:
        st.write("Reference trajectory")
        st.dataframe(reference_trajectory, width="stretch", hide_index=True)
        st.write("ML-surrogate trajectory")
        st.dataframe(surrogate_trajectory, width="stretch", hide_index=True)

        st.download_button(
            "Download comparison CSV",
            comparison.to_csv(index=False).encode("utf-8"),
            file_name="interactive_reference_vs_ml_comparison.csv",
            mime="text/csv",
            width="content",
        )


# =====================================================================
# Footer
# =====================================================================

st.divider()
col_a, col_b = st.columns(2)
with col_a:
    st.markdown(
        "<div class='small-note'>"
        "Reference = local interpolation through the Mode-1 synthetic dataset."
        "</div>",
        unsafe_allow_html=True,
    )
with col_b:
    st.markdown(
        "<div class='small-note'>"
        "ML surrogate = saved Phase-5 continuous regression models."
        "</div>",
        unsafe_allow_html=True,
    )
