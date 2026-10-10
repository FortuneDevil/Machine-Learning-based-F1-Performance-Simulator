"""
05_app.py

Interactive race strategy simulator.

Pick a real race and driver. The app takes the driver's pace from their
first clean laps, predicts the rest of the race with the trained lap-time
model, searches every pit strategy, and compares the result with what the
team really did.

Run
---
streamlit run 05_app.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

from common import DRY_COMPOUNDS, LAPS_PATH, MODEL_LAPS_PATH, TEST_YEARS
from plot_style import (
    ACTUAL_COLOR,
    COMPOUND_COLORS,
    NEUTRAL_COLOR,
    PREDICTED_COLOR,
    plotly_layout,
)
from race_animation import (
    race_payload,
    render_race_html,
    results_table,
    simulate_cars,
)

# Car colours and short labels in the animated race. These are the dark-
# surface steps of the same three series colours used in the charts.
RACE_CAR_COLORS = {
    "Recommended": "#3987e5",
    "Team": "#d95926",
    "Yours": "#199e70",
}
RACE_CAR_LABELS = {
    "Recommended": "REC",
    "Team": "TEAM",
    "Yours": "YOU",
}
from strategy import (
    MIN_STINT_LAPS,
    actual_stops,
    best_strategies,
    context_from_laps,
    describe_strategy,
    is_valid_strategy,
    load_bundle,
    predict_laps,
    simulate_race_times,
    strategy_time,
    wear_curve,
)

st.set_page_config(
    page_title="F1 Race Strategy Simulator",
    page_icon="🏁",
    layout="wide",
)


# =====================================================================
# Cached loading
# =====================================================================


@st.cache_resource(show_spinner="Loading lap-time model…")
def get_bundle() -> dict:
    return load_bundle()


@st.cache_data(show_spinner="Loading race data…")
def get_laps() -> tuple[pd.DataFrame, pd.DataFrame]:
    model_laps = pd.read_csv(MODEL_LAPS_PATH)
    raw_laps = pd.read_csv(LAPS_PATH)
    raw_laps["Compound"] = raw_laps["Compound"].astype(str).str.upper()
    return model_laps, raw_laps


@st.cache_data(show_spinner="Searching pit strategies…")
def get_ranking(
    year: int,
    round_number: int,
    driver: str,
    max_stops: int,
    pit_loss_s: float,
) -> pd.DataFrame:
    context, race_compounds, _ = get_race(year, round_number, driver)
    context.pit_loss_s = pit_loss_s
    return best_strategies(get_bundle(), context, race_compounds, max_stops)


def get_race(year: int, round_number: int, driver: str):
    """Context, compounds already used, and the team's real stops."""
    model_laps, raw_laps = get_laps()

    def pick(df: pd.DataFrame) -> pd.DataFrame:
        return df[
            (df["Year"] == year)
            & (df["Round"] == round_number)
            & (df["Driver"] == driver)
        ].sort_values("LapNumber")

    driver_model = pick(model_laps)
    driver_raw = pick(raw_laps)

    context = context_from_laps(get_bundle(), driver_model, driver_raw)

    before = driver_raw[driver_raw["LapNumber"] < context.start_lap]
    race_compounds = sorted(
        set(before["Compound"]) & set(DRY_COMPOUNDS)
        | {context.start_compound}
    )

    return context, race_compounds, actual_stops(driver_raw, context)


def format_duration(seconds: float) -> str:
    minutes, rest = divmod(float(seconds), 60.0)
    hours, minutes = divmod(int(minutes), 60)
    return f"{hours}:{minutes:02d}:{rest:04.1f}"


# =====================================================================
# Load
# =====================================================================

try:
    bundle = get_bundle()
    model_laps, raw_laps = get_laps()
except FileNotFoundError as exc:
    st.error(str(exc))
    st.info("Run scripts 01 to 03 first, then start the app again.")
    st.stop()

st.title("🏁 F1 Race Strategy Simulator")
st.caption(
    "Real race data • lap-time model trained on 2022–2024 • "
    "2025 is the untouched test season"
)

# =====================================================================
# Sidebar: pick a race and driver
# =====================================================================

st.sidebar.header("Race")

years = sorted(model_laps["Year"].unique(), reverse=True)
year = st.sidebar.selectbox(
    "Season",
    years,
    format_func=lambda y: f"{y} (test season)" if y in TEST_YEARS
    else f"{y} (seen in training)",
)

season = model_laps[model_laps["Year"] == year]
events = (
    season[["Round", "EventName"]].drop_duplicates().sort_values("Round")
)
round_number = st.sidebar.selectbox(
    "Grand Prix",
    events["Round"].tolist(),
    format_func=lambda r: events.set_index("Round").loc[r, "EventName"],
)

race = season[season["Round"] == round_number]
drivers = (
    race[["Driver", "FinishPosition"]]
    .drop_duplicates()
    .sort_values("FinishPosition")
)
driver = st.sidebar.selectbox(
    "Driver",
    drivers["Driver"].tolist(),
    format_func=lambda d: (
        f"{d} (finished P"
        f"{int(drivers.set_index('Driver').loc[d, 'FinishPosition'])})"
        if pd.notna(drivers.set_index("Driver").loc[d, "FinishPosition"])
        else d
    ),
)

context, race_compounds, team_stops = get_race(year, round_number, driver)

st.sidebar.header("Simulation")

max_stops = st.sidebar.radio(
    "Strategies to search",
    [1, 2],
    index=1,
    format_func=lambda n: "One stop only" if n == 1 else "One and two stops",
)

pit_loss = st.sidebar.number_input(
    "Time lost per pit stop (s)",
    min_value=10.0,
    max_value=45.0,
    value=float(round(context.pit_loss_s, 1)),
    step=0.5,
    help="Default is the median green-flag pit loss at this circuit in "
         "2022–2024.",
)
context.pit_loss_s = float(pit_loss)

ranking = get_ranking(year, round_number, driver, max_stops, float(pit_loss))

if ranking.empty:
    st.error("No valid strategy was found for this driver and race.")
    st.stop()

best = ranking.iloc[0]
team_valid = all(compound in DRY_COMPOUNDS for _, compound in team_stops)
team_time = strategy_time(bundle, context, team_stops) if team_valid else np.nan

# =====================================================================
# Headline
# =====================================================================

event_name = events.set_index("Round").loc[round_number, "EventName"]
st.subheader(f"{driver}, {year} {event_name}")

st.markdown(
    f"Pace is taken from the first clean laps (up to lap "
    f"**{context.start_lap - 1}**). Everything from lap "
    f"**{context.start_lap}** to lap **{context.total_laps}** is predicted."
)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Baseline pace", f"{context.baseline_time:.3f} s")
c2.metric("Recommended stops", f"{int(best['n_stops'])}")
c3.metric("Predicted race time", format_duration(best["time_s"]))
c4.metric(
    "Predicted gain over team's strategy",
    f"{team_time - best['time_s']:.1f} s" if team_valid else "n/a",
)

st.markdown(
    f"**Recommended:** {best['label']}  \n"
    f"**Team really ran:** {describe_strategy(context, team_stops)}"
)

if team_valid and team_time < best["time_s"]:
    st.caption(
        "The gain is negative because the team's real strategy lies outside "
        "the search: it ran a tyre longer than the simulator is allowed to "
        "plan, or used a pit lap between the two-stop grid points."
    )

tab_animation, tab_race, tab_search, tab_risk, tab_wear, tab_custom = st.tabs(
    [
        "Animated race",
        "Race prediction",
        "Strategy search",
        "Uncertainty",
        "Tyre wear",
        "Try your own",
    ]
)

# =====================================================================
# Tab: race prediction vs reality
# =====================================================================

with tab_race:
    st.markdown(
        "Predicted lap times for the strategy the team really used, against "
        "the laps the driver really drove. Only clean green-flag laps are "
        "shown; pit laps and caution laps are left out."
    )

    if not team_valid:
        st.warning(
            "The team used wet-weather tyres in this race, which the "
            "simulator does not model."
        )
    else:
        predicted = predict_laps(bundle, context, team_stops)
        actual = race[race["Driver"] == driver].sort_values("LapNumber")

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=actual["LapNumber"],
                y=actual["LapTimeS"],
                mode="markers",
                name="Actual clean laps",
                marker=dict(size=8, color=ACTUAL_COLOR,
                            line=dict(width=1, color="white")),
                hovertemplate="%{y:.3f} s<extra>Actual</extra>",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=predicted["LapNumber"],
                y=predicted["pred_lap_time"],
                mode="lines",
                name="Predicted",
                line=dict(width=2, color=PREDICTED_COLOR),
                hovertemplate="%{y:.3f} s<extra>Predicted</extra>",
            )
        )
        for pit_lap, compound in team_stops:
            fig.add_vline(x=pit_lap + 0.5, line_width=1, line_dash="dot",
                          line_color=NEUTRAL_COLOR)
            fig.add_annotation(x=pit_lap + 0.5, y=1, yref="paper",
                               text=f"pit → {compound.title()}",
                               showarrow=False, yanchor="bottom",
                               font=dict(size=11))

        fig.update_layout(**plotly_layout(height=420))
        fig.update_xaxes(title_text="Lap")
        fig.update_yaxes(title_text="Lap time (s)")
        st.plotly_chart(fig, width="stretch")

        joined = actual.merge(
            predicted[["LapNumber", "pred_lap_time"]], on="LapNumber"
        )
        if not joined.empty:
            error = joined["pred_lap_time"] - joined["LapTimeS"]
            e1, e2, e3 = st.columns(3)
            e1.metric("Clean laps compared", f"{len(joined)}")
            e2.metric("Mean absolute error", f"{error.abs().mean():.3f} s")
            e3.metric("Average bias", f"{error.mean():+.3f} s")

        with st.expander("Table view"):
            st.dataframe(
                joined[["LapNumber", "compound", "tyre_life", "LapTimeS",
                        "pred_lap_time"]].rename(
                    columns={
                        "LapNumber": "Lap",
                        "compound": "Compound",
                        "tyre_life": "Tyre age",
                        "LapTimeS": "Actual (s)",
                        "pred_lap_time": "Predicted (s)",
                    }
                ),
                width="stretch",
                hide_index=True,
            )

# =====================================================================
# Tab: strategy search
# =====================================================================

with tab_search:
    st.markdown(
        "Each line is a one-stop race: the pit lap on the horizontal axis, "
        "the tyre fitted at the stop as the line. Lower is faster. Only "
        "strategies allowed by the rules and within observed tyre life are "
        "drawn."
    )

    one_stop = ranking[ranking["n_stops"] == 1].copy()

    if one_stop.empty:
        st.info("No valid one-stop strategy exists for this race.")
    else:
        one_stop["pit_lap"] = one_stop["stops"].map(lambda s: s[0][0])
        one_stop["new_compound"] = one_stop["stops"].map(lambda s: s[0][1])
        one_stop["gap"] = one_stop["time_s"] - best["time_s"]

        fig = go.Figure()
        for compound in DRY_COMPOUNDS:
            line = one_stop[one_stop["new_compound"] == compound].sort_values(
                "pit_lap"
            )
            if line.empty:
                continue
            fig.add_trace(
                go.Scatter(
                    x=line["pit_lap"],
                    y=line["gap"],
                    mode="lines",
                    name=f"Switch to {compound.title()}",
                    line=dict(width=2, color=COMPOUND_COLORS[compound]),
                    hovertemplate="+%{y:.1f} s<extra>"
                                  f"{compound.title()}</extra>",
                )
            )

        if team_stops:
            fig.add_vline(x=team_stops[0][0], line_width=1, line_dash="dot",
                          line_color=NEUTRAL_COLOR)
            fig.add_annotation(x=team_stops[0][0], y=1, yref="paper",
                               text="team's first stop", showarrow=False,
                               yanchor="bottom", font=dict(size=11))

        fig.update_layout(**plotly_layout(height=420))
        fig.update_xaxes(title_text="Pit lap")
        fig.update_yaxes(title_text="Seconds slower than the best strategy",
                         rangemode="tozero")
        st.plotly_chart(fig, width="stretch")

    st.markdown("**Fastest predicted strategies**")
    top = ranking.head(10)[["label", "n_stops", "time_s", "gap_to_best_s"]]
    st.dataframe(
        top.rename(
            columns={
                "label": "Strategy",
                "n_stops": "Stops",
                "time_s": "Predicted time (s)",
                "gap_to_best_s": "Gap to best (s)",
            }
        ).style.format(
            {"Predicted time (s)": "{:.1f}", "Gap to best (s)": "+{:.1f}"}
        ),
        width="stretch",
        hide_index=True,
    )

# =====================================================================
# Tab: uncertainty
# =====================================================================

with tab_risk:
    st.markdown(
        "The model is not exact, so each strategy has a spread of possible "
        "race times. Every simulated race draws the model's typical error "
        "for each stint and each lap, as measured on the 2024 validation "
        "season."
    )

    # One representative of the best one-stop and best two-stop plans, plus
    # the team's real strategy.
    picks: list[tuple[str, list]] = []
    for n_stops in sorted(ranking["n_stops"].unique()):
        row = ranking[ranking["n_stops"] == n_stops].iloc[0]
        picks.append((f"Best {int(n_stops)}-stop: {row['label']}", row["stops"]))
    if team_valid:
        picks.append(
            (f"Team: {describe_strategy(context, team_stops)}", team_stops)
        )

    simulations = {
        label: simulate_race_times(bundle, context, stops, seed=42 + i)
        for i, (label, stops) in enumerate(picks)
    }
    reference_time = float(best["time_s"])

    fig = go.Figure()
    for label, times in simulations.items():
        fig.add_trace(
            go.Box(
                x=times - reference_time,
                name=label,
                orientation="h",
                boxpoints=False,
                marker_color=ACTUAL_COLOR,
                line=dict(width=2),
                hoverinfo="x",
            )
        )
    fig.update_layout(
        **plotly_layout(height=140 + 90 * len(simulations),
                        showlegend=False, hovermode="closest")
    )
    fig.update_xaxes(
        title_text="Race time relative to the recommended strategy's "
                   "prediction (s)",
        showgrid=True,
    )
    st.plotly_chart(fig, width="stretch")

    summary = pd.DataFrame(
        [
            {
                "Strategy": label,
                "Median (s)": np.median(times) - reference_time,
                "5% (s)": np.quantile(times, 0.05) - reference_time,
                "95% (s)": np.quantile(times, 0.95) - reference_time,
            }
            for label, times in simulations.items()
        ]
    )
    st.dataframe(
        summary.style.format(
            {"Median (s)": "{:+.1f}", "5% (s)": "{:+.1f}", "95% (s)": "{:+.1f}"}
        ),
        width="stretch",
        hide_index=True,
    )
    st.caption(
        "When the spreads overlap heavily, the model cannot really separate "
        "the strategies: the predicted gain is smaller than its own error."
    )

# =====================================================================
# Tab: tyre wear
# =====================================================================

with tab_wear:
    st.markdown(
        f"What the model has learned about tyre wear at **{context.circuit}**"
        ". Fuel load is held at its mid-race value so the lines show wear "
        "alone. Each line stops at the longest stint seen in the data."
    )

    fig = go.Figure()
    for compound in DRY_COMPOUNDS:
        curve = wear_curve(bundle, context, compound)
        fig.add_trace(
            go.Scatter(
                x=curve["tyre_life"],
                y=curve["pred_lap_time"],
                mode="lines",
                name=compound.title(),
                line=dict(width=2, color=COMPOUND_COLORS[compound]),
                hovertemplate="%{y:.3f} s<extra>"
                              f"{compound.title()}</extra>",
            )
        )
    fig.update_layout(**plotly_layout(height=420))
    fig.update_xaxes(title_text="Tyre age (laps)")
    fig.update_yaxes(title_text="Predicted lap time (s)")
    st.plotly_chart(fig, width="stretch")

# =====================================================================
# Tab: custom strategy
# =====================================================================

with tab_custom:
    st.markdown("Build a strategy and see how it compares with the best one.")

    first_lap = context.start_lap + MIN_STINT_LAPS - 1
    last_lap = context.total_laps - MIN_STINT_LAPS

    n_custom = st.radio("Number of stops", [1, 2], horizontal=True)
    custom: list[tuple[int, str]] = []
    columns = st.columns(2)

    # Start the controls on the best strategy with this many stops, so the
    # user explores outward from a valid plan.
    same_stops = ranking[ranking["n_stops"] == n_custom]
    starting_plan = same_stops.iloc[0]["stops"] if not same_stops.empty else []

    # Widget keys include the race, so the defaults reset with the selection.
    key_prefix = f"custom_{year}_{round_number}_{driver}_{n_custom}"

    for i in range(n_custom):
        with columns[i]:
            if i < len(starting_plan):
                default_lap, default_compound = starting_plan[i]
            else:
                default_lap = int(
                    first_lap
                    + (last_lap - first_lap) * (i + 1) / (n_custom + 1)
                )
                default_compound = next(
                    c for c in reversed(DRY_COMPOUNDS)
                    if c != context.start_compound
                )

            lap = st.slider(
                f"Stop {i + 1}: pit at the end of lap",
                first_lap, last_lap, int(default_lap),
                key=f"{key_prefix}_lap_{i}",
            )
            compound = st.selectbox(
                f"Stop {i + 1}: tyre to fit",
                DRY_COMPOUNDS,
                index=DRY_COMPOUNDS.index(default_compound),
                format_func=str.title,
                key=f"{key_prefix}_compound_{i}",
            )
            custom.append((lap, compound))

    custom = sorted(custom)
    custom_usable = len({lap for lap, _ in custom}) == len(custom)

    if not custom_usable:
        st.warning("The two stops are on the same lap.")
    else:
        valid = is_valid_strategy(bundle, context, custom, race_compounds)
        custom_time = strategy_time(bundle, context, custom)

        k1, k2 = st.columns(2)
        k1.metric("Predicted race time", format_duration(custom_time))
        k2.metric("Gap to recommended", f"+{custom_time - best['time_s']:.1f} s")

        if not valid:
            st.warning(
                "This strategy breaks a rule the optimiser enforces: it uses "
                "only one dry compound, has a stint shorter than "
                f"{MIN_STINT_LAPS} laps, or runs a tyre longer than was ever "
                "observed here. The prediction is less reliable."
            )

        custom_laps = predict_laps(bundle, context, custom)
        best_laps = predict_laps(bundle, context, best["stops"])

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=best_laps["LapNumber"], y=best_laps["pred_lap_time"],
                mode="lines", name="Recommended",
                line=dict(width=2, color=ACTUAL_COLOR),
                hovertemplate="%{y:.3f} s<extra>Recommended</extra>",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=custom_laps["LapNumber"], y=custom_laps["pred_lap_time"],
                mode="lines", name="Yours",
                line=dict(width=2, color=PREDICTED_COLOR),
                hovertemplate="%{y:.3f} s<extra>Yours</extra>",
            )
        )
        fig.update_layout(**plotly_layout(height=380))
        fig.update_xaxes(title_text="Lap")
        fig.update_yaxes(title_text="Predicted lap time (s)")
        st.plotly_chart(fig, width="stretch")
        st.caption(
            "Lap times exclude the pit stop itself; each stop adds "
            f"{context.pit_loss_s:.1f} s to the race time."
        )

# =====================================================================
# Tab: animated race
# =====================================================================
# This block runs last because it uses the strategy built in "Try your
# own", but its tab is shown first.

with tab_animation:
    st.markdown(
        "The same driver runs each strategy at once, starting together on "
        f"lap **{context.start_lap}**. Press **Play**. Wheel colour shows "
        "the tyre: red soft, yellow medium, white hard."
    )

    a2, a3 = st.columns([2.2, 1])
    with a2:
        randomise = st.checkbox(
            "Add random race-to-race variation",
            value=False,
            help="Off: every lap is the model's prediction. On: one random "
                 "race is drawn using the model's measured error, so the "
                 "result changes from race to race.",
        )
    with a3:
        if st.button("New random race", disabled=not randomise):
            st.session_state["race_seed"] = (
                st.session_state.get("race_seed", 0) + 1
            )

    race_strategies = {"Recommended": list(best["stops"])}
    if team_valid:
        race_strategies["Team"] = team_stops
    if custom_usable:
        race_strategies["Yours"] = custom

    race_cars = simulate_cars(
        bundle,
        context,
        race_strategies,
        random_seed=(
            st.session_state.get("race_seed", 0) if randomise else None
        ),
    )

    race_html = render_race_html(
        race_payload(
            context,
            race_cars,
            colors=RACE_CAR_COLORS,
            short_names=RACE_CAR_LABELS,
            title=f"{driver} · {year} {event_name}",
        )
    )
    components.html(race_html, height=640, scrolling=False)

    st.dataframe(
        pd.DataFrame(results_table(race_cars)).style.format(
            {"Race time (s)": "{:.1f}", "Gap to winner (s)": "+{:.1f}"}
        ),
        width="stretch",
        hide_index=True,
    )
    if custom_usable and not is_valid_strategy(
        bundle, context, custom, race_compounds
    ):
        st.warning(
            "“Yours” breaks a rule the optimiser enforces (for example it "
            "runs a tyre longer than was ever observed here), so its time "
            "is less reliable and it may appear to beat the recommendation."
        )

    st.caption(
        "“Yours” is the strategy set in the Try your own tab. "
        + " ".join(
            f"{name}: {describe_strategy(context, stops)}."
            for name, stops in race_strategies.items()
        )
    )

# =====================================================================
# Footer: honesty about the model
# =====================================================================

st.divider()

test_results = bundle["test_results"].set_index("model")
model_row = test_results.loc[bundle["model_name"]]
naive_row = test_results.loc["No change from baseline"]

st.markdown(
    f"**Model:** {bundle['model_name']}. On the 2025 test season its "
    f"lap-time error is **{model_row['rmse_s']:.3f} s** RMSE, against "
    f"**{naive_row['rmse_s']:.3f} s** for assuming the pace never changes."
)
st.caption(
    "Not modelled: safety cars, traffic and overtaking, weather changes, "
    "tyre warm-up on the out-lap, and the difference between new and used "
    "tyre sets. The recommended strategy is the fastest in clean air."
)
