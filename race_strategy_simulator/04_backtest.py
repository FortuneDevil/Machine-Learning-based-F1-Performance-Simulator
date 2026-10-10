"""
04_backtest.py

Backtest the strategy simulator on the untouched 2025 test season.

For every points finisher in every dry test race the script:

1. takes the driver's baseline pace from their first clean laps,
2. predicts the rest of their race for the strategy they really ran,
3. searches all candidate strategies for the fastest predicted one,
4. compares prediction and recommendation with what actually happened.

Two questions are answered with real outcomes:

- Race time: given the strategy a driver really used, how close is the
  predicted race time to the real one? (Only for races that stayed under
  green flags, since safety cars are not modelled.)
- Strategy: how close is the recommended first pit lap to the one the team
  really chose?

Outputs
-------
reports/04_backtest/backtest_drivers.csv
reports/04_backtest/backtest_summary.csv
reports/04_backtest/*.png

Run
---
python 04_backtest.py
"""

from __future__ import annotations

import sys
import time

import numpy as np
import pandas as pd

from common import (
    DRY_COMPOUNDS,
    LAPS_PATH,
    MODEL_LAPS_PATH,
    REPORT_DIR,
)
from plot_style import (
    ACTUAL_COLOR,
    NEUTRAL_COLOR,
    PREDICTED_COLOR,
    apply_matplotlib_style,
)
from strategy import (
    actual_stops,
    best_strategies,
    context_from_laps,
    describe_strategy,
    load_bundle,
    predict_laps,
    strategy_time,
)

apply_matplotlib_style()

import matplotlib.pyplot as plt  # noqa: E402

OUT_DIR = REPORT_DIR / "04_backtest"

# Backtest the points finishers: their races are the least disturbed by
# damage, penalties and blue flags.
MAX_FINISH_POSITION = 10

DRIVER_KEYS = ["Year", "Round", "Driver"]


def backtest_driver(
    bundle: dict,
    driver_laps: pd.DataFrame,
    raw_laps: pd.DataFrame,
) -> dict | None:
    """Predict and optimise one driver's race; return one result row."""
    context = context_from_laps(bundle, driver_laps, raw_laps)
    stops = actual_stops(raw_laps, context)

    simulated = raw_laps[raw_laps["LapNumber"] >= context.start_lap]
    before = raw_laps[raw_laps["LapNumber"] < context.start_lap]

    # The simulator does not model wet tyres.
    if not simulated["Compound"].isin(DRY_COMPOUNDS).all():
        return None

    race_compounds = sorted(
        set(before["Compound"]) & set(DRY_COMPOUNDS)
        | {context.start_compound}
    )

    ranking = best_strategies(bundle, context, race_compounds)
    if ranking.empty:
        return None

    best = ranking.iloc[0]
    predicted_actual = strategy_time(bundle, context, stops)

    # A real race time is comparable only if the driver completed every
    # lap and none of the simulated laps ran under a caution.
    finished = len(simulated) == context.n_laps
    all_green = bool((simulated["TrackStatus"].astype(str) == "1").all())
    comparable = finished and all_green and simulated["LapTimeS"].notna().all()

    actual_time = float(simulated["LapTimeS"].sum()) if comparable else np.nan

    first = driver_laps.iloc[0]

    return {
        "Year": int(first["Year"]),
        "Round": int(first["Round"]),
        "EventName": first["EventName"],
        "Driver": first["Driver"],
        "FinishPosition": first["FinishPosition"],
        "start_lap": context.start_lap,
        "total_laps": context.total_laps,
        "actual_strategy": describe_strategy(context, stops),
        "actual_stops": len(stops),
        "actual_first_pit_lap": stops[0][0] if stops else np.nan,
        "recommended_strategy": best["label"],
        "recommended_stops": int(best["n_stops"]),
        "recommended_first_pit_lap": (
            best["stops"][0][0] if best["stops"] else np.nan
        ),
        "predicted_time_actual_s": predicted_actual,
        "predicted_time_recommended_s": float(best["time_s"]),
        "predicted_gain_s": predicted_actual - float(best["time_s"]),
        "green_race": comparable,
        "actual_time_s": actual_time,
        "race_time_error_s": predicted_actual - actual_time,
        # Reference with no model: every lap at the baseline pace.
        "naive_time_error_s": (
            context.baseline_time * context.n_laps
            + len(stops) * context.pit_loss_s
            - actual_time
        ),
    }


def summarise(results: pd.DataFrame) -> pd.DataFrame:
    """Headline backtest numbers."""
    green = results[results["green_race"]]
    stopped = results.dropna(
        subset=["actual_first_pit_lap", "recommended_first_pit_lap"]
    )
    pit_gap = (
        stopped["recommended_first_pit_lap"] - stopped["actual_first_pit_lap"]
    )

    rows = [
        ("Driver races backtested", len(results)),
        ("Races covered", results[["Year", "Round"]].drop_duplicates().shape[0]),
        ("Driver races fully under green flag", len(green)),
        (
            "Race-time MAE on green races (s)",
            green["race_time_error_s"].abs().mean(),
        ),
        (
            "Race-time MAE on green races (% of race time)",
            (green["race_time_error_s"].abs() / green["actual_time_s"]).mean()
            * 100,
        ),
        ("Race-time bias on green races (s)", green["race_time_error_s"].mean()),
        (
            "Race-time MAE with no model, constant baseline pace (s)",
            green["naive_time_error_s"].abs().mean(),
        ),
        (
            "Same number of stops as the team (%)",
            (results["recommended_stops"] == results["actual_stops"]).mean()
            * 100,
        ),
        ("First pit lap MAE vs team (laps)", pit_gap.abs().mean()),
        (
            "First pit lap within 3 laps of team (%)",
            (pit_gap.abs() <= 3).mean() * 100,
        ),
        (
            "Median predicted gain of recommended strategy (s)",
            results["predicted_gain_s"].median(),
        ),
    ]

    return pd.DataFrame(rows, columns=["metric", "value"])


# =====================================================================
# Figures
# =====================================================================


def plot_first_pit_lap(results: pd.DataFrame) -> None:
    data = results.dropna(
        subset=["actual_first_pit_lap", "recommended_first_pit_lap"]
    )
    limit = float(
        max(data["actual_first_pit_lap"].max(),
            data["recommended_first_pit_lap"].max())
    ) + 2

    fig, ax = plt.subplots(figsize=(5.6, 5.6))
    ax.plot([0, limit], [0, limit], color=NEUTRAL_COLOR, linewidth=1.2,
            linestyle="--")
    ax.scatter(data["actual_first_pit_lap"],
               data["recommended_first_pit_lap"],
               s=26, alpha=0.6, color=ACTUAL_COLOR,
               edgecolors="#fcfcfb", linewidths=0.6)

    ax.set_xlim(0, limit)
    ax.set_ylim(0, limit)
    ax.set_xlabel("First pit lap the team chose")
    ax.set_ylabel("First pit lap the simulator recommends")
    ax.set_title("Recommended vs real first pit stop, 2025")

    fig.savefig(OUT_DIR / "first_pit_lap_recommended_vs_actual.png")
    plt.close(fig)


def plot_race_time_error(results: pd.DataFrame) -> None:
    green = results[results["green_race"]]
    if green.empty:
        return

    fig, ax = plt.subplots(figsize=(8, 3.8))
    ax.hist(green["race_time_error_s"], bins=25, color=ACTUAL_COLOR,
            edgecolor="#fcfcfb", linewidth=0.6)
    ax.axvline(0, color=NEUTRAL_COLOR, linewidth=1.2)

    ax.set_xlabel("Predicted minus actual race time (seconds)")
    ax.set_ylabel("Driver races")
    ax.set_title("Race-time prediction error for the strategy really used, "
                 "green-flag races 2025")

    fig.savefig(OUT_DIR / "race_time_error.png")
    plt.close(fig)


def plot_example_trace(
    bundle: dict,
    results: pd.DataFrame,
    model_laps: pd.DataFrame,
    raw_laps: pd.DataFrame,
) -> None:
    """Actual vs predicted lap times for the best-predicted green winner."""
    green = results[results["green_race"]].sort_values(
        ["FinishPosition", "race_time_error_s"],
        key=lambda s: s.abs() if s.name == "race_time_error_s" else s,
    )
    if green.empty:
        return

    pick = green.iloc[0]
    mask = lambda df: (  # noqa: E731
        (df["Year"] == pick["Year"])
        & (df["Round"] == pick["Round"])
        & (df["Driver"] == pick["Driver"])
    )

    driver_model = model_laps[mask(model_laps)]
    driver_raw = raw_laps[mask(raw_laps)]

    context = context_from_laps(bundle, driver_model, driver_raw)
    predicted = predict_laps(bundle, context, actual_stops(driver_raw, context))

    fig, ax = plt.subplots(figsize=(9, 4.4))
    ax.scatter(driver_model["LapNumber"], driver_model["LapTimeS"], s=18,
               color=ACTUAL_COLOR, label="Actual clean laps", zorder=3)
    ax.plot(predicted["LapNumber"], predicted["pred_lap_time"],
            color=PREDICTED_COLOR, label="Predicted")

    ax.set_xlabel("Lap")
    ax.set_ylabel("Lap time (s)")
    ax.set_title(
        f"{pick['Driver']}, {int(pick['Year'])} {pick['EventName']}: "
        "predicted from the first laps only"
    )
    ax.legend(loc="upper right", ncols=2)

    fig.savefig(OUT_DIR / "example_lap_trace.png")
    plt.close(fig)


# =====================================================================
# Main
# =====================================================================


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    try:
        bundle = load_bundle()
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}")
        return 1

    model_laps = pd.read_csv(MODEL_LAPS_PATH)
    raw_laps = pd.read_csv(LAPS_PATH)
    raw_laps["Compound"] = raw_laps["Compound"].astype(str).str.upper()

    test = model_laps[
        (model_laps["Split"] == "test")
        & (model_laps["FinishPosition"] <= MAX_FINISH_POSITION)
    ]

    raw_by_driver = dict(tuple(raw_laps.groupby(DRIVER_KEYS)))

    start = time.perf_counter()
    rows = []

    for key, driver_laps in test.groupby(DRIVER_KEYS):
        row = backtest_driver(
            bundle,
            driver_laps.sort_values("LapNumber"),
            raw_by_driver[key].sort_values("LapNumber"),
        )
        if row is not None:
            rows.append(row)

    results = pd.DataFrame(rows).sort_values(
        ["Year", "Round", "FinishPosition"]
    )
    summary = summarise(results)

    results.to_csv(OUT_DIR / "backtest_drivers.csv", index=False)
    summary.to_csv(OUT_DIR / "backtest_summary.csv", index=False)

    print(f"Backtested {len(results)} driver races in "
          f"{time.perf_counter() - start:.0f} s\n")
    print(summary.round(3).to_string(index=False))

    winners = results[results["FinishPosition"] == 1]
    print("\nRace winners:\n")
    print(
        winners[
            ["EventName", "Driver", "actual_strategy",
             "recommended_strategy", "predicted_gain_s"]
        ].round(1).to_string(index=False)
    )

    plot_first_pit_lap(results)
    plot_race_time_error(results)
    plot_example_trace(bundle, results, model_laps, raw_laps)

    print(f"\nSaved: {OUT_DIR}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
