"""
02_build_dataset.py

Turn the raw race laps into a modelling table.

Idea
----
Absolute lap time depends mostly on the circuit and the car, which a model
cannot know for a future race. What a strategist needs is how lap time
*changes* during the race. So each driver's early clean laps define a
baseline pace, and every later lap is expressed as a percentage change from
that baseline:

    delta_pct = 100 * (lap_time / baseline_time - 1)

The baseline is known a few laps into the race, so predicting the rest of the
race from it uses no future information.

Outputs
-------
data/model_laps.csv   one row per clean racing lap after the baseline window
data/races.csv        one row per race (split, dry/wet, lap counts)
data/pit_loss.csv     time lost by a pit stop at each circuit

Run
---
python 02_build_dataset.py
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from common import (
    BASELINE_FIRST_LAP,
    BASELINE_LAST_LAP,
    BASELINE_MAX_LAPS,
    BASELINE_MIN_LAPS,
    DRY_COMPOUNDS,
    LAPS_PATH,
    MODEL_LAPS_PATH,
    PIT_LOSS_PATH,
    RACES_PATH,
    TEST_YEARS,
    TRAIN_YEARS,
    VALIDATION_YEARS,
)

# A race counts as wet if more than this share of its laps were run on
# intermediate or wet tyres. Wet races follow different rules and are dropped.
WET_RACE_SHARE = 0.05

# Laps slower than this multiple of the driver's own median clean lap are
# treated as traffic, mistakes or leftover caution laps.
SLOW_LAP_FACTOR = 1.04

# The schedule names some venues differently from season to season. A
# circuit must keep one name so the model can learn its tyre wear.
CIRCUIT_ALIASES = {
    "Miami Gardens": "Miami",
}

RACE_KEYS = ["Year", "Round"]
DRIVER_KEYS = ["Year", "Round", "Driver"]


def assign_split(year: int) -> str:
    if year in TRAIN_YEARS:
        return "train"
    if year in VALIDATION_YEARS:
        return "validation"
    if year in TEST_YEARS:
        return "test"
    return "unused"


def flag_clean_laps(laps: pd.DataFrame) -> pd.Series:
    """Green-flag racing laps on dry tyres with a trustworthy lap time."""
    green = laps["TrackStatus"].astype(str) == "1"

    clean = (
        laps["IsAccurate"].astype(bool)
        & green
        & ~laps["PitIn"].astype(bool)
        & ~laps["PitOut"].astype(bool)
        & laps["LapTimeS"].notna()
        & laps["TyreLife"].notna()
        & laps["Compound"].isin(DRY_COMPOUNDS)
        & ~laps["Rainfall"].astype(bool)
        & (laps["LapNumber"] > 1)
    )

    # Remove slow outliers relative to the driver's own race pace.
    median_pace = (
        laps.loc[clean]
        .groupby(DRIVER_KEYS)["LapTimeS"]
        .median()
        .rename("MedianPace")
    )
    pace = laps.join(median_pace, on=DRIVER_KEYS)["MedianPace"]

    return clean & (laps["LapTimeS"] <= SLOW_LAP_FACTOR * pace)


def build_baselines(laps: pd.DataFrame) -> pd.DataFrame:
    """Baseline pace per driver and race from their first clean laps."""
    window = laps[
        laps["Clean"]
        & (laps["Stint"] == 1)
        & laps["LapNumber"].between(BASELINE_FIRST_LAP, BASELINE_LAST_LAP)
    ].sort_values("LapNumber")

    window = window.groupby(DRIVER_KEYS).head(BASELINE_MAX_LAPS)

    baselines = window.groupby(DRIVER_KEYS).agg(
        baseline_time=("LapTimeS", "median"),
        baseline_laps=("LapTimeS", "size"),
        baseline_lap=("LapNumber", "median"),
        baseline_end_lap=("LapNumber", "max"),
        baseline_compound=("Compound", "first"),
        baseline_tyre_life=("TyreLife", "median"),
        baseline_track_temp=("TrackTemp", "median"),
    )

    return baselines[
        baselines["baseline_laps"] >= BASELINE_MIN_LAPS
    ].reset_index()


def build_pit_loss(laps: pd.DataFrame) -> pd.DataFrame:
    """
    Time lost by one pit stop at each circuit.

    For every green-flag stop: in-lap + out-lap minus two normal laps. Only
    training and validation seasons are used, so the test season stays
    untouched.
    """
    history = laps[laps["Split"].isin(["train", "validation"])]

    pace = (
        history[history["Clean"]]
        .groupby(DRIVER_KEYS)["LapTimeS"]
        .median()
        .rename("MedianPace")
    )

    ordered = history.sort_values(DRIVER_KEYS + ["LapNumber"])
    by_driver = ordered.groupby(DRIVER_KEYS)

    # The lap after each in-lap is the out-lap.
    out_lap_time = by_driver["LapTimeS"].shift(-1)
    out_lap_is_pit_out = by_driver["PitOut"].shift(-1, fill_value=False)
    out_lap_green = by_driver["Green"].shift(-1, fill_value=False)

    stops = ordered[
        ordered["PitIn"].astype(bool)
        & out_lap_is_pit_out.astype(bool)
        & ordered["Green"]
        & out_lap_green.astype(bool)
        & ordered["LapTimeS"].notna()
        & out_lap_time.notna()
    ].copy()

    stops["OutLapTimeS"] = out_lap_time.loc[stops.index]
    stops = stops.join(pace, on=DRIVER_KEYS).dropna(subset=["MedianPace"])

    stops["PitLossS"] = (
        stops["LapTimeS"] + stops["OutLapTimeS"] - 2.0 * stops["MedianPace"]
    )

    # Discard slow stops (repairs, penalties served in the pit lane).
    stops = stops[stops["PitLossS"].between(10.0, 45.0)]

    pit_loss = (
        stops.groupby("Circuit")["PitLossS"]
        .agg(pit_loss_s="median", n_stops="size")
        .reset_index()
    )

    overall = pd.DataFrame(
        [
            {
                "Circuit": "ALL",
                "pit_loss_s": stops["PitLossS"].median(),
                "n_stops": len(stops),
            }
        ]
    )

    return pd.concat([pit_loss, overall], ignore_index=True)


def main() -> int:
    if not LAPS_PATH.exists():
        print(f"ERROR: {LAPS_PATH} not found. Run 01_download_data.py first.")
        return 1

    laps = pd.read_csv(LAPS_PATH)

    laps["Circuit"] = (
        laps["Location"].str.strip().replace(CIRCUIT_ALIASES)
    )
    laps["Split"] = laps["Year"].map(assign_split)
    laps["Compound"] = laps["Compound"].astype(str).str.upper()

    print(f"Raw laps: {len(laps):,} from "
          f"{laps.groupby(RACE_KEYS).ngroups} races")

    # -----------------------------------------------------------------
    # Race-level table and wet-race filter
    # -----------------------------------------------------------------
    laps["WetTyre"] = laps["Compound"].isin(["INTERMEDIATE", "WET"])
    laps["Green"] = laps["TrackStatus"].astype(str) == "1"

    races = (
        laps.groupby(RACE_KEYS)
        .agg(
            EventName=("EventName", "first"),
            Circuit=("Circuit", "first"),
            Split=("Split", "first"),
            TotalLaps=("TotalLaps", "first"),
            wet_share=("WetTyre", "mean"),
            green_share=("Green", "mean"),
        )
        .reset_index()
    )
    races["is_dry"] = races["wet_share"] <= WET_RACE_SHARE

    laps = laps.merge(races[RACE_KEYS + ["is_dry"]], on=RACE_KEYS)
    laps = laps[laps["is_dry"]].copy()

    print(f"Dry races kept: {int(races['is_dry'].sum())} of {len(races)}")

    # -----------------------------------------------------------------
    # Clean laps, baselines, target
    # -----------------------------------------------------------------
    laps["Clean"] = flag_clean_laps(laps)
    baselines = build_baselines(laps)

    model_laps = laps[laps["Clean"]].merge(baselines, on=DRIVER_KEYS)
    model_laps = model_laps[
        model_laps["LapNumber"] > model_laps["baseline_end_lap"]
    ].copy()

    model_laps["delta_pct"] = 100.0 * (
        model_laps["LapTimeS"] / model_laps["baseline_time"] - 1.0
    )

    # Fuel burns off steadily, so laps since the baseline is the fuel proxy.
    model_laps["laps_since_baseline_frac"] = (
        model_laps["LapNumber"] - model_laps["baseline_lap"]
    ) / model_laps["TotalLaps"]
    model_laps["lap_frac"] = model_laps["LapNumber"] / model_laps["TotalLaps"]

    model_laps = model_laps.rename(
        columns={
            "TyreLife": "tyre_life",
            "Compound": "compound",
            "Circuit": "circuit",
            "TotalLaps": "total_laps",
        }
    )

    columns = [
        "Year", "Round", "EventName", "circuit", "Split", "Driver", "Team",
        "LapNumber", "Stint", "LapTimeS", "delta_pct",
        "tyre_life", "compound", "laps_since_baseline_frac", "lap_frac",
        "total_laps", "baseline_time", "baseline_lap", "baseline_end_lap",
        "baseline_compound", "baseline_tyre_life", "baseline_track_temp",
        "FinishPosition",
    ]
    model_laps = model_laps[columns].sort_values(
        DRIVER_KEYS + ["LapNumber"]
    )

    clean_counts = (
        model_laps.groupby(RACE_KEYS).size().rename("n_model_laps")
    )
    races = races.join(clean_counts, on=RACE_KEYS)
    races["n_model_laps"] = races["n_model_laps"].fillna(0).astype(int)

    pit_loss = build_pit_loss(laps)

    model_laps.to_csv(MODEL_LAPS_PATH, index=False)
    races.to_csv(RACES_PATH, index=False)
    pit_loss.to_csv(PIT_LOSS_PATH, index=False)

    # -----------------------------------------------------------------
    # Console summary
    # -----------------------------------------------------------------
    print(f"\nModel laps: {len(model_laps):,}")
    print(
        model_laps.assign(
            race=model_laps["Year"] * 100 + model_laps["Round"]
        )
        .groupby("Split")
        .agg(
            races=("race", "nunique"),
            laps=("delta_pct", "size"),
            mean_delta_pct=("delta_pct", "mean"),
            std_delta_pct=("delta_pct", "std"),
        )
        .round(3)
        .to_string()
    )

    print("\nCircuits per split:")
    print(
        races[races["is_dry"]]
        .pivot_table(index="Circuit", columns="Split", values="Round",
                     aggfunc="count", fill_value=0)
        .to_string()
    )

    print("\nPit loss (s):")
    print(pit_loss.round(2).to_string(index=False))

    print(f"\nSaved: {MODEL_LAPS_PATH}\n       {RACES_PATH}\n       {PIT_LOSS_PATH}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
