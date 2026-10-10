"""
strategy.py

The race strategy simulation engine.

Everything here is shared by the backtest (04) and the interactive app (05):

    lap-time model  ->  predicted lap times for any tyre plan
                    ->  total race time for every candidate pit strategy
                    ->  best strategy, plus a Monte Carlo spread

Conventions
-----------
A strategy is a list of stops ``[(pit_lap, compound), ...]``. The car pits at
the END of ``pit_lap`` and runs the new compound, on fresh tyres, from the
next lap. Each stop adds the circuit's pit-loss time.

The simulation covers the laps after the driver's baseline window, because
the baseline pace is what the predictions are scaled from.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product

import joblib
import numpy as np
import pandas as pd

from common import DRY_COMPOUNDS, MODEL_PATH

NUMERIC_FEATURES = [
    "tyre_life",
    "laps_since_baseline_frac",
    "lap_frac",
    "total_laps",
    "baseline_tyre_life",
    "baseline_track_temp",
]

CATEGORICAL_FEATURES = [
    "compound",
    "baseline_compound",
    "circuit",
]

FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET = "delta_pct"

# Shortest stint the optimiser will consider.
MIN_STINT_LAPS = 5

# Two-stop pit laps are searched on this grid to keep the search fast.
TWO_STOP_STEP = 2

# Real pit-lane entries this close to the flag are not counted as stops.
FINAL_LAPS_WITHOUT_STOPS = 3


@dataclass
class RaceContext:
    """What is known about one driver's race once the baseline is set."""

    circuit: str
    total_laps: int
    baseline_time: float
    baseline_lap: float
    baseline_compound: str
    baseline_tyre_life: float
    baseline_track_temp: float
    start_lap: int            # first lap that is simulated
    start_compound: str       # tyre fitted at start_lap
    start_tyre_life: float    # its age at start_lap
    pit_loss_s: float

    @property
    def n_laps(self) -> int:
        return self.total_laps - self.start_lap + 1


def load_bundle() -> dict:
    """Load the trained lap-time model and its supporting tables."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"{MODEL_PATH} not found. Run 03_train_models.py first."
        )
    return joblib.load(MODEL_PATH)


def pit_loss_for(bundle: dict, circuit: str) -> float:
    """Pit-stop time loss for a circuit, falling back to the overall median."""
    table = bundle["pit_loss"]
    return float(table.get(circuit, table["ALL"]))


def max_stint_for(bundle: dict, circuit: str, compound: str) -> int:
    """
    Longest stint the optimiser may plan on a compound at a circuit.

    Tree models cannot extrapolate tyre wear beyond what they have seen, so
    stints are limited to tyre ages that were actually observed.
    """
    table = bundle["max_tyre_life"]
    key = (circuit, compound)

    if key in table:
        return int(table[key])

    return int(table[("ALL", compound)])


# =====================================================================
# Lap-by-lap prediction
# =====================================================================


def tyre_plan_arrays(
    context: RaceContext,
    stops: list[tuple[int, str]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Lap numbers, compound and tyre age for every simulated lap."""
    laps = np.arange(context.start_lap, context.total_laps + 1)

    compound = np.full(len(laps), context.start_compound, dtype=object)
    tyre_life = context.start_tyre_life + (laps - context.start_lap)

    for pit_lap, new_compound in sorted(stops):
        after = laps > pit_lap
        compound[after] = new_compound
        tyre_life = np.where(after, laps - pit_lap, tyre_life)

    return laps, compound, tyre_life.astype(float)


def tyre_plan(
    context: RaceContext,
    stops: list[tuple[int, str]],
) -> pd.DataFrame:
    """Compound and tyre age for every simulated lap of a strategy."""
    laps, compound, tyre_life = tyre_plan_arrays(context, stops)

    return pd.DataFrame(
        {
            "LapNumber": laps,
            "compound": compound,
            "tyre_life": tyre_life,
        }
    )


def add_context_features(
    plan: pd.DataFrame,
    context: RaceContext,
) -> pd.DataFrame:
    """Attach the race-level model inputs to a lap-by-lap tyre plan."""
    plan = plan.copy()

    plan["laps_since_baseline_frac"] = (
        plan["LapNumber"] - context.baseline_lap
    ) / context.total_laps
    plan["lap_frac"] = plan["LapNumber"] / context.total_laps
    plan["total_laps"] = context.total_laps
    plan["baseline_tyre_life"] = context.baseline_tyre_life
    plan["baseline_track_temp"] = context.baseline_track_temp
    plan["baseline_compound"] = context.baseline_compound
    plan["circuit"] = context.circuit

    return plan


def predict_laps(
    bundle: dict,
    context: RaceContext,
    stops: list[tuple[int, str]],
) -> pd.DataFrame:
    """Predicted lap time for every simulated lap of one strategy."""
    plan = add_context_features(tyre_plan(context, stops), context)

    plan["pred_delta_pct"] = bundle["model"].predict(plan[FEATURES])
    plan["pred_lap_time"] = context.baseline_time * (
        1.0 + plan["pred_delta_pct"] / 100.0
    )

    return plan


def strategy_time(
    bundle: dict,
    context: RaceContext,
    stops: list[tuple[int, str]],
) -> float:
    """Predicted time from start_lap to the flag, including pit stops."""
    laps = predict_laps(bundle, context, stops)
    return float(laps["pred_lap_time"].sum() + len(stops) * context.pit_loss_s)


def wear_curve(
    bundle: dict,
    context: RaceContext,
    compound: str,
) -> pd.DataFrame:
    """
    Predicted lap-time change as one compound ages at this circuit.

    Fuel load is held at its mid-race value, so the curve isolates tyre
    wear from the car getting lighter.
    """
    ages = np.arange(
        1, max_stint_for(bundle, context.circuit, compound) + 1
    )
    mid_lap = context.total_laps / 2.0

    plan = pd.DataFrame(
        {
            "LapNumber": mid_lap,
            "compound": compound,
            "tyre_life": ages.astype(float),
        }
    )
    plan = add_context_features(plan, context)
    plan["pred_delta_pct"] = bundle["model"].predict(plan[FEATURES])
    plan["pred_lap_time"] = context.baseline_time * (
        1.0 + plan["pred_delta_pct"] / 100.0
    )

    return plan[["tyre_life", "compound", "pred_delta_pct", "pred_lap_time"]]


# =====================================================================
# Strategy search
# =====================================================================


def describe_strategy(
    context: RaceContext,
    stops: list[tuple[int, str]],
) -> str:
    """Readable label such as 'MEDIUM -> lap 22 HARD'."""
    parts = [context.start_compound]
    parts.extend(f"lap {lap} {compound}" for lap, compound in sorted(stops))
    return " -> ".join(parts)


def stint_lengths(
    context: RaceContext,
    stops: list[tuple[int, str]],
) -> list[tuple[str, float]]:
    """(compound, tyre age at the end of the stint) for every stint."""
    ordered = sorted(stops)
    stints = []

    compound = context.start_compound
    start_age = context.start_tyre_life
    stint_start = context.start_lap

    for pit_lap, new_compound in ordered:
        stints.append((compound, start_age + (pit_lap - stint_start)))
        compound, start_age, stint_start = new_compound, 1.0, pit_lap + 1

    stints.append(
        (compound, start_age + (context.total_laps - stint_start))
    )

    return stints


def is_valid_strategy(
    bundle: dict,
    context: RaceContext,
    stops: list[tuple[int, str]],
    race_compounds: list[str],
) -> bool:
    """
    Apply the racing rules and the model's safe range.

    - at least two different dry compounds must be used in the race
    - no stint may run longer than tyres were observed to last
    - every stint must be at least MIN_STINT_LAPS long
    """
    ordered = sorted(stops)
    pit_laps = [lap for lap, _ in ordered]

    boundaries = [context.start_lap - 1] + pit_laps + [context.total_laps]
    if any(b - a < MIN_STINT_LAPS for a, b in zip(boundaries, boundaries[1:])):
        return False

    used = set(race_compounds) | {compound for _, compound in ordered}
    if len(used) < 2:
        return False

    for compound, end_age in stint_lengths(context, stops):
        if end_age > max_stint_for(bundle, context.circuit, compound):
            return False

    return True


def candidate_strategies(
    bundle: dict,
    context: RaceContext,
    race_compounds: list[str],
    max_stops: int = 2,
) -> list[list[tuple[int, str]]]:
    """
    Every valid 1-stop strategy and a grid of 2-stop strategies.

    ``race_compounds`` are the dry compounds the driver has already used
    (normally just the starting tyre), needed for the two-compound rule.
    """
    first = context.start_lap + MIN_STINT_LAPS - 1
    last = context.total_laps - MIN_STINT_LAPS

    candidates: list[list[tuple[int, str]]] = [[]]

    for pit_lap, compound in product(range(first, last + 1), DRY_COMPOUNDS):
        candidates.append([(pit_lap, compound)])

    if max_stops >= 2:
        grid = range(first, last + 1, TWO_STOP_STEP)
        for lap_1, lap_2 in product(grid, grid):
            if lap_2 - lap_1 < MIN_STINT_LAPS:
                continue
            for c_1, c_2 in product(DRY_COMPOUNDS, DRY_COMPOUNDS):
                candidates.append([(lap_1, c_1), (lap_2, c_2)])

    return [
        stops
        for stops in candidates
        if is_valid_strategy(bundle, context, stops, race_compounds)
    ]


def evaluate_strategies(
    bundle: dict,
    context: RaceContext,
    strategies: list[list[tuple[int, str]]],
) -> pd.DataFrame:
    """
    Predicted race time for many strategies in one model call.

    Returns one row per strategy, fastest first.
    """
    if not strategies:
        return pd.DataFrame(
            columns=["strategy_id", "stops", "n_stops", "label", "time_s"]
        )

    # Build all tyre plans as arrays first; one DataFrame per strategy
    # would dominate the run time when thousands are searched.
    plans = [tyre_plan_arrays(context, stops) for stops in strategies]
    n_laps = context.n_laps

    laps = add_context_features(
        pd.DataFrame(
            {
                "strategy_id": np.repeat(np.arange(len(strategies)), n_laps),
                "LapNumber": np.concatenate([p[0] for p in plans]),
                "compound": np.concatenate([p[1] for p in plans]),
                "tyre_life": np.concatenate([p[2] for p in plans]),
            }
        ),
        context,
    )
    laps["pred_lap_time"] = context.baseline_time * (
        1.0 + bundle["model"].predict(laps[FEATURES]) / 100.0
    )

    running_time = laps.groupby("strategy_id")["pred_lap_time"].sum()

    table = pd.DataFrame(
        {
            "strategy_id": range(len(strategies)),
            "stops": strategies,
            "n_stops": [len(stops) for stops in strategies],
            "label": [describe_strategy(context, s) for s in strategies],
        }
    )
    table["time_s"] = (
        running_time.reindex(table["strategy_id"]).to_numpy()
        + table["n_stops"] * context.pit_loss_s
    )

    return table.sort_values("time_s").reset_index(drop=True)


def best_strategies(
    bundle: dict,
    context: RaceContext,
    race_compounds: list[str],
    max_stops: int = 2,
) -> pd.DataFrame:
    """Search all candidate strategies and rank them by predicted time."""
    table = evaluate_strategies(
        bundle,
        context,
        candidate_strategies(bundle, context, race_compounds, max_stops),
    )

    if not table.empty:
        table["gap_to_best_s"] = table["time_s"] - table["time_s"].iloc[0]

    return table


# =====================================================================
# Uncertainty
# =====================================================================


def simulate_race_times(
    bundle: dict,
    context: RaceContext,
    stops: list[tuple[int, str]],
    n_simulations: int = 2000,
    seed: int = 42,
) -> np.ndarray:
    """
    Monte Carlo distribution of race time for one strategy.

    The model's validation errors are split into a stint-level offset (a
    whole stint running faster or slower than predicted, which does not
    average out) and lap-to-lap noise (which largely does). Each simulated
    race draws one offset per stint and one noise term per lap.

    The driver-level error (a mis-judged baseline pace) is left out on
    purpose: it moves every strategy for the driver by the same amount, so
    it says nothing about which strategy is faster.
    """
    rng = np.random.default_rng(seed)
    laps = predict_laps(bundle, context, stops)

    stint_sd = bundle["residual"]["stint_sd_pct"]
    lap_sd = bundle["residual"]["lap_sd_pct"]

    stint_index = np.zeros(len(laps), dtype=int)
    for pit_lap, _ in sorted(stops):
        stint_index += (laps["LapNumber"].to_numpy() > pit_lap).astype(int)

    n_stints = len(stops) + 1

    stint_offsets = rng.normal(0.0, stint_sd, size=(n_simulations, n_stints))
    lap_noise = rng.normal(0.0, lap_sd, size=(n_simulations, len(laps)))

    delta_pct = (
        laps["pred_delta_pct"].to_numpy()[None, :]
        + stint_offsets[:, stint_index]
        + lap_noise
    )

    lap_times = context.baseline_time * (1.0 + delta_pct / 100.0)

    return lap_times.sum(axis=1) + len(stops) * context.pit_loss_s


# =====================================================================
# Building a context from real race data
# =====================================================================


def context_from_laps(
    bundle: dict,
    driver_laps: pd.DataFrame,
    all_driver_laps: pd.DataFrame,
) -> RaceContext:
    """
    Build the simulation context for one driver in one real race.

    ``driver_laps`` are the driver's rows from data/model_laps.csv (clean
    laps after the baseline). ``all_driver_laps`` are all their laps from
    data/laps.csv, used to read the tyre actually fitted at the start lap.
    """
    first = driver_laps.iloc[0]
    start_lap = int(first["baseline_end_lap"]) + 1

    raw = all_driver_laps.set_index("LapNumber")
    start_row = raw.loc[start_lap] if start_lap in raw.index else None

    if start_row is not None and start_row["Compound"] in DRY_COMPOUNDS:
        start_compound = start_row["Compound"]
        start_tyre_life = float(start_row["TyreLife"])
    else:
        start_compound = first["baseline_compound"]
        start_tyre_life = float(first["baseline_tyre_life"]) + (
            start_lap - float(first["baseline_lap"])
        )

    circuit = first["circuit"]

    return RaceContext(
        circuit=circuit,
        total_laps=int(first["total_laps"]),
        baseline_time=float(first["baseline_time"]),
        baseline_lap=float(first["baseline_lap"]),
        baseline_compound=first["baseline_compound"],
        baseline_tyre_life=float(first["baseline_tyre_life"]),
        baseline_track_temp=float(first["baseline_track_temp"]),
        start_lap=start_lap,
        start_compound=start_compound,
        start_tyre_life=start_tyre_life,
        pit_loss_s=pit_loss_for(bundle, circuit),
    )


def actual_stops(
    all_driver_laps: pd.DataFrame,
    context: RaceContext,
) -> list[tuple[int, str]]:
    """
    The pit stops the driver really made from the start lap onwards.

    Pit-lane passes in the last laps are ignored: they are not tyre
    strategy (for example a race that finishes behind the safety car).
    """
    laps = all_driver_laps.sort_values("LapNumber")
    stops = []

    for _, row in laps[laps["PitIn"].astype(bool)].iterrows():
        pit_lap = int(row["LapNumber"])
        if (
            pit_lap < context.start_lap
            or pit_lap > context.total_laps - FINAL_LAPS_WITHOUT_STOPS
        ):
            continue

        after = laps[laps["LapNumber"] == pit_lap + 1]
        if after.empty:
            continue

        stops.append((pit_lap, str(after.iloc[0]["Compound"]).upper()))

    return stops
