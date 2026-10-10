"""
race_animation.py

Animated race between pit strategies.

Each "car" is the same driver running a different strategy. All cars start
together at the first simulated lap. Their lap times come from the lap-time
model, so the animation shows what the simulator predicts: who is ahead on
each lap, when a stop costs track position, and how the gap evolves.

This module prepares the numbers. The drawing itself (circuit, cars, timing
tower, gap chart, playback controls) lives in race_view.html, which is
filled with the race data and embedded in the app.

The circuit outline is real (see fetch_circuits.py). The position of the
start line on it and the direction of travel are illustrative: the lap data
holds lap times, not car positions.
"""

from __future__ import annotations

import json

import numpy as np

from common import DATA_DIR, PROJECT_ROOT
from strategy import RaceContext, predict_laps

CIRCUITS_PATH = DATA_DIR / "circuits.json"
TEMPLATE_PATH = PROJECT_ROOT / "race_view.html"
DATA_PLACEHOLDER = "__RACE_DATA__"


# =====================================================================
# Lap times for each car
# =====================================================================


def stint_index(lap_numbers: np.ndarray, stops: list[tuple[int, str]]) -> np.ndarray:
    """Which stint (0, 1, 2, ...) each simulated lap belongs to."""
    index = np.zeros(len(lap_numbers), dtype=int)
    for pit_lap, _ in sorted(stops):
        index += (lap_numbers > pit_lap).astype(int)
    return index


def simulate_cars(
    bundle: dict,
    context: RaceContext,
    strategies: dict[str, list[tuple[int, str]]],
    random_seed: int | None = None,
) -> dict[str, dict]:
    """
    Lap times and the time-to-distance curve for every car.

    With ``random_seed=None`` the lap times are the model's predictions.
    With a seed, one random race is drawn using the model's validation
    error: lap-to-lap noise and an offset per stint. The cars are the same
    driver on the same day, so they share the lap noise and the offset of
    the opening stint; only stints after a stop get their own offset.
    """
    rng = None if random_seed is None else np.random.default_rng(random_seed)

    stint_sd = bundle["residual"]["stint_sd_pct"]
    lap_sd = bundle["residual"]["lap_sd_pct"]

    shared_lap_noise = None
    shared_first_stint = 0.0

    if rng is not None:
        shared_lap_noise = rng.normal(0.0, lap_sd, size=context.n_laps)
        shared_first_stint = rng.normal(0.0, stint_sd)

    cars = {}

    for name, stops in strategies.items():
        laps = predict_laps(bundle, context, stops)
        lap_numbers = laps["LapNumber"].to_numpy()
        delta_pct = laps["pred_delta_pct"].to_numpy(dtype=float)

        if rng is not None:
            stints = stint_index(lap_numbers, stops)
            offsets = np.concatenate(
                [[shared_first_stint], rng.normal(0.0, stint_sd, len(stops))]
            )
            delta_pct = delta_pct + shared_lap_noise + offsets[stints]

        car = {
            "stops": sorted(stops),
            "lap_numbers": lap_numbers,
            "compound": laps["compound"].to_numpy(),
            "tyre_age": laps["tyre_life"].to_numpy(dtype=float),
            "lap_times": context.baseline_time * (1.0 + delta_pct / 100.0),
        }
        add_timeline(car, context)
        cars[name] = car

    return cars


def add_timeline(car: dict, context: RaceContext) -> None:
    """
    Attach the car's time-to-distance curve.

    ``knot_time`` and ``knot_progress`` describe laps completed as a
    function of race time. A pit stop is a flat section: the car reaches
    the line, stands still for the pit-loss time, then continues.
    """
    pit_laps = {lap for lap, _ in car["stops"]}

    times = [0.0]
    progress = [float(context.start_lap - 1)]
    pit_windows = []
    lap_end_time = []

    clock = 0.0
    for lap, lap_time in zip(car["lap_numbers"], car["lap_times"]):
        clock += float(lap_time)
        times.append(clock)
        progress.append(float(lap))

        if lap in pit_laps:
            pit_windows.append((clock, clock + context.pit_loss_s))
            clock += context.pit_loss_s
            times.append(clock)
            progress.append(float(lap))

        # Time at which the car starts the next lap, pit stop included.
        lap_end_time.append(clock)

    car["knot_time"] = np.array(times)
    car["knot_progress"] = np.array(progress)
    car["pit_windows"] = pit_windows
    car["lap_end_time"] = np.array(lap_end_time)
    car["finish_time"] = clock


def results_table(cars: dict[str, dict]) -> list[dict]:
    """Finishing order with race time and gap to the winner."""
    ordered = sorted(cars.items(), key=lambda item: item[1]["finish_time"])
    winner_time = ordered[0][1]["finish_time"]

    return [
        {
            "Position": position,
            "Car": name,
            "Stops": len(car["stops"]),
            "Race time (s)": car["finish_time"],
            "Gap to winner (s)": car["finish_time"] - winner_time,
        }
        for position, (name, car) in enumerate(ordered, start=1)
    ]


# =====================================================================
# Circuit outline
# =====================================================================


def generic_outline(n_points: int = 160) -> list[list[float]]:
    """A circuit-like loop, used when no real outline is available."""
    theta = np.linspace(0.0, 2.0 * np.pi, n_points, endpoint=False)

    radius = (
        1.0
        + 0.22 * np.cos(2.0 * theta + 0.6)
        + 0.12 * np.cos(3.0 * theta - 1.1)
        + 0.06 * np.cos(5.0 * theta + 2.0)
    )

    return np.column_stack(
        [1.7 * radius * np.cos(theta), radius * np.sin(theta)]
    ).round(4).tolist()


def circuit_outline(circuit: str) -> tuple[list[list[float]], bool]:
    """Outline points for a circuit and whether they are the real layout."""
    if CIRCUITS_PATH.exists():
        outlines = json.loads(CIRCUITS_PATH.read_text(encoding="utf-8"))
        if circuit in outlines:
            return outlines[circuit], True

    return generic_outline(), False


# =====================================================================
# Race view
# =====================================================================


def race_payload(
    context: RaceContext,
    cars: dict[str, dict],
    colors: dict[str, str],
    short_names: dict[str, str],
    title: str,
) -> dict:
    """Everything the race view needs, as plain JSON-ready values."""
    outline, is_real = circuit_outline(context.circuit)

    return {
        "title": title,
        "startLap": context.start_lap,
        "totalLaps": context.total_laps,
        "averageLap": float(
            np.mean([car["lap_times"].mean() for car in cars.values()])
        ),
        "track": outline,
        "trackNote": (
            f"{context.circuit} layout. Start line position and direction "
            "of travel are illustrative."
            if is_real
            else "Generic track shape, not the real circuit."
        ),
        "cars": [
            {
                "name": name,
                "short": short_names[name],
                "color": colors[name],
                "knotT": car["knot_time"].round(3).tolist(),
                "knotP": car["knot_progress"].tolist(),
                "pits": [list(window) for window in car["pit_windows"]],
                "lapNumbers": car["lap_numbers"].tolist(),
                "compound": [str(c) for c in car["compound"]],
                "tyreAge": car["tyre_age"].tolist(),
                "lapEnd": car["lap_end_time"].round(3).tolist(),
                "finish": float(car["finish_time"]),
            }
            for name, car in cars.items()
        ],
    }


def render_race_html(payload: dict) -> str:
    """Fill the race-view template with one race's data."""
    template = TEMPLATE_PATH.read_text(encoding="utf-8")

    # "</" is escaped so the data can never close the script element.
    data = json.dumps(payload, ensure_ascii=False).replace("</", "<\\/")

    return template.replace(DATA_PLACEHOLDER, data)
