"""
01_download_data.py

Download real Formula 1 race lap data with FastF1.

One CSV per race is written to data/raw/, so the script can be stopped and
restarted: races that are already on disk are skipped. All races are then
combined into data/laps.csv.

Run
---
python 01_download_data.py            (all seasons)
python 01_download_data.py 2024 2025  (only these seasons)
"""

from __future__ import annotations

import sys
import time

import fastf1
import pandas as pd

from common import ALL_YEARS, FASTF1_CACHE_DIR, LAPS_PATH, RAW_DIR

MAX_ATTEMPTS = 3


def download_race(year: int, event: pd.Series) -> pd.DataFrame:
    """Load one race and return a flat per-lap table."""
    round_number = int(event["RoundNumber"])

    session = fastf1.get_session(year, round_number, "R")
    session.load(laps=True, telemetry=False, weather=True, messages=False)

    laps = session.laps.reset_index(drop=True)
    weather = laps.get_weather_data().reset_index(drop=True)
    results = session.results.set_index("Abbreviation")

    table = pd.DataFrame(
        {
            "Year": year,
            "Round": round_number,
            "EventName": event["EventName"],
            "Location": event["Location"],
            "TotalLaps": int(laps["LapNumber"].max()),
            "Driver": laps["Driver"],
            "Team": laps["Team"],
            "LapNumber": laps["LapNumber"].astype(int),
            "LapTimeS": laps["LapTime"].dt.total_seconds(),
            "Stint": laps["Stint"],
            "Compound": laps["Compound"],
            "TyreLife": laps["TyreLife"],
            "FreshTyre": laps["FreshTyre"],
            "TrackStatus": laps["TrackStatus"].astype(str),
            "IsAccurate": laps["IsAccurate"],
            "PitIn": laps["PitInTime"].notna(),
            "PitOut": laps["PitOutTime"].notna(),
            "Position": laps["Position"],
            "AirTemp": weather["AirTemp"],
            "TrackTemp": weather["TrackTemp"],
            "Rainfall": weather["Rainfall"],
        }
    )

    table["GridPosition"] = table["Driver"].map(results["GridPosition"])
    table["FinishPosition"] = table["Driver"].map(results["Position"])

    return table


def main() -> int:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    FASTF1_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    fastf1.Cache.enable_cache(str(FASTF1_CACHE_DIR))
    fastf1.set_log_level("ERROR")

    failed: list[str] = []

    # Optional: pass seasons on the command line to download only those,
    # which lets several seasons download in parallel.
    years = [int(arg) for arg in sys.argv[1:]] or ALL_YEARS

    for year in years:
        schedule = None

        for attempt in range(1, MAX_ATTEMPTS + 1):
            try:
                schedule = fastf1.get_event_schedule(
                    year, include_testing=False
                )
                break
            except Exception as exc:
                print(
                    f"attempt {attempt} failed for the {year} schedule: "
                    f"{type(exc).__name__}: {exc}",
                    flush=True,
                )
                time.sleep(5 * attempt)

        if schedule is None:
            failed.append(f"{year} season schedule")
            continue

        for _, event in schedule.iterrows():
            round_number = int(event["RoundNumber"])
            label = f"{year} round {round_number:02d} {event['EventName']}"
            path = RAW_DIR / f"{year}_{round_number:02d}.csv"

            if path.exists():
                print(f"have     {label}", flush=True)
                continue

            for attempt in range(1, MAX_ATTEMPTS + 1):
                try:
                    start = time.perf_counter()
                    table = download_race(year, event)
                    table.to_csv(path, index=False)
                    print(
                        f"saved    {label}: {len(table):,} laps "
                        f"({time.perf_counter() - start:.0f} s)",
                        flush=True,
                    )
                    break
                except Exception as exc:
                    print(
                        f"attempt {attempt} failed for {label}: "
                        f"{type(exc).__name__}: {exc}",
                        flush=True,
                    )
                    time.sleep(5 * attempt)
            else:
                failed.append(label)

    files = sorted(RAW_DIR.glob("*.csv"))

    if not files:
        print("ERROR: no race data was downloaded.")
        return 1

    laps = pd.concat((pd.read_csv(f) for f in files), ignore_index=True)
    laps.to_csv(LAPS_PATH, index=False)

    print(f"\nCombined {len(files)} races, {len(laps):,} laps -> {LAPS_PATH}")

    if failed:
        print("Races that could not be downloaded:")
        for label in failed:
            print(f"  - {label}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
