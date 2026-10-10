"""Shared paths and constants for the race strategy simulator."""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
MODEL_DIR = PROJECT_ROOT / "models"
REPORT_DIR = PROJECT_ROOT / "reports"

# FastF1's cache uses long nested file names. Keeping it in a short path
# avoids the 260-character Windows path limit.
FASTF1_CACHE_DIR = Path.home() / ".cache" / "fastf1"

LAPS_PATH = DATA_DIR / "laps.csv"
MODEL_LAPS_PATH = DATA_DIR / "model_laps.csv"
PIT_LOSS_PATH = DATA_DIR / "pit_loss.csv"
RACES_PATH = DATA_DIR / "races.csv"
MODEL_PATH = MODEL_DIR / "lap_time_model.pkl"

# Temporal split: the model never sees a later season than it is tested on.
# 2022-2025 is one technical-regulation era, so the cars are comparable.
TRAIN_YEARS = [2022, 2023]
VALIDATION_YEARS = [2024]
TEST_YEARS = [2025]
ALL_YEARS = TRAIN_YEARS + VALIDATION_YEARS + TEST_YEARS

DRY_COMPOUNDS = ["SOFT", "MEDIUM", "HARD"]

# A driver's baseline pace is the median of their first clean laps. It is
# known a few laps into the race, before any normal strategy decision, so
# predicting later laps relative to it uses no future information.
BASELINE_FIRST_LAP = 2
BASELINE_LAST_LAP = 12
BASELINE_MIN_LAPS = 3
BASELINE_MAX_LAPS = 8

RANDOM_STATE = 42
