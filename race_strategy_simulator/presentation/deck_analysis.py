"""
Numbers and figures for the mathematical version of the presentation.

Everything is recomputed from the project's data and saved model, and written
to presentation/assets/ (numbers.json and PNG figures). build_deck.py reads
those files.

Run from the project folder:
    python presentation/deck_analysis.py
"""
from __future__ import annotations

import importlib
import json
import math
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))

from common import DRY_COMPOUNDS, LAPS_PATH, MODEL_LAPS_PATH, REPORT_DIR  # noqa: E402
from strategy import (  # noqa: E402
    FEATURES,
    MIN_STINT_LAPS,
    TARGET,
    TWO_STOP_STEP,
    best_strategies,
    candidate_strategies,
    context_from_laps,
    actual_stops,
    evaluate_strategies,
    load_bundle,
    simulate_race_times,
)

train_models = importlib.import_module("03_train_models")

ASSETS = Path(__file__).resolve().parent / "assets"
ASSETS.mkdir(exist_ok=True)

INK = "#14171A"
RED = "#E10600"
GREY = "#B4B9C0"
MUTED = "#5B6168"
COMPOUND_LINE = {"SOFT": RED, "MEDIUM": "#C99700", "HARD": INK}

plt.rcParams.update({
    "font.family": "Arial",
    "font.size": 13,
    "axes.edgecolor": GREY,
    "axes.labelcolor": MUTED,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "axes.grid": True,
    "grid.color": "#E6E8EB",
    "grid.linewidth": 0.8,
    "legend.frameon": False,
    "savefig.dpi": 220,
    "savefig.bbox": "tight",
    "savefig.facecolor": "white",
    "mathtext.fontset": "cm",
})

numbers: dict = {}


def rmse(a, b) -> float:
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


laps = pd.read_csv(MODEL_LAPS_PATH)
train = laps[laps["Split"] == "train"]
validation = laps[laps["Split"] == "validation"]
y_train = train[TARGET].to_numpy()
y_val = validation[TARGET].to_numpy()

bundle = load_bundle()
sd = bundle["residual"]
numbers["sd"] = sd
numbers["final_model_iterations"] = int(bundle["model"].named_steps["model"].n_iter_)

# ---------------------------------------------------------------- baseline ---
# MSE of predicting zero change = (mean of y)^2 + variance of y.
numbers["val_mean_y"] = float(y_val.mean())
numbers["val_sd_y"] = float(y_val.std())
numbers["val_rmse_zero"] = rmse(y_val, 0.0)
numbers["train_mean_y"] = float(y_train.mean())

# ---------------------------------------------------- boosting, stage by stage
models = train_models.create_models()
boosted = models["Gradient boosting"].fit(train[FEATURES], y_train)
encoder, booster = boosted.named_steps["encode"], boosted.named_steps["model"]
numbers["train_fit_iterations"] = int(booster.n_iter_)
numbers["early_stopping"] = bool(booster.do_early_stopping_)

x_train, x_val = encoder.transform(train[FEATURES]), encoder.transform(validation[FEATURES])
stage_train = [rmse(y_train, p) for p in booster.staged_predict(x_train)]
stage_val = [rmse(y_val, p) for p in booster.staged_predict(x_val)]
val_prediction = boosted.predict(validation[FEATURES])
numbers["val_rmse_boosting"] = rmse(y_val, val_prediction)
numbers["train_rmse_boosting"] = stage_train[-1]

linear = models["Linear regression"].fit(train[FEATURES], y_train)
numbers["val_rmse_linear"] = rmse(y_val, linear.predict(validation[FEATURES]))
numbers["train_rmse_linear"] = rmse(y_train, linear.predict(train[FEATURES]))
numbers["linear_coefficients"] = int(linear.named_steps["model"].coef_.size) + 1

fig, ax = plt.subplots(figsize=(6.4, 3.7))
stages = np.arange(1, len(stage_val) + 1)
ax.plot(stages, stage_train, color=GREY, linewidth=2.2, label="Training seasons")
ax.plot(stages, stage_val, color=RED, linewidth=2.6, label="Validation season")
ax.axhline(numbers["val_rmse_linear"], color=INK, linewidth=1.2, linestyle="--")
ax.text(stages[-1], numbers["val_rmse_linear"] + 0.008, "linear regression (validation)",
        ha="right", va="bottom", color=INK, fontsize=11)
ax.set_xlabel("Boosting iteration  $m$")
ax.set_ylabel("RMSE (% of pace)")
ax.legend(loc="upper right")
fig.savefig(ASSETS / "fig_boosting.png")
plt.close(fig)

# ------------------------------------------------------- tree size (leaves) ---
leaf_grid = [2, 3, 4, 6, 8, 12, 16, 31, 63]
complexity = []
for leaves in leaf_grid:
    candidate = clone(models["Gradient boosting"])
    candidate.set_params(model__max_leaf_nodes=leaves)
    candidate.fit(train[FEATURES], y_train)
    complexity.append({
        "leaves": leaves,
        "train": rmse(y_train, candidate.predict(train[FEATURES])),
        "validation": rmse(y_val, candidate.predict(validation[FEATURES])),
        "iterations": int(candidate.named_steps["model"].n_iter_),
    })
numbers["complexity"] = complexity

# The library's default settings: the first model that was tried.
default = clone(models["Gradient boosting"])
default.set_params(model__max_leaf_nodes=31, model__learning_rate=0.1,
                   model__min_samples_leaf=20, model__l2_regularization=0.0,
                   model__max_iter=100)
default.fit(train[FEATURES], y_train)
numbers["default_settings"] = {
    "train": rmse(y_train, default.predict(train[FEATURES])),
    "validation": rmse(y_val, default.predict(validation[FEATURES])),
}

fig, ax = plt.subplots(figsize=(6.4, 3.7))
ax.plot(leaf_grid, [c["train"] for c in complexity], color=GREY, linewidth=2.2,
        marker="o", label="Training seasons")
ax.plot(leaf_grid, [c["validation"] for c in complexity], color=RED, linewidth=2.6,
        marker="o", label="Validation season")
ax.axhline(numbers["val_rmse_linear"], color=INK, linewidth=1.2, linestyle="--")
ax.text(leaf_grid[-1], numbers["val_rmse_linear"] - 0.006, "linear regression (validation)",
        ha="right", va="top", color=INK, fontsize=11)
ax.set_xscale("log", base=2)
ax.set_xticks(leaf_grid)
ax.set_xticklabels([str(v) for v in leaf_grid])
ax.set_xlabel("Leaves per tree  $J$")
ax.set_ylabel("RMSE (% of pace)")
ax.legend(loc="center right")
fig.savefig(ASSETS / "fig_complexity.png")
plt.close(fig)

# ------------------------------------------------- error variance components ---
residual = y_val - val_prediction
variance = {k: sd[k] ** 2 for k in ("race_sd_pct", "stint_sd_pct", "lap_sd_pct")}
total = sum(variance.values())
cluster_size = float(validation.groupby(["Year", "Round", "Driver"]).size().mean())
rho = variance["race_sd_pct"] / total
numbers["variance"] = {
    "race": variance["race_sd_pct"], "stint": variance["stint_sd_pct"],
    "lap": variance["lap_sd_pct"], "sum": total,
    "residual_mse": float(np.mean(residual ** 2)),
    "residual_mean": float(residual.mean()),
    "share_race": variance["race_sd_pct"] / total,
    "share_stint": variance["stint_sd_pct"] / total,
    "share_lap": variance["lap_sd_pct"] / total,
    "laps_per_driver_race": cluster_size,
    "driver_races": int(validation.groupby(["Year", "Round", "Driver"]).ngroups),
    "design_effect": 1 + (cluster_size - 1) * rho,
    "effective_laps": len(validation) / (1 + (cluster_size - 1) * rho),
    "validation_laps": int(len(validation)),
}

# --------------------------------------- race-time error implied by the model ---
backtest = pd.read_csv(REPORT_DIR / "04_backtest" / "backtest_drivers.csv")
baseline = laps.groupby(["Year", "Round", "Driver"])["baseline_time"].first()


def stint_sizes(row) -> list[int]:
    """Laps in each stint of the simulated part of the race."""
    pits = [int(v) for v in re.findall(r"lap (\d+)", row["actual_strategy"])]
    edges = [row["start_lap"] - 1] + pits + [row["total_laps"]]
    return [b - a for a, b in zip(edges, edges[1:])]


green = backtest[backtest["green_race"]].copy()
implied = []
for _, row in green.iterrows():
    sizes = np.array(stint_sizes(row), dtype=float)
    n = sizes.sum()
    scale = baseline.loc[(row["Year"], row["Round"], row["Driver"])] / 100.0
    implied.append(scale * math.sqrt(
        n ** 2 * variance["race_sd_pct"]
        + (sizes ** 2).sum() * variance["stint_sd_pct"]
        + n * variance["lap_sd_pct"]))
green["implied_sd_s"] = implied
z = green["race_time_error_s"] / green["implied_sd_s"]
numbers["race_time"] = {
    "n": int(len(green)),
    "implied_sd_s": float(np.sqrt(np.mean(green["implied_sd_s"] ** 2))),
    "implied_mae_s": float(np.mean(green["implied_sd_s"]) * math.sqrt(2 / math.pi)),
    "observed_mae_s": float(green["race_time_error_s"].abs().mean()),
    "observed_rmse_s": float(np.sqrt(np.mean(green["race_time_error_s"] ** 2))),
    "observed_bias_s": float(green["race_time_error_s"].mean()),
    "z_sd": float(z.std()),
    "z_within_2": float((z.abs() < 2).mean()),
}

fig, ax = plt.subplots(figsize=(6.0, 3.5))
ax.hist(z, bins=np.arange(-3.5, 3.75, 0.5), density=True, color=GREY, edgecolor="white")
grid = np.linspace(-3.5, 3.5, 200)
ax.plot(grid, np.exp(-grid ** 2 / 2) / math.sqrt(2 * math.pi), color=RED, linewidth=2.6)
ax.set_xlabel(r"Standardised race-time error  $z = (\hat{R} - R)\,/\,\sigma_R$")
ax.set_ylabel("Density")
ax.grid(axis="x", visible=False)
fig.savefig(ASSETS / "fig_race_time_z.png")
plt.close(fig)

pit = backtest.dropna(subset=["actual_first_pit_lap", "recommended_first_pit_lap"])
difference = pit["recommended_first_pit_lap"] - pit["actual_first_pit_lap"]
numbers["pit_lap"] = {
    "n": int(len(pit)),
    "mean_signed_laps": float(difference.mean()),
    "share_later": float((difference > 0).mean()),
}

# ------------------------------------------- one race: objective and its noise ---
raw = pd.read_csv(LAPS_PATH, low_memory=False)
def search(row):
    """Context, team stops and ranked strategies for one backtest row."""
    mask = ((laps["Year"] == row["Year"]) & (laps["Round"] == row["Round"])
            & (laps["Driver"] == row["Driver"]))
    raw_mask = ((raw["Year"] == row["Year"]) & (raw["Round"] == row["Round"])
                & (raw["Driver"] == row["Driver"]))
    raw_laps = raw[raw_mask]
    context = context_from_laps(bundle, laps[mask], raw_laps)
    before = raw_laps[raw_laps["LapNumber"] < context.start_lap]
    race_compounds = sorted(set(before["Compound"]) & set(DRY_COMPOUNDS)
                            | {context.start_compound})
    return (context, actual_stops(raw_laps, context),
            best_strategies(bundle, context, race_compounds))


# The example is the green-flag race winner whose one-stop search is widest,
# so the shape of the objective is visible.
winners = green[(green["actual_stops"] == 1) & (green["recommended_stops"] == 1)
                & (green["FinishPosition"] == 1)]
searches = {index: search(row) for index, row in winners.iterrows()}
chosen = max(searches, key=lambda i: (searches[i][2]["n_stops"] == 1).sum())
example = winners.loc[chosen]
context, team, ranking = searches[chosen]
best = ranking.iloc[0]

first = context.start_lap + MIN_STINT_LAPS - 1
last = context.total_laps - MIN_STINT_LAPS
pit_laps = last - first + 1
grid_laps = len(range(first, last + 1, TWO_STOP_STEP))
two_stop_pairs = sum(1 for a in range(first, last + 1, TWO_STOP_STEP)
                     for b in range(first, last + 1, TWO_STOP_STEP)
                     if b - a >= MIN_STINT_LAPS)

scale = context.baseline_time / 100.0


def noise_sd(stops) -> float:
    """SD of simulated race time from stint and lap error (seconds)."""
    edges = [context.start_lap - 1] + [lap for lap, _ in sorted(stops)] + [context.total_laps]
    sizes = np.diff(edges).astype(float)
    return scale * math.sqrt((sizes ** 2).sum() * variance["stint_sd_pct"]
                             + sizes.sum() * variance["lap_sd_pct"])


draws_best = simulate_race_times(bundle, context, best["stops"], 4000, seed=1)
draws_team = simulate_race_times(bundle, context, team, 4000, seed=2)
team_time = float(evaluate_strategies(bundle, context, [team])["time_s"].iloc[0])
gap = team_time - float(best["time_s"])
sd_gap = math.sqrt(noise_sd(best["stops"]) ** 2 + noise_sd(team) ** 2)
numbers["example"] = {
    "event": f"{example['Year']} {example['EventName']}", "driver": example["Driver"],
    "start_lap": context.start_lap, "total_laps": context.total_laps,
    "n_laps": context.n_laps, "baseline_s": context.baseline_time,
    "pit_loss_s": context.pit_loss_s,
    "start_compound": context.start_compound,
    "team": example["actual_strategy"], "recommended": best["label"],
    "team_pit_lap": int(team[0][0]), "recommended_pit_lap": int(best["stops"][0][0]),
    "one_stop_candidates": 3 * pit_laps, "two_stop_candidates": 9 * two_stop_pairs,
    "valid_strategies": int(len(ranking)),
    "valid_one_stop": int((ranking["n_stops"] == 1).sum()),
    "valid_two_stop": int((ranking["n_stops"] == 2).sum()),
    "within_2s_of_best": int((ranking["gap_to_best_s"] <= 2.0).sum()),
    "within_5s_of_best": int((ranking["gap_to_best_s"] <= 5.0).sum()),
    "gap_team_minus_best_s": gap,
    "noise_sd_best_s": noise_sd(best["stops"]),
    "monte_carlo_sd_best_s": float(draws_best.std()),
    "sd_gap_s": sd_gap,
    "prob_best_faster_normal": 0.5 * (1 + math.erf(gap / sd_gap / math.sqrt(2))),
    "prob_best_faster_monte_carlo": float((draws_best < draws_team).mean()),
}

one_stop = ranking[ranking["n_stops"] == 1].copy()
one_stop["pit_lap"] = one_stop["stops"].map(lambda s: s[0][0])
one_stop["compound"] = one_stop["stops"].map(lambda s: s[0][1])

fig, ax = plt.subplots(figsize=(6.6, 3.9))
band = noise_sd(best["stops"])
ax.axhspan(0, band, color="#F2F3F5", zorder=0)
ax.text(one_stop["pit_lap"].max(), band * 0.95,
        rf"one standard deviation of $\hat{{R}}(s)$: {band:.0f} s",
        color=MUTED, fontsize=11, va="top", ha="right")
for compound, group in one_stop.groupby("compound"):
    group = group.sort_values("pit_lap")
    ax.plot(group["pit_lap"], group["gap_to_best_s"], color=COMPOUND_LINE[compound],
            linewidth=2.4, label=f"to {compound.title()}")
ax.axvline(team[0][0], color=INK, linewidth=1.2, linestyle="--")
ax.text(team[0][0] + 0.4, band * 0.62, "team's stop", color=INK, fontsize=11)
ax.set_ylim(0, band * 1.12)
ax.set_xlabel("Pit lap  $p$  (one-stop strategies)")
ax.set_ylabel("Seconds slower than the best")
ax.legend(loc=(0.72, 0.42))
fig.savefig(ASSETS / "fig_objective.png")
plt.close(fig)

(ASSETS / "numbers.json").write_text(json.dumps(numbers, indent=2, default=float))
print(json.dumps(numbers, indent=2, default=float))
