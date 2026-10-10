# F1 Race Strategy Simulator

A predictive-modelling project on real Formula 1 data: predict how a driver's
lap time changes through a race, then use that prediction to simulate pit
strategies and compare them with what teams really did.

## The question

> A few laps into a race, can a model predict the rest of it well enough to
> choose the pit strategy?

The answer from the backtest is split:

- **Race time: yes.** Given the strategy a driver really used, predicted race
  time is within **0.57%** of the real one (24 s over a race of about 70
  minutes), less than half the error of assuming constant pace (55 s).
- **Choosing the pit lap: only roughly.** The recommended strategy has the
  same number of stops as the team's in 65% of cases, but the recommended
  first pit lap is on average 8 laps from the team's. The predicted gain
  between good strategies (a few seconds) is smaller than the model's own
  error, and the app's uncertainty view shows this directly.

## Data

Real race laps from the [FastF1](https://docs.fastf1.dev) library: lap times,
tyre compound, tyre age, track status and weather.

| Split | Seasons | Dry races | Laps used |
|---|---|---:|---:|
| Train | 2022, 2023 | 35 | 24,039 |
| Validation | 2024 | 19 | 15,332 |
| Test | 2025 | 21 | 15,844 |

92 races were downloaded. 12 with significant running on wet-weather tyres
were dropped, and 5 more produced no usable laps after cleaning. 2022–2025
is one technical-regulation era, so the cars are comparable across the
split.

The split is **by season**. Laps from one race are strongly correlated, so a
random split would put the same race on both sides and overstate accuracy.

## Target

Absolute lap time depends mostly on the circuit and the car. What a strategy
needs is how lap time *changes*, so each lap is expressed relative to the
driver's own baseline pace:

```text
delta_pct = 100 * (lap_time / baseline_time - 1)
```

The baseline is the median of the driver's first clean laps (up to 8 laps
between laps 2 and 12). It is known early in the race, so predicting later
laps from it uses no future information.

Only clean laps are modelled: green flag, dry tyres, no pit entry or exit,
and no more than 4% slower than the driver's median pace.

## Inputs

| Feature | Meaning |
|---|---|
| `tyre_life` | Laps completed on the current tyres |
| `compound` | Soft, medium or hard |
| `laps_since_baseline_frac` | Fuel burned since the baseline (share of race distance) |
| `lap_frac` | Share of the race completed |
| `total_laps` | Race length |
| `baseline_compound`, `baseline_tyre_life` | Tyre the baseline was set on |
| `baseline_track_temp` | Track temperature when the baseline was set |
| `circuit` | Venue |

Every input is known at the moment of prediction for any hypothetical
strategy, which is what lets the model be used as a simulator.

## Models

Compared on the 2024 validation season:

| Model | RMSE (% of pace) | RMSE (s per lap) | R² |
|---|---:|---:|---:|
| Gradient boosting | 0.902 | 0.798 | 0.454 |
| Gradient boosting, monotonic tyre wear | 0.909 | 0.803 | 0.445 |
| Linear regression | 0.930 | 0.822 | 0.419 |
| Linear, wear per compound | 0.938 | 0.828 | 0.410 |
| Mean change | 1.222 | 1.084 | -0.002 |
| No change from baseline | 1.698 | 1.550 | -0.936 |

The gradient-boosting model was selected and refitted on 2022–2024. On the
untouched 2025 test season:

| Model | RMSE (% of pace) | RMSE (s per lap) | R² |
|---|---:|---:|---:|
| Gradient boosting | 0.966 | 0.835 | 0.437 |
| No change from baseline | 1.773 | 1.558 | -0.896 |

The trees are deliberately small (4 leaves). Larger trees fitted
race-specific noise and did worse on validation than linear regression.

An R² near 0.44 is what real lap data allows with these inputs: traffic,
driver management and track evolution are not observed. The validation error
splits into three nested parts:

| Level | Spread (% of pace) | Meaning |
|---|---:|---|
| Driver over a whole race | 0.545 | Mostly a mis-judged baseline; shifts all strategies equally |
| Stint within a race | 0.438 | Separates strategies |
| Single lap within a stint | 0.650 | Largely averages out over a stint |

## Simulator

`strategy.py` turns the model into a race simulator:

1. Build a lap-by-lap tyre plan for a strategy (which compound, how old).
2. Predict every lap time from the model and the driver's baseline pace.
3. Add the circuit's pit-loss time per stop (median green-flag loss at that
   circuit in 2022–2024).
4. Search every one-stop strategy and a grid of two-stop strategies, and
   rank them by predicted race time.
5. Draw Monte Carlo races using the stint-level and lap-level errors above
   to show the spread of each strategy.

Rules enforced in the search: at least two dry compounds in the race, stints
of at least 5 laps, and no stint longer than the 98th percentile of tyre age
observed on that compound at that circuit.

## Animated race

The first tab of the app plays the race. The same driver runs up to three
strategies at once (the recommended one, the team's real one, and one you
set yourself), starting together on the first simulated lap.

The view shows the real circuit outline, top-down cars whose wheel colour is
the tyre compound (red soft, yellow medium, white hard), a pit box the cars
drive into during a stop, a running-order panel with live gaps and tyre
age, and a chart of each car's lead that draws itself lap by lap. Playback
has four speeds and a slider to scrub through the race.

By default every lap is the model's prediction. With random variation
switched on, each press of "New random race" draws one possible race from
the model's measured error, so the finishing order can change.

What is real and what is not:

- The circuit outlines are real, from the open
  [f1-circuits](https://github.com/bacinger/f1-circuits) dataset
  (`fetch_circuits.py`). Without that file a generic loop is drawn.
- The start line's position on the outline and the direction of travel are
  illustrative. The lap data holds lap times, not car positions.
- Cars move at a constant speed within a lap; only their lap times differ.
- Two cars closer than a car length are drawn one car length apart so they
  do not overprint. The running-order panel always shows the true gap.

## Backtest on 2025

`04_backtest.py` runs the simulator for every top-10 finisher with usable
data in the dry 2025 races (182 driver races across 20 races).

| Result | Value |
|---|---:|
| Driver races fully under green flag | 60 |
| Race-time error for the real strategy (MAE) | 24.3 s (0.57%) |
| Race-time bias | +2.2 s |
| Race-time error with constant baseline pace (MAE) | 54.6 s |
| Same number of stops as the team | 64.8% |
| First pit lap, average distance from the team's | 8.0 laps |
| First pit lap within 3 laps of the team's | 26.4% |
| Median predicted gain of the recommended strategy | 2.2 s |

Race time can only be checked on races that stayed green, because safety
cars are not modelled.

## Limitations

- **Survivorship in tyre data.** Long stints are only seen when the tyres
  held up, so the model is optimistic about old tyres and tends to recommend
  later stops than teams make.
- **No traffic or track position.** Teams often stop early to undercut a
  rival or to rejoin in clear air. The simulator finds the fastest strategy
  in clean air only.
- **No safety cars or weather changes.** These decide many real races.
- **Compound names are not constant.** "Soft" is a different rubber at
  different circuits; the model only sees the label and the circuit.
- **The team's choice is not ground truth for the best strategy.** It is a
  reference from experts with more information, not a proven optimum.
- **Strategy gains are within model error.** Read the uncertainty view
  before trusting a recommendation.

## Run it

```bash
pip install -r requirements.txt

python 01_download_data.py     # about 30 minutes; resumable
python 02_build_dataset.py
python 03_train_models.py
python 04_backtest.py
python fetch_circuits.py       # optional: real circuit outlines
streamlit run 05_app.py
```

On Windows, `run_app.bat` starts the app with the environment at
`%USERPROFILE%\.venvs\f1sim`.

## Files

| File | Purpose |
|---|---|
| `01_download_data.py` | Download race laps with FastF1 |
| `02_build_dataset.py` | Clean laps, baselines, target, pit loss |
| `03_train_models.py` | Compare models, final test, save the model |
| `04_backtest.py` | Backtest the simulator on 2025 |
| `05_app.py` | Interactive Streamlit app |
| `strategy.py` | Simulation engine shared by the backtest and the app |
| `race_animation.py` | Race data for the animated race (first tab of the app) |
| `race_view.html` | The animated race view itself: circuit, cars, timing panel |
| `fetch_circuits.py` | Download real circuit outlines (optional) |
| `common.py` | Paths, split and baseline settings |
| `plot_style.py` | Chart colours and styling |
| `reports/` | Tables and figures written by scripts 03 and 04 |
