# From Prediction to Simulation: A Machine Learning-Based Vehicle Performance Simulator

## Academic Predictive Modelling Course Project

This project develops a **continuous-to-continuous machine-learning surrogate model** and embeds the learned models inside a simplified longitudinal vehicle simulator.

The central idea is:

> **Predict → replace the response map with a learned surrogate → simulate → compare the surrogate trajectory with a data-grounded reference trajectory.**

The project uses a synthetic F1-style vehicle dataset. It is designed as a Predictive Modelling project rather than a race simulation, CFD solver, or real-vehicle validation study.

---

# 1. Project Objective

The project asks a system-level question:

> **Does strong pointwise prediction accuracy remain strong when the learned model is repeatedly used inside a vehicle simulation?**

The workflow combines:

- Data auditing
- Exploratory data analysis
- PCA
- Nonlinear dependence analysis
- Interaction analysis
- Multiple regression
- Nonlinear/tree-based regression
- Model validation
- Data-grounded reference interpolation
- Longitudinal vehicle simulation
- Interactive what-if analysis

The final modelling problem is **purely continuous**: all six predictors and all four predicted outputs are numerical.

---

# 2. Dataset

## Source dataset

**F1 2026: Active Aero & Energy Clipping — 200k**

The original dataset contains 200,000 synthetic observations generated from a physics-based system and structured sampling over a continuous operating space.

For this project, one operating configuration was selected and held fixed during preprocessing. The categorical configuration flag was then removed from the modelling data.

## Final dataset

```text
100,000 observations
10 numerical variables
6 continuous predictors
4 continuous targets
0 categorical modelling variables
```

Final dataset:

```text
data/F1_2026_Synthetic_Dataset_Mode1.csv
```

### Continuous predictors

| Variable | Meaning |
|---|---|
| `velocity_ms` | Vehicle velocity |
| `flap_front_angle` | Front flap angle |
| `flap_rear_angle` | Rear flap angle |
| `ride_height_front` | Front ride height |
| `ride_height_rear` | Rear ride height |
| `battery_soc` | Battery state of charge |

### Continuous targets

| Variable | Meaning | Role |
|---|---|---|
| `cl_total` | Total lift/load coefficient | Primary |
| `cd_total` | Total drag coefficient | Primary |
| `cop_shift` | Centre-of-pressure shift | Secondary |
| `energy_drain_kw` | Energy drain | Secondary |

---

# 3. Why the Problem Is Continuous

The categorical configuration variable is excluded from the final modelling matrix.

The predictive problem is therefore:

$$
X = [v,\alpha_f,\alpha_r,h_f,h_r,SOC]
$$

with continuous outputs:

$$
C_D = f(X), \qquad C_L = f(X)
$$

and secondary outputs:

$$
CoP = f(X), \qquad P_{drain}=f(X)
$$

This makes the machine-learning component strictly continuous-to-continuous.

---

# 4. Project Architecture

```text
                 CONTINUOUS DATASET
                         │
                         ▼
                 Data audit + EDA
                         │
                         ▼
                       PCA
                         │
                         ▼
              Nonlinear structure analysis
                         │
              ┌──────────┼───────────┐
              ▼          ▼           ▼
           Linear    Polynomial   Extra Trees
              │          │           │
              └──────────┼───────────┘
                         ▼
                 Model selection
                         │
                         ▼
              Final surrogate models
                         │
             ┌───────────┴───────────┐
             ▼                       ▼
     Data-grounded reference     ML surrogate
          interpolation             models
             │                       │
             ▼                       ▼
     Reference simulation       ML simulation
             │                       │
             └───────────┬───────────┘
                         ▼
                 Compare trajectories
                         │
                         ▼
                  Interactive GUI
```

---

# 5. Implementation Status

The full core pipeline is currently implemented as nine Python scripts.

| Phase | Script | Purpose |
|---|---|---|
| 1 | `01_data_audit.py` | Dataset integrity and statistical audit |
| 2 | `02_data_exploration.py` | EDA, distributions, correlations, response maps and derived quantities |
| 3 | `03_pca_analysis.py` | PCA of the six continuous predictors |
| 4 | `04_nonlinear_structure.py` | Pearson/Spearman dependence, mutual information, interactions, linear-vs-tree comparison and feature importance |
| 5 | `05_regression_models.py` | Candidate model selection and final test evaluation |
| 6 | `06_surrogate_validation.py` | Independent validation of saved surrogate models |
| 7 | `07_reference_model.py` | Data-grounded local reference interpolation |
| 8 | `08_vehicle_longitudinal_simulator.py` | Reference-vs-ML vehicle simulation |
| 9 | `09_interactive_simulator.py` | Streamlit interactive simulator |

No Jupyter notebooks are required.

---

# 6. Phase 1 — Data Audit

File:

```text
01_data_audit.py
```

The audit checks:

- Dataset dimensions
- Missing values
- Duplicate rows
- Data types
- Infinite values
- Numerical plausibility
- Descriptive statistics
- Correlation structure

The final dataset contains 100,000 rows and 10 numerical variables with no missing or duplicate rows.

---

# 7. Phase 2 — Data Exploration

File:

```text
02_data_exploration.py
```

The analysis focuses on relationships that matter for the final simulator rather than producing only generic plots.

Important relationships include:

$$
(\alpha_f,\alpha_r) \rightarrow C_D
$$

$$
(\alpha_f,\alpha_r) \rightarrow C_L
$$

and the effect of velocity on the derived force quantities.

The strongest first-order relationships identified include:

- Rear flap angle with `cd_total`
- Front and rear flap angles with `cl_total`
- Rear flap angle with `cop_shift`
- Velocity with `energy_drain_kw`

The analysis also shows that the six predictor variables are largely independent of one another, making the continuous feature space well suited to direct supervised modelling.

---

# 8. Phase 3 — PCA

File:

```text
03_pca_analysis.py
```

PCA is applied only to the six continuous predictors after standardisation.

The resulting six principal components explain almost equal amounts of variance. All six components are required to retain 95% of the input variance.

### Interpretation

PCA therefore does **not** provide a useful low-dimensional replacement for the six original predictors in this dataset.

This is an important result rather than a failure: the original variables carry distinct information, so the final regression models keep all six continuous inputs.

---

# 9. Phase 4 — Nonlinear Structure

File:

```text
04_nonlinear_structure.py
```

This phase investigates whether simple linear relationships are sufficient.

The analysis includes:

- Pearson correlation
- Spearman correlation
- Mutual information
- Linear regression baseline
- Extra Trees comparison
- Pairwise interaction analysis
- Feature importance
- Residual structure

The results show that tree-based models capture the response maps more accurately for several targets, especially `cl_total`, `cop_shift`, and `energy_drain_kw`.

A particularly strong finding is that `cop_shift` and `cd_total` depend heavily on rear flap angle, while energy drain is dominated by vehicle velocity.

---

# 10. Phase 5 — Regression Models

File:

```text
05_regression_models.py
```

The candidate models are:

1. Multiple Linear Regression
2. Polynomial Ridge Regression (degree 2)
3. Extra Trees Regressor
4. HistGradientBoosting Regressor

The data is divided into:

```text
70% training
15% validation
15% final test
```

The validation set is used for model selection. The final test set remains untouched until the final evaluation.

## Selected models

| Target | Selected model |
|---|---|
| `cl_total` | Extra Trees |
| `cd_total` | Multiple Linear Regression |
| `cop_shift` | Extra Trees |
| `energy_drain_kw` | Extra Trees |

## Final test performance

| Target | R² | RMSE | MAE |
|---|---:|---:|---:|
| `cl_total` | 0.999527 | 0.013744 | 0.010609 |
| `cd_total` | 1.000000 | 0.000000 | 0.000000 |
| `cop_shift` | 0.999994 | 0.000538 | 0.000342 |
| `energy_drain_kw` | 0.999873 | 1.693197 | 0.938197 |

These are test-set results on observations not used for final model fitting.

The Extra Trees models use 60 trees. Each fully grown tree is about 12 MB, so the earlier 250-tree setting produced about 3 GB per model (9 GB for three targets), which does not fit in 16 GB of RAM alongside the simulator. 60 trees gives 0.7 GB per model at a cost of roughly 5% in `cl_total` RMSE.

---

# 11. Phase 6 — Surrogate Validation

File:

```text
06_surrogate_validation.py
```

This phase loads the saved models instead of retraining them and reproduces the untouched Phase-5 test evaluation.

The saved models are:

```text
models/cd_model.pkl
models/cl_model.pkl
models/cop_shift_model.pkl
models/energy_drain_kw_model.pkl
```

The validation also converts `Cd` and `Cl` into force quantities using:

$$
F_D=\frac{1}{2}\rho C_DAv^2
$$

$$
F_L=\frac{1}{2}\rho C_LAv^2
$$

with:

```text
rho = 1.225 kg/m^3
A   = 1.5 m^2
```

The derived drag-force prediction is exact to numerical precision for the selected model, while the downforce prediction remains very close to the reference values.

---

# 12. Phase 7 — Data-Grounded Reference Model

File:

```text
07_reference_model.py
```

The source dataset is static. It does not provide a measured time trajectory or directly expose the hidden equations of the synthetic generator.

Therefore, the project uses a **data-grounded local reference interpolation** rather than claiming access to the hidden physics equations.

The reference model uses:

```text
StandardScaler
130 nearest neighbours
Euclidean distance in standardised input space
Tricube-weighted local quadratic fit, centred on the query
```

The model class lives in `reference_model.py` and is imported by Phases 7, 8 and 9.

Exact dataset points return their stored response, while queries outside the observed input envelope are rejected.

The resulting leave-one-out local reconstruction diagnostics are:

| Target | R² | RMSE | MAE |
|---|---:|---:|---:|
| `cl_total` | 0.999670 | 0.011448 | 0.009123 |
| `cd_total` | 1.000000 | 0.000000 | 0.000000 |
| `cop_shift` | 0.999959 | 0.001406 | 0.000471 |
| `energy_drain_kw` | 0.999983 | 0.606466 | 0.474851 |

An earlier version used an inverse-distance-squared average of 12 neighbours. That average is biased wherever the response has a slope, and its leave-one-out RMSE (`cl_total` 0.056, `cd_total` 0.0090) was several times worse than the ML surrogate's test RMSE. The simulator comparison was then dominated by the reference's own error. The local quadratic fit reproduces the linear `cd_total` response exactly and is as accurate as the surrogate for `cl_total`.

Saved model:

```text
models/reference_aero_model.pkl
```

The reference model is a benchmark for the simulator. It is **not** independent real-world telemetry.

---

# 13. Phase 8 — Longitudinal Vehicle Simulator

File:

```text
08_vehicle_longitudinal_simulator.py
```

The simulator feeds the same operating state into two response paths:

```text
                 Same continuous state
                    /            \
                   /              \
          Reference model      ML surrogate
                   |              |
                   ▼              ▼
                 Cd, Cl         Cd, Cl
                   |              |
                   └──────┬───────┘
                          ▼
                 Same vehicle dynamics
                          ▼
                       trajectory
```

## Simulation assumptions

These are explicit demonstration assumptions, not measured F1 vehicle parameters.

| Parameter | Value |
|---|---:|
| Air density | 1.225 kg/m³ |
| Frontal area | 1.5 m² |
| Vehicle mass | 800 kg |
| Rolling resistance coefficient | 0.015 |
| Gravity | 9.81 m/s² |
| Time step | 0.05 s |
| Duration | 15 s |
| Initial velocity | 80 m/s |

The default demonstration changes the rear flap during the run:

```text
0–5 s   : 5°
5–10 s  : 20°
10–15 s : 5°
```

Both response paths receive exactly the same control schedule.

## Vehicle model

Drag:

$$
F_D=\frac12\rho C_DAv^2
$$

Downforce/load:

$$
F_L=\frac12\rho C_LAv^2
$$

Rolling resistance:

$$
F_{RR}=C_{rr}(mg+F_L)
$$

Coast-down acceleration:

$$
a=-\frac{F_D+F_{RR}}{m}
$$

Velocity is integrated with an explicit Euler step and distance with the trapezoidal rule. If the speed falls below the lowest velocity in the dataset (27.7 m/s), the run stops cleanly instead of leaving the observed envelope.

A coast-down formulation is used because the dataset does not contain throttle, engine torque, tyre, steering, or track-geometry trajectories.

## Simulation result

For the current 15-second demonstration:

| Quantity | Reference | ML surrogate | Difference |
|---|---:|---:|---:|
| Final speed | 41.1229 m/s | 41.1307 m/s | +0.0079 m/s |
| Final distance | 853.1946 m | 853.2905 m | +0.0959 m |

Trajectory-level errors:

| Quantity | RMSE | MAE |
|---|---:|---:|
| Speed | 0.0068 m/s | 0.0064 m/s |
| Distance | 0.0502 m | 0.0399 m |
| Acceleration | 0.0008 m/s² | 0.0006 m/s² |
| `cl_total` | 0.01761 | 0.01706 |
| `cd_total` | 0.00000 | 0.00000 |
| Drag force | 0.43 N | 0.39 N |
| Downforce | 61.22 N | 53.83 N |

This is the main end-to-end result of the project.

## Surrogate difference versus reference uncertainty

The simulator also reports how the surrogate-reference difference compares with the reference's own leave-one-out error:

| Target | Closed-loop RMSE | Open-loop RMSE | Reference leave-one-out RMSE |
|---|---:|---:|---:|
| `cl_total` | 0.017614 | 0.017607 | 0.011448 |
| `cd_total` | 0.000000 | 0.000000 | 0.000000 |
| `cop_shift` | 0.000369 | 0.000369 | 0.001406 |
| `energy_drain_kw` | 1.164587 | 1.154836 | 0.606466 |

Open-loop means the surrogate is evaluated on exactly the states the reference path visited, which removes state divergence between the two runs. A difference at or below the reference's own error cannot be attributed to the surrogate.

---

# 14. Phase 9 — Interactive Simulator

File:

```text
09_interactive_simulator.py
```

The final interface is built with Streamlit.

Run it with:

```bash
streamlit run 09_interactive_simulator.py
```

On Windows, `run_simulator.bat` starts the app with the project environment at `%USERPROFILE%\.venvs\f1sim`.

The app does not carry its own copy of the physics. It imports the vehicle parameters, force equations, model loading and simulation loop from `08_vehicle_longitudinal_simulator.py`, so the interactive runs and the scripted Phase-8 run always use the same engine.

The interface provides continuous controls for:

- Velocity
- Front flap angle
- Rear flap angle
- Front ride height
- Rear ride height
- Battery SOC

It displays the corresponding predicted response and derived quantities, then allows a simulation to be run explicitly.

The interface includes:

- Live `Cd` / `Cl` values
- Drag and downforce
- Energy drain
- Reference-vs-ML trajectory plots
- Speed and distance comparison
- Coefficient comparison
- Error metrics
- What-if active-control scenarios
- Downloadable simulation results

If a run slows below the lowest velocity in the dataset (27.7 m/s) before the requested duration, it stops there and the app shows a warning saying why.

The F1-style visual in the interface is generated directly in the app; no external image assets are required.

---

# 15. Project Results in One View

### Pointwise ML prediction

The final regression models achieve extremely high test-set accuracy:

```text
Cl       R² = 0.999527
Cd       R² = 1.000000
CoP      R² = 0.999994
Energy   R² = 0.999873
```

### Reference reconstruction

The local data-grounded interpolation can reconstruct nearby responses with:

```text
Cl       R² = 0.999670
Cd       R² = 1.000000
CoP      R² = 0.999959
Energy   R² = 0.999983
```

### System-level simulation

The ML surrogate produces a very similar vehicle trajectory to the reference simulation:

```text
Speed RMSE     = 0.0068 m/s
Distance RMSE  = 0.0502 m
```

The key scientific question is therefore evaluated at **two levels**:

```text
Pointwise prediction accuracy
          ↓
System-level simulation accuracy
```

---

# 16. What the Project Demonstrates

The project demonstrates how a continuous predictive model can be transformed into a usable simulation component.

Instead of stopping at:

> “The model has a high R².”

we ask:

> **“What happens when the model is used repeatedly inside a dynamic system?”**

This creates a stronger connection between predictive modelling and numerical simulation.

---

# 17. Scientific Limitations

The following limitations should be stated clearly in the report and presentation.

### Synthetic data

The source dataset is synthetic. The project therefore does not validate the model against measured real-world vehicle telemetry.

### Static dataset

The original data is tabular rather than a recorded time series. The temporal trajectory is generated by the vehicle-dynamics simulator.

### Simplified vehicle dynamics

The simulator does not model the complete vehicle system. It omits detailed powertrain, tyres, steering, suspension dynamics, track geometry, and race strategy.

### Reference model

The reference path is a local interpolation through the available dataset. It does not recover or independently verify the hidden equations used to generate the synthetic dataset.

### Operating envelope

The reference interpolator rejects states outside the observed feature envelope. The simulator is therefore intended to operate within the supported continuous domain.

---

# 18. Why This Is Not a Full F1 Simulator

This project is deliberately **not**:

- A race simulator
- A CFD solver
- A game engine
- A real-vehicle digital twin
- A time-series forecasting system
- A classification problem
- A reinforcement-learning system

The actual focus is:

> **Continuous predictive modelling + surrogate modelling + longitudinal vehicle simulation.**

---

# 19. Recommended Presentation Story

The presentation should be organised around the modelling question rather than around a list of algorithms.

### Slide 1 — Title

**From Prediction to Simulation: A Machine Learning-Based Vehicle Performance Simulator**

### Slide 2 — The Question

Can a highly accurate predictive model also reproduce system-level simulation behaviour?

### Slide 3 — Dataset

```text
100,000 observations
6 continuous inputs
4 continuous outputs
```

### Slide 4 — Understanding the Input Space

Show correlation structure and response maps.

### Slide 5 — PCA

Show why the six-dimensional input structure is not well represented by a small number of PCs.

### Slide 6 — Linear vs Nonlinear Models

Show the improvement from multiple linear regression to nonlinear/tree-based models.

### Slide 7 — Final Surrogate

```text
6 continuous inputs
        ↓
      ML models
        ↓
4 continuous outputs
```

### Slide 8 — Reference vs ML

Explain the data-grounded reference interpolation.

### Slide 9 — Simulator

Show the feedback loop:

```text
State → prediction → forces → dynamics → new state
```

### Slide 10 — Main Result

Reference trajectory vs ML-surrogate trajectory.

### Slide 11 — Does Pointwise Accuracy Survive Simulation?

Compare regression metrics with trajectory-level metrics.

### Slide 12 — What-If Experiment

Example:

> What happens when the rear flap angle is changed during the run?

### Slide 13 — Limitations

Clearly state synthetic data and simplified dynamics.

### Slide 14 — Conclusion

The main conclusion should be:

> **A continuous machine-learning surrogate with very high pointwise accuracy can be embedded into a vehicle-dynamics simulator and evaluated at the system level against a data-grounded reference response.**

---

# 20. Repository Structure

```text
PM_prj/
│
├── data/
│   └── F1_2026_Synthetic_Dataset_Mode1.csv
│
├── models/
│   ├── cd_model.pkl
│   ├── cl_model.pkl
│   ├── cop_shift_model.pkl
│   ├── energy_drain_kw_model.pkl
│   └── reference_aero_model.pkl
│
├── reports/
│   ├── 01_data_audit/
│   ├── 02_exploratory_analysis/
│   ├── 03_pca_analysis/
│   ├── 04_nonlinear_structure/
│   ├── 05_regression_models/
│   ├── 06_surrogate_validation/
│   ├── 07_reference_model/
│   └── 08_vehicle_longitudinal_simulator/
│
├── 01_data_audit.py
├── 02_data_exploration.py
├── 03_pca_analysis.py
├── 04_nonlinear_structure.py
├── 05_regression_models.py
├── 06_surrogate_validation.py
├── 07_reference_model.py
├── 08_vehicle_longitudinal_simulator.py
├── 09_interactive_simulator.py
├── reference_model.py
└── README.md
```

---

# 21. Installation

Recommended environment:

```text
Python 3.10+
```

Install the main dependencies:

```bash
pip install numpy pandas scipy scikit-learn matplotlib seaborn plotly joblib streamlit
```

On Windows, create the virtual environment in a short path (for example `%USERPROFILE%\.venvs\f1sim`). Streamlit ships files with long names, and a virtual environment inside a deeply nested project folder can exceed the 260-character path limit and leave Streamlit half-installed.

Run the analysis scripts in order when reproducing the project from scratch. The simulator itself needs only:

```bash
python 05_regression_models.py
python 07_reference_model.py
python 08_vehicle_longitudinal_simulator.py
streamlit run 09_interactive_simulator.py
```

The model files are not shipped with the repository; Phases 5 and 7 generate them (about 2.1 GB in total).

---

# 22. Reproducibility

The project is intentionally organised as independent Python scripts so that every stage can be run, inspected, and validated separately.

The saved models decouple model training from simulation. Once the files in `models/` exist, the later simulator stages load them rather than retraining them.

---

# 23. Final Project Statement

> **We develop a continuous machine-learning surrogate for a synthetic F1-style vehicle performance system, embed the learned response models into a longitudinal vehicle simulator, and compare the resulting trajectory against a data-grounded reference interpolation to determine whether strong pointwise predictive accuracy translates into strong system-level simulation accuracy.**

That is the central idea of the project.
