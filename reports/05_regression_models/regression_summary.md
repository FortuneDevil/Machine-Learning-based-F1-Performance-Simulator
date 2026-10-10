# F1 2026 Mode-1 Dataset — Final Regression Analysis

## Objective

This phase selects continuous-to-continuous regression models for the four continuous aerodynamic outputs. Cd and Cl are treated as the primary simulator targets; CoP shift and energy drain are secondary outputs.

## Inputs

- `velocity_ms`
- `flap_front_angle`
- `flap_rear_angle`
- `ride_height_front`
- `ride_height_rear`
- `battery_soc`

## Targets

- `cl_total`
- `cd_total`
- `cop_shift`
- `energy_drain_kw`

## Model candidates

- Multiple Linear Regression
- Polynomial Ridge Regression (degree 2)
- Extra Trees Regressor
- HistGradientBoosting Regressor

## Data split

- 70% training
- 15% validation
- 15% final test

The validation set was used for model selection. The final test set was reserved for the final performance estimate.

## Validation results

| target | model | r2 | rmse | mae | n_train | n_validation |
| --- | --- | --- | --- | --- | --- | --- |
| cd_total | Multiple Linear Regression | 1.00000000 | 0.00000000 | 0.00000000 | 69999.00000000 | 15000.00000000 |
| cd_total | Polynomial Ridge Regression | 1.00000000 | 0.00000000 | 0.00000000 | 69999.00000000 | 15000.00000000 |
| cd_total | Extra Trees | 0.99998979 | 0.00034054 | 0.00020135 | 69999.00000000 | 15000.00000000 |
| cd_total | HistGradientBoosting | 0.99955730 | 0.00224192 | 0.00116637 | 69999.00000000 | 15000.00000000 |
| cl_total | Extra Trees | 0.99948953 | 0.01410763 | 0.01084280 | 69999.00000000 | 15000.00000000 |
| cl_total | HistGradientBoosting | 0.99908150 | 0.01892377 | 0.01394650 | 69999.00000000 | 15000.00000000 |
| cl_total | Polynomial Ridge Regression | 0.99894115 | 0.02031820 | 0.01643249 | 69999.00000000 | 15000.00000000 |
| cl_total | Multiple Linear Regression | 0.98886702 | 0.06588319 | 0.05474576 | 69999.00000000 | 15000.00000000 |
| cop_shift | Extra Trees | 0.99999285 | 0.00059600 | 0.00038112 | 69999.00000000 | 15000.00000000 |
| cop_shift | HistGradientBoosting | 0.99983423 | 0.00286939 | 0.00199603 | 69999.00000000 | 15000.00000000 |
| cop_shift | Polynomial Ridge Regression | 0.98764248 | 0.02477426 | 0.01574194 | 69999.00000000 | 15000.00000000 |
| cop_shift | Multiple Linear Regression | 0.92515785 | 0.06096889 | 0.04384132 | 69999.00000000 | 15000.00000000 |
| energy_drain_kw | Extra Trees | 0.99987788 | 1.64567921 | 0.97171970 | 69999.00000000 | 15000.00000000 |
| energy_drain_kw | HistGradientBoosting | 0.99977920 | 2.21279745 | 1.45914695 | 69999.00000000 | 15000.00000000 |
| energy_drain_kw | Polynomial Ridge Regression | 0.99700897 | 8.14432445 | 6.12460138 | 69999.00000000 | 15000.00000000 |
| energy_drain_kw | Multiple Linear Regression | 0.90086488 | 46.88762447 | 37.76148000 | 69999.00000000 | 15000.00000000 |

## Selected models

| target | model | r2 | rmse | mae | n_train | n_validation |
| --- | --- | --- | --- | --- | --- | --- |
| cl_total | Extra Trees | 0.99948953 | 0.01410763 | 0.01084280 | 69999.00000000 | 15000.00000000 |
| cd_total | Multiple Linear Regression | 1.00000000 | 0.00000000 | 0.00000000 | 69999.00000000 | 15000.00000000 |
| cop_shift | Extra Trees | 0.99999285 | 0.00059600 | 0.00038112 | 69999.00000000 | 15000.00000000 |
| energy_drain_kw | Extra Trees | 0.99987788 | 1.64567921 | 0.97171970 | 69999.00000000 | 15000.00000000 |

## Final untouched-test results

| target | selected_model | r2 | rmse | mae | n_train_plus_validation | n_test |
| --- | --- | --- | --- | --- | --- | --- |
| cl_total | Extra Trees | 0.99952736 | 0.01374444 | 0.01060864 | 84999.00000000 | 15001.00000000 |
| cd_total | Multiple Linear Regression | 1.00000000 | 0.00000000 | 0.00000000 | 84999.00000000 | 15001.00000000 |
| cop_shift | Extra Trees | 0.99999422 | 0.00053818 | 0.00034185 | 84999.00000000 | 15001.00000000 |
| energy_drain_kw | Extra Trees | 0.99987259 | 1.69319705 | 0.93819727 | 84999.00000000 | 15001.00000000 |

## Tree-model feature importance

### `cl_total`
| feature | importance |
| --- | --- |
| flap_front_angle | 0.43760426 |
| flap_rear_angle | 0.41987357 |
| ride_height_front | 0.14174315 |
| battery_soc | 0.00026175 |
| ride_height_rear | 0.00025936 |
| velocity_ms | 0.00025791 |

### `cop_shift`
| feature | importance |
| --- | --- |
| flap_rear_angle | 0.72715639 |
| flap_front_angle | 0.27280062 |
| battery_soc | 0.00001096 |
| ride_height_front | 0.00001076 |
| ride_height_rear | 0.00001067 |
| velocity_ms | 0.00001060 |

### `energy_drain_kw`
| feature | importance |
| --- | --- |
| velocity_ms | 0.91520363 |
| flap_rear_angle | 0.05988536 |
| flap_front_angle | 0.02463075 |
| ride_height_rear | 0.00009382 |
| battery_soc | 0.00009350 |
| ride_height_front | 0.00009294 |

## Interpretation

Model selection is based on predictive performance but should also consider simulator behaviour. The final surrogate should not be chosen solely because it has the highest R² if a simpler model produces nearly identical simulation behaviour.

The primary simulator will use the final Cd and Cl models to calculate aerodynamic drag and downforce. The CoP and energy models can be incorporated into later extensions of the simulator.

## Next phase

Build the aerodynamic surrogate interface and then construct the reference-vs-ML simulator. The simulator will use the saved Cd/Cl models and the same continuous inputs used during training.