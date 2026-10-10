# F1 2026 Mode-1 Dataset — Reference Aerodynamic Model

## Purpose

The reference model is a data-grounded interpolation layer used to provide a reference aerodynamic response to the simulator. It does not reconstruct the hidden equations of the synthetic physics engine and must not be interpreted as independent real-world data.

## Dataset

- Observations: 100,000
- Continuous inputs: 6
- Continuous outputs: 4
- Aero mode: fixed to Mode 1

## Inputs

- `velocity_ms`
- `flap_front_angle`
- `flap_rear_angle`
- `ride_height_front`
- `ride_height_rear`
- `battery_soc`

## Outputs

- `cl_total`
- `cd_total`
- `cop_shift`
- `energy_drain_kw`

## Reference interpolation method

- Standardization: `StandardScaler`
- Neighbours: `130`
- Metric: Euclidean distance in standardized input space
- Local fit: tricube-weighted polynomial of degree `2`
- Exact dataset points return their exact stored response
- Queries outside the observed feature envelope are rejected

The purpose is to provide a smooth, local, data-grounded reference response for arbitrary simulator states within the observed operating domain.

## Leave-one-out local interpolation diagnostic

| target | r2 | rmse | mae | max_absolute_error | n_validation |
| --- | --- | --- | --- | --- | --- |
| cl_total | 0.99967027 | 0.01144778 | 0.00912331 | 0.05093576 | 5000.00000000 |
| cd_total | 1.00000000 | 0.00000000 | 0.00000000 | 0.00000000 | 5000.00000000 |
| cop_shift | 0.99995943 | 0.00140649 | 0.00047112 | 0.02935648 | 5000.00000000 |
| energy_drain_kw | 0.99998274 | 0.60646591 | 0.47485145 | 3.75923866 | 5000.00000000 |

The leave-one-out diagnostic excludes each observation itself from its local neighbourhood. It therefore measures how well the response can be reconstructed from nearby observations, rather than trivially returning the observation itself.

## Nearest-neighbour distance statistics

| statistic | distance |
| --- | --- |
| min | 0.06952778 |
| 1% | 0.18479841 |
| 5% | 0.24293744 |
| 25% | 0.32256768 |
| 50% | 0.37691841 |
| 75% | 0.42780038 |
| 95% | 0.50261294 |
| 99% | 0.60666815 |
| max | 1.14057239 |
| mean | 0.37691922 |
| std | 0.08786258 |

## Reference prediction speed

| n_predictions | elapsed_seconds | predictions_per_second | milliseconds_per_prediction |
| --- | --- | --- | --- |
| 10000.000000 | 4.973437 | 2010.682070 | 0.497344 |

## Role in the simulator

The later simulator will use two aerodynamic response paths:

1. **Reference path:** local interpolation from the complete dataset.
2. **ML surrogate path:** the saved Phase-5 regression models.

Both paths receive the same continuous operating state. Their predicted Cd/Cl values are then passed through the same aerodynamic force equations and vehicle-dynamics equations. This isolates the effect of replacing the data-grounded response with the learned ML surrogate.

## Important limitation

Because the source dataset is synthetic and static, the reference model is not an independent measurement of a real F1 car and should not be presented as such.

## Next phase

Connect this reference response and the Phase-5 ML surrogate to the aerodynamic force equations and construct the vehicle longitudinal-dynamics simulation.