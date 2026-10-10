# F1 2026 Mode-1 Dataset — Vehicle Longitudinal Simulator

## Purpose

Phase 8 connects the Phase-7 data-grounded reference aerodynamic response
and the Phase-5 ML surrogate to the same longitudinal vehicle-dynamics model.
The simulator uses identical controls, equations, and initial conditions for
both paths.

The reference path is a local interpolation through the complete synthetic
dataset. It is not independent real-world telemetry and it does not reconstruct
the hidden physics engine equations.

## Dataset

- Dataset: `F1_2026_Synthetic_Dataset_Mode1.csv`
- Aero mode: fixed to Mode 1
- Continuous inputs: 6
- Continuous outputs: 4
- No categorical modelling variables are introduced.

## Simulation assumptions

| parameter | value |
| --- | ---: |
| air_density_kg_m3 | 1.225 |
| frontal_area_m2 | 1.500 |
| vehicle_mass_kg | 800.0 |
| rolling_resistance_coefficient | 0.0150 |
| gravity_m_s2 | 9.81 |
| time_step_s | 0.050 |
| nominal_duration_s | 15.00 |
| initial_velocity_ms | 80.00 |
| initial_front_flap_deg | 15.00 |
| initial_rear_flap_deg | 5.00 |
| initial_front_ride_height | 30.00 |
| initial_rear_ride_height | 52.50 |
| initial_battery_soc | 50.00 |

These values are explicit demonstration assumptions, not measured F1 vehicle
parameters.

## Active-aero control schedule

The rear flap is deliberately changed during the experiment to make the
simulation dynamic:

- 0–5 s: rear flap = 5 degrees
- 5–10 s: rear flap = 20 degrees
- 10–15 s: rear flap = 5 degrees

Both reference and ML paths receive exactly this same schedule.

## Longitudinal model

Aerodynamic drag:

`F_D = 0.5 * rho * A * Cd * v^2`

Downforce:

`F_L = 0.5 * rho * A * Cl * v^2`

Rolling resistance:

`F_RR = Crr * (m*g + F_L)`

Coast-down acceleration:

`a = -(F_D + F_RR) / m`

Velocity is integrated with an explicit Euler step and distance with the
trapezoidal rule.

## Simulation result

| quantity | reference final | surrogate final | difference |
| --- | ---: | ---: | ---: |
| final speed (m/s) | 41.122874 | 41.130731 | 0.007856 |
| final distance (m) | 853.194576 | 853.290460 | 0.095884 |

### Trajectory-level errors

| quantity | rmse | mae | max_absolute_error | final_reference | final_surrogate | final_difference |
| --- | --- | --- | --- | --- | --- | --- |
| Speed (m/s) | 0.00677971 | 0.00638410 | 0.00808297 | 41.12287437 | 41.13073078 | 0.00785641 |
| Distance (m) | 0.05016815 | 0.03991310 | 0.09588431 | 853.19457588 | 853.29046019 | 0.09588431 |
| Acceleration (m/s^2) | 0.00081736 | 0.00056677 | 0.00178680 | -1.18231292 | -1.18218401 | 0.00012891 |
| Lift coefficient Cl | 0.01761350 | 0.01705480 | 0.02416713 | 2.20058966 | 2.18259297 | -0.01799669 |
| Drag coefficient Cd | 0.00000000 | 0.00000000 | 0.00000000 | 0.50000000 | 0.50000000 | -0.00000000 |
| CoP shift | 0.00036903 | 0.00029804 | 0.00089334 | 0.38977001 | 0.38980255 | 0.00003254 |
| Energy drain (kW) | 1.16458734 | 0.72742967 | 5.47544430 | 31.74344037 | 32.34115526 | 0.59771489 |
| Drag force (N) | 0.42989027 | 0.38942053 | 0.65038093 | 776.84483476 | 777.14169109 | 0.29685633 |
| Downforce (N) | 61.22063033 | 53.82529932 | 95.29601517 | 3419.03342439 | 3392.36798546 | -26.66543893 |

## Key aerodynamic comparison

- Cd trajectory RMSE: **0.00000000**
- Cl trajectory RMSE: **0.01761350**
- Downforce trajectory RMSE: **61.220630 N**
- Vehicle-speed trajectory RMSE: **0.00677971 m/s**

The speed error is the most important end-to-end quantity in this phase:
it measures how much the learned aerodynamic surrogate changes the simulated
vehicle trajectory relative to the data-grounded reference path.

## How much of the difference is the reference's own error?

| target | closed_loop_rmse | open_loop_rmse | reference_loo_rmse |
| --- | --- | --- | --- |
| cl_total | 0.01761350 | 0.01760703 | 0.01144778 |
| cd_total | 0.00000000 | 0.00000000 | 0.00000000 |
| cop_shift | 0.00036903 | 0.00036903 | 0.00140649 |
| energy_drain_kw | 1.16458734 | 1.15483554 | 0.60646591 |

- `closed_loop_rmse`: difference along the two separately simulated paths.
- `open_loop_rmse`: difference when the surrogate is evaluated on exactly the
  states the reference path visited, which removes state divergence.
- `reference_loo_rmse`: the reference interpolator's own leave-one-out error
  from Phase 7.

A surrogate-reference difference at or below `reference_loo_rmse` is within
the reference's own uncertainty and should not be read as surrogate error.

## Computational timing

- Reference model load time: 0.069593 s
- Surrogate model load time: 7.935869 s
- Two-path simulation time: 24.538091 s
- Simulated reference steps: 301
- Simulated surrogate steps: 301

## Interpretation

The central experiment is not “real F1 telemetry versus simulation”. It is:

**data-grounded reference response versus learned ML surrogate response under
the same vehicle-dynamics equations.**

This isolates the modelling approximation introduced by replacing the
reference aerodynamic map with the machine-learned models.

## Important limitation

Because the source dataset is synthetic and static, this simulator is a
controlled surrogate experiment rather than a validated race-car simulation.
The next interface layer can expose the six continuous operating variables,
show live Cd/Cl/drag/downforce values, and allow the user to run alternative
what-if aero schedules.
