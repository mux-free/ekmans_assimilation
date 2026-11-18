# Ekman ensemble model

A small 1D Ekman model for parameter-ensemble and assimilation experiments.
Requires Python 3.10+, NumPy, xarray; pytest for tests. For NetCDF output, install a supported backend (SciPy is included in requirements).

## Overview

- `Ekman`: one numerical model; uses forward Euler.
- `EkmanConfig`: vertical grid, Coriolis parameter, time-step and spin-up settings.
- `ModelState`: full velocity profiles and current `ug`, `vg`, `K`.
- `StateUpdater.update_step(model)`: replace physical parameters, replace wind profiles, or apply Gaussian increments.
- `Simulation`: one trajectory, scheduling, checkpoints and immediate updates.
- `EkmanEns`: manages multiple independent `Simulation` objects, provides ensemble state vectors.
- Simple xarray-based RMSE, bias, MAE and vector RMSE functions.

## Initialization and scheduling

```python
from ekman import EkmanConfig, EkmanEns, StateUpdater

config = EkmanConfig(nz=100, lz=500, output_interval=60)
ens = EkmanEns(ug=[8, 10, 12], vg=0, K=[1, 2, 4], config=config)
ens.spin_up()  # each member spins up independently; forecast time resets to t0=0

ens.schedule(3600, StateUpdater(du=1.0, h_obs=100, sigma=25))
ds = ens.run(until=7200)

print(ds.u.dims)             # ('ensemble', 'time', 'z')
print(ds.ensemble.values)    # [1, 2, 3]
print(ens.current().u.sel(z=100))
```

Scalar `ug`, `vg`, or `K` inputs are broadcast to all members. Arrays are paired by position, **not** combined by Cartesian product. The ensemble dimension is numbered `1..N`.

## EnKF state vector interface

```python
ens.run(until=3600)
variables = ("u", "v", "K")
Xb = ens.state_vector(variables)  # shape (N, 2*(nz-1)+1)
Xa = enkf._analysis_step(Xb, obs_vect, rng_key)  # (N, num_states, state_dim)
ens.set_state_vector(Xa, variables)   # accepts both 2D and 3D posterior
```

The `state_vector()` layout follows the exact order of `variables`. `u` and `v` contain **only interior grid levels** (`z[1:-1]`), since both boundary values are fixed. `ug`, `vg`, and `K` each contribute a single scalar. Default selection: `("u", "v", "ug", "vg", "K")`. For `nz=100`, this contains `201` values per member. For `("u", "v", "K")`, there are `199` values per member.

Your EnKF must have its `num_states * state_dim` equal to the selected state-vector length. Its observation operator matrix and localization must use the same variable ordering; spatial localization cannot be applied to scalar parameters in the same way as wind grid points without explicitly designing it. EnKF may produce negative `K`; the setter checks this and rejects the entire update before changing member states. Consider estimating `log(K)` later.

## Immediate updates and background/analysis snapshots

```python
ens.run(until=3600)
ens.update_step(StateUpdater(K=5))  # all members, applied immediately

# Member 1 background and analysis at t=3600
prior = ens.simulations[0].analyses[3600]["background"]
post = ens.simulations[0].analyses[3600]["analysis"]
```

`update_step` accepts either one `StateUpdater` applied to all members, or a list of exactly one per member. `schedule(time, ...)` works the same way for future updates. New `u`/`v` profiles may be full (`nz+1`) or interior only (`nz-1`); prescribed boundary values are enforced after updates.

## Native time without interpolation

`Simulation` integrates to requested output and update times and stores the resulting **actual physical times**. `EkmanEns.dataset()` combines members only when their saved time values are identical, keeping a standard 1D `time` coordinate. Because all members normally share `output_interval`, and `EkmanEns.schedule` schedules events for all members at the same physical time, no interpolation is needed even when members have different stable numerical steps. If individual member schedules produce different saved times, `dataset()` raises rather than silently interpolating or padding.

`time=0` represents the **end of spin-up**. Recorded output times are not the internal integration-step times; `timestep` varies by member and is stored as an `(ensemble, time)` coordinate.

## Run

```bash
pip install -r requirements.txt
python example.py
pytest -q
```

## Synthetic-observation EnKF twin experiment

`python twin_enkf.py` runs a truth model (`ug=10 m/s`, `vg=0`, `K=2 m²/s`),
creates one noisy **u** observation at `z=100 m` every hour for six cycles,
then assimilates these into a 24-member parameter ensemble.
The observation error has standard deviation 0.5 m/s, so `R=[[0.25]]`.
The simulation continues one hour beyond the sixth assimilation.

- `enkf.py`: a small self-contained adaptation of the supplied JAX stochastic
  EnKF analysis step. It sets `num_states=1`, `state_dim=D` to match the
  flattened Ekman state; no localization is used.
- `twin_enkf.py`: experiment script showing the truth run, observation matrix,
  repeated EnKF calls and applying posterior states via `set_state_vector()`.
- Selected state variables are `(u, v, ug, K)`; fixed boundary levels are
  excluded, and `vg=0` is held fixed. The observation operator sees only the
  `u(z=100 m)` component.
- `time=0` remains the end of spin-up. All ensemble members share the
  unmodified 1D output-time coordinate.

**Scientific limitation:** A fully spun-up, fixed-parameter truth is almost
steady. Six noisy observations of only `u(100 m)` generally cannot uniquely
identify both `ug` and `K`. This is a minimal end-to-end demonstration, not a
parameter-recovery benchmark. Consider additionally observing `v` and several
heights, or introducing a common transient forcing in a subsequent experiment.
