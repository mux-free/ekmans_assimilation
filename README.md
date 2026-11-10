# Minimal restartable Ekman model

Purpose: a readable numerical experiment for studying how changes in eddy diffusivity, geostrophic wind, and localized wind increments affect subsequent forecasts.

## TEST
pytest -q

## Design

- `Ekman` calculates tendencies and integrates the model with forward Euler.
- `ModelState` holds the current wind arrays, parameters, physical time and timestep.
- `StateUpdater` applies optional changes to `ug`, `vg`, `K`, and Gaussian wind increments (`du`, `dv`).
- `Simulation` schedules updates, records full profiles at `output_interval`, and provides `.copy()` for independent experiments.
- `rmse`, `bias`, `mae`, `vector_rmse` are simple xarray-aware functions.

Example usage is in `example.py`. `Simulation.run(until)` returns `xr.Dataset`, and `sim.analyses[time]` has both the pre-update `background` and post-update `analysis` snapshots. `u_g`, `v_g`, `K` are scalar coordinates when constant; if any changes, it becomes a time-dependent coordinate.

## Assumptions

- Constant, positive eddy viscosity `K` throughout the column; no turbulence closure yet.
- Fixed surface winds (0, 0); upper winds constrained to (ug, vg).
- Explicit forward Euler integration; `dt = min(0.45 * dz**2 / K, dt_max)` with shortened steps at exact output, update, and final times. Changing `K` recomputes the stable step size automatically.
- `spin_up()` is separate from `Simulation` and resets the clock to zero afterward.
- The `time=0` recorded state is *after* an optional initial kick.
- An update at an output time replaces the recorded prior with the analysis at that time; the prior is still available in `sim.analyses[time]["background"]`.
- Metrics require matching coordinates; interpolate observations to the model grid explicitly if needed.
- The current implementation records snapshots in memory and is intended for relatively small research experiments.

## Next extensions

Add real or synthetic observations, simple parameter estimation and more flexible wind increments. 
