# Interactive Ekman assimilation

A revised, runnable project based on the supplied repository text snapshot.
The original `main(sim_config, ekman_config)` was not in that snapshot; its
replacement is `EkmanExperiment(sim_config, ekman_config)`.

## Start

```bash
python -m pip install -e '.[test,notebook]'
python -m pytest -q
jupyter lab interactive_ekman.ipynb
```

The Python API and batch example work without notebook extras:

```bash
python -m pip install -e '.[test]'
python -m src.example
```

Run commands from this directory. The default 72-hour model spin-up takes
roughly 15–20 seconds on the validation machine. A shorter spin-up changes
the physical initial condition; it is not just a performance setting.

## The small API

```python
from src import EkmanExperiment, EkmanConfig

exp = EkmanExperiment.from_yaml("configs/config.yaml", EkmanConfig(), seed=14)

# For a NEW experiment: signed prior biases, independent ensemble spreads.
exp.set_parameters(ug_truth=10, K_truth=2, ug_error=-2, K_error=1,
                   sigma_ug=1.5, spread_K=0.8,
                   save_to="configs/my_experiment.yaml")
exp.reset()

# Stop at any chosen physical time, in seconds after spin-up.
exp.run_until(3600)
obs = exp.observe(heights=[100, 200], variables=("u", "v"), std=0.25)
record = exp.assimilate(obs, variables=("ug",))

# K remains different between members, but unchanged by this analysis.
exp.run_until(5400)
obs2 = exp.observe([7.2, 8.6, 2.0, 1.2], heights=[100, 200],
                   variables=("u", "v"), std=0.5)
exp.assimilate(obs2, variables=("ug", "K"))
exp.run_until(7200)

# Optional later joint state/parameter update.
exp.assimilate(exp.observe(), variables=("ug", "K", "u", "v"))
branch = exp.copy()
datasets = exp.datasets()  # keys: truth, ensemble, control
```

| Method | Meaning |
|---|---|
| `set_parameters(...)` | Edit truth, initial biases or spreads; requires `reset()` before further work. Optional `save_to` writes YAML. |
| `set_errors(ug_error=..., K_error=...)` | Short form for changing initial signed biases. |
| `reset()` | Recreate and spin up truth and ensemble, copy the no-DA control, reset observation/filter RNG streams and analysis history. |
| `run_until(t)` / `run(t)` | Advance to absolute time `t`; pause there; later calls continue. `run()` uses configured total duration. |
| `request_stop()` / `interrupt()` | Ask an active background run to stop after a synchronized short integration chunk. Wait for `run_until` to return before assimilation. |
| `observe(...)` | Create synthetic observations or attach entered values at the current time. |
| `assimilate(obs, variables=...)` / `enkf_step(...)` | Perform one analysis; change only selected variables. |
| `copy()` | Independent checkpoint including histories and random generator states. |
| `save_config(path)` | Save current initial-condition settings to a specified YAML file. |

The notebook panel runs forecasts in a worker so its Stop button can be used.
The legacy `t0_assim`, `dt_assim`, and `n_assim` YAML fields are retained as
experiment metadata, but do not schedule automatic analyses: the caller or
panel explicitly requests every assimilation. The revised default `state_vars`
is `[ug]` to match the requested experiment.
The class's ordinary `run_until` is synchronous. Do not use the notebook's hard
kernel interruption as the pause mechanism; use a chosen stop time or the
cooperative Stop button. There is no rewind: branch a checkpoint before testing
an alternative observation. All mutable arrays are accessible for research,
but direct edits bypass controller validation; prefer its methods.

## Statistical and physical conventions

- **Uncertain is separate from updated.** Both `ug` and `K` are sampled on reset.
  `variables=("ug",)` restricts the analysis, not prior uncertainty. To make K
  known, set `K_error=0, spread_K=0`. To make it fixed but wrong, set
  `K_error` nonzero and `spread_K=0`.
- `ug_error` and `K_error` are **signed offsets**: prior center = truth + error.
  `sigma_ug` is the Gaussian SD. `spread_K` is a uniform half-width; the lower
  bound is truncated at `1e-6 m²/s`. If truncation occurs, the realized prior
  mean is not the nominal center. Use priors away from zero for clean comparisons.
- The forecast always evolves both u and v. Observations always come from the
  complete physical wind grid, independently of which variables are estimated.
- Observation ordering is all heights for the first observed variable, then all
  heights for the next: `[u(z1), u(z2), v(z1), v(z2)]` for two heights.
  Heights between levels use linear interpolation; out-of-domain heights fail.
- `std=0.5` means an SD of 0.5 m/s, hence a diagonal variance of 0.25.
  A full positive-definite correlated covariance can be passed as `R=` instead.
- `noise=False` creates a truth-valued diagnostic observation but still uses the
  assumed `std`/`R` for the analysis; it does not imply zero observation variance.
- K is transformed to **log(K)** during assimilation and exponentiated back.
  Gain rows corresponding to K therefore refer to log(K), not physical K.
  This is an EnKF approximation in transformed coordinates, not a constrained
  Gaussian update in physical K. Overflow/underflow fails rather than clipping.
- u/v analysis vectors contain only the interior `nz-1` levels. For `nz=100`,
  `(u,v,ug,K)` has 200 entries. Boundary conditions are reapplied afterward.
  With an ug-only update, the upper u boundary changes immediately but interior
  winds are unchanged at analysis time; later forecasts respond dynamically.
- Each observation object is rejected if reused in the same branch or at the
  wrong time. Several independent observations at the same time are allowed.
  Entering the same measurement again as a new object circumvents this guard;
  do not do this unless using a deliberately tempered iterative algorithm.
- This is a **stochastic unlocalized EnKF**. Inflation and localization are
  deliberately not implemented; nondefault requests raise clear errors. The
  original code accepted these options without using them. Global parameter
  localization needs its own design, not a height assigned arbitrarily to K/ug.
- `exp.analyses` retains every full background and analysis, observation, gain,
  innovation and observation-space prediction. Main history keeps the latest
  analysis when several updates have the same time. `copy()` is an in-memory
  checkpoint; NetCDF exports contain model history, not a full restart including RNGs.

## Numerical implementation

The vectorized forward-Euler model is retained. Its timestep now accounts for
both diffusion and Coriolis rotation. For a vertical discrete diffusion
eigenvalue `-a` and Coriolis parameter `f`, explicit Euler requires
`dt <= 2a/(a²+f²)`; a 0.9 safety factor and `dt_max` are applied.
This is a linear stability bound, not an accuracy guarantee: timestep/grid
convergence remain part of research validation.

The controller synchronizes truth, ensemble and control at short physical-time
boundaries, while each uses its own stable substeps. Thus changing K in the
assimilating ensemble cannot change the integration of truth or control.
Stopping at an off-grid time changes Euler step partitioning, so an arbitrary
split run need not be bit-identical to one that never stopped there. Matched
experiment branches should use the same stop/observation schedule.

## Files and migration

| File | Role |
|---|---|
| `src/experiment.py` | Controller replacing the missing main function. |
| `src/ekman.py` | Existing model structure, with corrected state-vector, validation, prior and numerical handling. |
| `src/enkf.py` | Small validated NumPy EnKF with persistent RNG. |
| `src/observation_operator.py` | Full-grid wind interpolation independent of the selected analysis variables. |
| `src/widgets.py` | Optional notebook panel and plots. |
| `interactive_ekman.ipynb` | Working examples and interactive controls. |
| `src/example.py` | Six-cycle ug-only example and exported results. |
| `REVIEW.md`, `RESEARCH_PLAN.md` | Findings, interpretation, and experimental programme. |

This is a clean revised project, not a live GitHub checkout. The archive also
contains the original extracted text files for reference. The old tests and
examples refer to classes already absent from the active supplied implementation;
the revised tests target `EkmanEns` and `EkmanExperiment`. Do not merge old
`tests/test_ekman.py` unchanged into this test suite. Old full-boundary state
vectors and the old `ObservationOperator` interface need migration to the
interior-vector and `observe()` APIs. No notebook containing the original main
function was available to migrate automatically.
