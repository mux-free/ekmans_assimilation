# Review of the supplied Ekman repository snapshot

Reviewed 18 November 2026. Scope: the attached text export, not the live GitHub
branch or the binary `ekman_oop.zip`. The requested
`main(sim_config, ekman_config)` is not present. The only function named `main`
is a different CLI entry point under `src/deprecated/ekman.py`. Findings below
refer to the active `src/ekman.py` and `src/enkf.py` unless stated otherwise.
Line numbers refer to the extracted original files included with this delivery.

## Findings that can invalidate the experiment

| Priority | Finding and evidence | Consequence | Revision |
|---|---|---|---|
| P1 | `EnsembleCreator.__call__`, `src/ekman.py:118`: sampling is gated by `data_assimilation.state_vars`. Selecting only ug returns only `ug_ens`. | The generator cannot supply an uncertain K ensemble for the requested ug-only update. A caller may substitute one fixed K or fail; the missing main prevents determining which. | Always sample both parameters; select updated variables only in `assimilate`. |
| P1 | `src/enkf.py:178` recreates `default_rng(random_seed)` inside every analysis. `src/observation_operator.py:133` also recreates the synthetic-observation RNG. | Repeated analyses reuse perturbations; repeated synthetic observations with unchanged size/SD reuse the same noise unless the caller explicitly changes seeds. These are not independent successive errors. | Persistent, separate prior/data/filter streams; fresh draws at every call, reproducible by seed/reset. |
| P1 when enabled | `src/enkf.py:144–161` stores inflation/localization settings, but `analysis_step` never uses either. Reproduced identical outputs with inflation 1 vs 4 and localization enabled. | Experiments labelled inflated/localized actually use the same basic filter. | Minimal filter rejects unsupported nondefault options rather than claiming they work. |
| P1 for state DA | `src/ekman.py:256–278` documents interior winds but packs `nz+1` levels and accepts the same on unpacking. `(u,v,K)` has 203 entries at nz=100 rather than 199. | Observation/state indexing built from the documented layout is wrong. Surface/top degrees of freedom are included and their analyzed values are later discarded by boundary enforcement. | Pack only interior winds; observe from an independently constructed full-grid operator. |
| P1 robustness | `src/ekman.py:235–251` mutates u before validating v or parameter shapes; `K=np.nan` passes the positivity check. Nonfinite end times are not rejected. | A failed update can leave a partly modified model; NaNs can contaminate or prevent meaningful integration. | Validate complete updates before mutation; reject nonfinite inputs and mismatched posterior layouts. |

“P1” here means fix before drawing conclusions from the affected experiment;
it does not mean every default run necessarily fails.

## Reproducibility and implementation defects

1. **The provided tests cannot collect against the active code.** Running
   `PYTHONPATH=src python -m pytest -q` stops at an ImportError for `Ekman`.
   The tests and `src/example.py` import `Ekman`, `Simulation`, `StateUpdater`,
   which the active `src/ekman.py` no longer defines. The alternative
   `tests/ekman_new.py` is another implementation, not a regression suite for
   the active model. README instructions describe stale APIs and missing scripts.
   This creates false confidence about what is actually covered.
2. **Default construction fails.** `EkmanEns()` assigns `config=None` and then
   accesses `self.config.lz` (`src/ekman.py:136–140`). It now creates a default config.
3. **The second config class is unusable.** `configs/config.py` lacks imports for
   `dataclass`, `Path` and `yaml`. Even after adding imports, its flat schema
   does not match the nested YAML. The revision uses one canonical class.
4. **Dependency metadata omits PyYAML**, which the model imports, and there is
   no test extra for pytest. The new project declares runtime and optional
   dependencies and removes JAX from this NumPy-only EnKF path.
5. **`dataset().squeeze()` removes time at a single snapshot**, not just the
   singleton ensemble dimension (`src/ekman.py:308–309`). This breaks uniform
   indexing at initialization. The revision squeezes only ensemble.
6. **Observation height validation is inconsistent.** The old helper accepts
   an int but a scalar float fails; some arrays/tuples are rejected inconsistently;
   out-of-domain heights are silently snapped to a boundary. Negative raw
   indices can wrap in the matrix builder. The new controller checks heights
   and interpolates between grid levels.
7. **Euler's diffusion-only CFL is not a general stability guarantee.** For
   this finite-domain linear operator, eigenvalues are `-a ± if`. Stability is
   `dt <= 2a/(a²+f²)`. With nz=100, L=500 m, f=1e-4 s^-1 and K=1e-4 m²/s,
   the slowest mode permits only about 0.7895 s, while the original chooses 5 s.
   This counterexample is outside the supplied default prior (K>=1), so it
   does not establish instability in those default runs. It matters for freely
   editable parameters and posterior K values. The revised limit handles both
   rotation and diffusion; separate accuracy tests are still necessary.

Other issues: separate prior draws repeatedly restart the same RNG stream,
which is not a clean design for independent priors; derived `ug_sim`/`K_sim`
values become stale if dictionaries are mutated after configuration creation;
the accepted `obs_std` constructor argument is unused; direct Gaussian K
updates can produce negative values; and replacing the main history at an
analysis time loses the background unless it is separately retained. The
controller addresses these through explicit configuration edits, one validated
observation representation, log(K), and a list of full analysis records.

## What appears sound

The active model evaluates both wind tendencies before updating either field,
so it does not accidentally mix old u with updated v in one Euler step.
Its signs implement

`du/dt = K u_zz + f(v-vg)`, `dv/dt = K v_zz - f(u-ug)`.

The basic EnKF cross-covariance formula is appropriate: X need contain only
the estimated parameters, while Y contains predicted wind observations from
the full model. Using `solve` rather than explicitly inverting the innovation
covariance is sensible. The original covariance denominators use N-1.
Positive K validation exists, but is incomplete and not a statistical treatment
of its positivity constraint. Event-time integration and independent copying
are useful foundations for the new controller.

## Statistical choices that are not code bugs

**Bias versus spread.** A signed `K_error=1.5` changes the prior center; it is
not an uncertainty SD. `spread_K=1.5` specifies a uniform half-width; without
truncation its SD is `1.5/sqrt(3)`. With truth K=1, the supplied config centers
K at 2.5 and samples approximately [1,4]. The truth is at the edge of that
support. Poor recovery may therefore reflect prior misspecification as well
as the filter. The same applies to signed ug_error.

**No instant interior wind correction during parameter-only DA.** Updating ug
or K changes subsequent tendencies; it does not reconstruct the current wind
profile. At the same physical time, interior predicted observations therefore
remain unchanged. The ug-dependent top boundary is the exception. Re-spinning
up after each parameter update would answer a different inverse problem; the
controller deliberately does not do that.

**Keeping K uncertain but unupdated is a restricted update.** Each K member
participates in wind predictions and hence in `Pyy`; ug–K correlation can develop
through the ug analysis. The marginal K ensemble is deliberately not conditioned
on the new observation. This is not the full Bayesian posterior for both
parameters. It is useful for the user's proposed controlled experiment, and is
related to consider/Schmidt filtering ideas without claiming this controller
implements every feature of those algorithms.

**A 72-hour individually spun-up prior is almost an equilibrium experiment for
the default K range.** It encodes wind–parameter correlations created by each
member's own equilibrium. It is quite different from cold-start initial wind
uncertainty. A zero-noise observation does not by itself resolve parameter
non-identifiability, and repeated steady observations of one scalar generally
add precision without adding an independent sensitivity direction.

**The 500 m top is part of the model physics.** Setting u=ug there imposes a
finite-height geostrophic boundary. For f=1e-4, the classical scale
`sqrt(2K/f)` ranges from about 141 to 283 m for K in [1,4]. The top is only
1.8–3.5 of those scales away. Check a taller domain before interpreting effects
near 400–500 m or treating this as an effectively infinite atmosphere.

## Validation supplied

The revised tests check selective updates, independent repeated noise,
observation ordering/interpolation and SD-to-variance conversion, duplicate
observation protection, repeated analyses, positive K, transactional failure,
configuration write/read/reset, checkpoint independence, off-grid pause/resume,
cooperative stopping, singleton output dimensions and NetCDF output. Numerical
checks compare Euler convergence with a matrix exponential of the same
semidiscrete equations and compare a long run with their exact steady solution.
An additional test verifies that a different assimilating K cannot change the
truth/control numerical trajectory. Optional UI tests exercise observation
preparation, assimilation, plotting and asynchronous Run callbacks.

These checks establish software behavior, not scientific skill or real-world
validity. `VALIDATION.txt` records the executed results. The six-cycle demo
under `outputs/` is one seed and one configuration; it is not a Monte Carlo
benchmark. A full visual browser audit of the notebook frontend was not done.

In that executed demo (the supplied physical/prior settings, seed 14, six
ug-only updates using u/v at 100 and 300 m), mean ug goes from 12.2976 to
9.4370 m/s; truth is 10. K stays memberwise constant, with ensemble mean
2.6216 m²/s versus truth 1. Interior vector RMSE initially falls, then reaches
1.9667 m/s at 4.5 h, compared with 1.6480 m/s for the no-DA control. This is
an illustration of why post-cycling forecast validation is necessary, not
evidence that ug-only assimilation generally worsens the forecast.
