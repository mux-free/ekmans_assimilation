# Research plan: estimating one parameter when another is uncertain

## Main question

When both large-scale forcing (ug) and vertical mixing (K) are uncertain, how
does restricting assimilation to ug affect (1) wind forecast skill, (2) ug
recovery, and (3) uncertainty calibration? Under which observation geometries
and flow regimes does ug absorb the error in K?

For the real-world connection, the working interpretation is boundary-layer
wind assimilation in a limited-area atmospheric model: ug represents uncertain
large-scale pressure-gradient forcing; K represents uncertain vertical mixing.
This is a mechanistic analogue, not a calibration of a full model's turbulence
scheme. Advection, horizontal heterogeneity, changing stratification and closure
physics are absent from the current Ekman column.

## Why a lower wind error need not mean a better ug

The proposed hypothesis can be understood with a local linearization of one
wind observation around a reference trajectory:

\[
\Delta y \simeq a\,\Delta u_g + b\,\Delta K + \epsilon.
\]

If K is fixed at the wrong value, an ug-only fit can compensate approximately by

\[
u_g^{\rm fit}-u_g^{\rm true}
\simeq -\frac{b}{a}(K^{\rm fixed}-K^{\rm true}),\quad a\ne0.
\]

The sign and strength depend on height, wind component and elapsed time;
there is no universal direction for this compensation.
For independent scalar priors, the linear ug gain is

\[
G_{u_g}=
\frac{a\,\sigma_{u_g}^2}
{a^2\sigma_{u_g}^2+b^2\sigma_K^2+R}.
\]

Thus retaining K uncertainty changes how much a wind innovation should alter
ug, even when K is not updated. After restricted updates the prior independence
assumption generally no longer holds; use the actual ensemble covariances.
These equations are a proposed local explanation, not an empirical result.

## Stage 1 — Establish the mechanism and numerical baseline

Use ug_truth=10 m/s, K_truth=2 m²/s, vg=0 and f=1e-4 s^-1 as an initial reference.
Retain the supplied grid first, but repeat at dz=10, 5 and 2.5 m and with dt_max
halved. Compare finite-domain equilibria and transient solutions with an exact
discrete linear solution. Repeat with a 1–2 km top while holding dz and observation
heights fixed; this tests top-boundary effects independently of resolution.

Run two explicitly separated initial-condition designs:

1. **Equilibrium design:** spin truth and members up individually. Diagnose
   residual tendencies rather than assuming a fixed spin-up duration suffices
   for every K. This reveals the steady wind–parameter mapping.
2. **Transient design:** use a prescribed common initial wind profile (or a
   separately controlled initial-state ensemble), then integrate truth and
   forecast with their respective parameters. Include a short 0–4 h spin-up
   comparison. This separates initial-state error from parameter error.

For a common initial interior profile, wind observations at time zero carry
no information about K until dynamics have created a response. If ug is
observed indirectly through the imposed top boundary, that special direct
constraint must be excluded or identified separately. For ug=10, K=2,
f=1e-4, characteristic scales include 1/f≈2.78 h and an inertial period
2π/f≈17.45 h; observe both short response times and longer relaxation.

## Stage 2 — Controlled assimilation comparisons

Use the same initial ensemble, same realized observations, same stopping times
and matched RNG streams across methods whenever their uncertainty assumptions
permit. “Known K” is an intentional different uncertainty model, not an otherwise
identical method. Generate/store the observation sequence once per replicate.

| Arm | Prior K treatment | Updated variables | Purpose |
|---|---|---|---|
| A | Uncertain, possibly biased | None | Matched no-DA baseline. |
| B | Known exactly: K_error=0, spread_K=0 | ug | Oracle reference for ug estimation. |
| C | Fixed but wrong: K_error≠0, spread_K=0 | ug | Effect of ignoring mixing uncertainty. |
| D | Uncertain and frozen memberwise | ug | Main requested experiment. |
| E | Uncertain | K | Symmetric test: can mixing absorb forcing error? |
| F | Uncertain | ug, K | Joint parameter estimation; assess confounding and identifiability. |
| G | Uncertain and frozen memberwise | u, v | Short-term state correction without parameter repair. |
| H | Uncertain | ug, K, u, v | Joint state/parameter estimation and possible imbalance. |

For a focused first result, prioritize A–D and F. Add G/H once the parameter-only
mechanism is clear. The current controller directly supports these arms; set K
spread/bias as in the table. It estimates K in log space in E/F/H, so include a
small-spread sensitivity check before attributing differences solely to the
choice of updated variables.

Separate signed bias from spread in the sweep. A manageable starting design is
ug_error in {-2,0,+2} m/s and K_error in {-0.5,0,+1} m²/s, with sigma_ug=1.5 m/s
and spread_K=0.4 m²/s. These K priors remain positive around K_truth=2 without
truncation. Next vary each spread independently, including priors whose support
misses truth. State explicitly which scenarios have truth outside prior support.

Start with N=20 (the supplied default), compare N=80 and 320 to distinguish
small-ensemble sampling error from systematic compensation. Use 100 independent
paired replicates for the initial experiment, increasing that number if the
confidence intervals are too wide. Report paired differences with bootstrap
intervals across replicate seeds, not across individual ensemble members.
Treat the number 100 as a planning starting point, not a guarantee of power.

## Stage 3 — Design observations to distinguish forcing and mixing

Compare u only at 100 m; u and v at 100 m; and both components at 50, 100, 200,
300 and 400 m. Keep at least some heights withheld, e.g. 150, 250 and 350 m.
Use SD=0.25, 0.5 and 1 m/s as illustrative noise levels, then replace them with
instrument- and representativeness-based estimates in the real-data stage.

For each design calculate a finite-difference sensitivity matrix over the
observed times and heights,

\[
S_{ij}=\partial [H x(t_i)]/\partial\theta_j,\qquad
\theta=(u_g,\log K).
\]

Whiten by the observation covariance and scale the parameter columns by their
prior SDs before comparing singular values. Inspect the angle between columns,
the smallest singular value, and contours of the two-parameter observation
misfit. Near-parallel columns mean observations cannot distinguish ug from K
well over that window. A normalized matrix avoids mistaking different units
for poor conditioning. When initial-state uncertainty is present, assess its
sensitivity directions too; the two-parameter matrix alone can be optimistic.

Repeated observations of one steady scalar mostly reduce noise along the same
sensitivity direction. Different heights/components or a known transient may
add directions. Check this numerically rather than assuming that observing v
or adding another height always resolves ambiguity. A low-dimensional grid
posterior or likelihood surface in (ug,K) is feasible here and is a useful
reference against which to assess EnKF Gaussian/linear approximations.

## Stage 4 — Cycle, stop assimilating, and test persistence

Compare 30 min, 1 h and 2 h cycles with a fixed number of updates, and also with
a fixed total observation window. Those are different experiments and should
not be pooled. Begin with six updates, then free forecasts to 0.5, 1, 3 and 6 h
after the final update. Longer runs can examine static-parameter spread collapse.

The original model prior is near equilibrium after 72 h, so additionally test
a known change in forcing, or a controlled initial wind perturbation, that is
shared across truth/forecast setups. A time-varying K truth requires extending
the current model; do not pretend the existing constant-K controller already
supports this. For transient forcing changes, define whether observations
represent before or after the change and keep that convention fixed.

Score these outcomes separately:

- Wind RMSE of the ensemble mean, vector RMSE, and mean bias by height and lead.
  Use held-out heights/times as the primary score; report assimilated locations
  separately. Exclude prescribed boundary levels from the primary interior score.
- ug bias/RMSE and K bias/RMSE in twin experiments; plot parameter trajectories
  next to wind skill to expose compensation.
- Ensemble CRPS for u and v, spread–error relations, interval coverage and rank
  histograms across independent cases. Include measurement uncertainty when
  evaluating predictive observation coverage. Ensemble parameter intervals from
  an EnKF are approximate and need empirical coverage checks.
- Innovations and their predicted covariance, normalized innovation squared,
  innovation autocorrelation, parameter–observation cross-covariances and gain.
  NIS is a diagnostic, not a guaranteed chi-square statistic for a small,
  nonlinear, non-Gaussian ensemble.
- ug–K ensemble correlations and wind increment profiles before/after each
  analysis; the latter distinguish instant state repair from slower parameter
  response. An ug-only analysis may initially leave interior wind error unchanged.

Use the uninflated filter as a clearly defined baseline. If spread collapse is
found, compare a specified parameter random walk, spread floor, or inflation
scheme as a separate methodological intervention. Tune on training replicates
and evaluate on held-out seeds. A deterministic ETKF/EnSRF comparison can test
sensitivity to observation-perturbation sampling; it is a future implementation,
not something included in this delivery.

## Stage 5 — Add model discrepancy before real observations

An identical-twin constant-K experiment is favorable to parameter recovery.
Next generate truth with a height-varying diffusivity K(z), while retaining a
constant effective K in the assimilating model. Implement the diffusion term
as `∂z[K(z) ∂z u]` (and for v), not simply `K(z) u_zz`; use a conservative
flux discretization and validate its new stability constraints. Then introduce
time-dependent forcing/mixing and correlated observation errors one at a time.

Ask whether the ug estimate remains tied to the true large-scale forcing or
becomes an effective correction for an inadequate mixing model. In these cases
there may be no single “true constant K”. Judge the effective parameter by
prediction and consistency, not by an arbitrary equality with one local K(z).
Instrument noise, vertical averaging and representativeness error must not be
collapsed into a deceptively small independent-error SD.

## Stage 6 — Transfer hypotheses to the atmospheric case

Use actual wind-profile sampling/averaging and realistic error covariance.
Obtain a separately constrained forcing estimate from pressure gradients or
larger-scale analyses, while acknowledging its uncertainty; an upper-level
wind observation is not automatically a perfect measurement of ug. Seek an
independent mixing constraint from turbulence/flux or stability information if
available. With wind profiles alone, agreement with observations cannot
establish physical identification of both parameters.

For a limited-area cold-start study, translate the toy result into a testable
hypothesis: does uncertainty in large-scale forcing let boundary-layer wind DA
make a persistent correction, and when does it merely compensate for mixing
error? Compare perturbing forcing/boundary conditions, mixing settings, both,
and initial winds, with matched observations. Evaluate independent profiles
and short-range forecasts after cycling ends. If retaining the earlier
4 h spin-up and six 30 min cycles, reproduce those windows in a transient toy
experiment; do not interpret a 72 h equilibrium initialization as the same regime.

Split by weather case/day or regime before tuning. If using the previously
discussed 60-case dataset, select folds by case rather than treating its many
vertical levels or cycles as independent samples. Check dependence on stability,
wind turning, low-level jets and boundary forcing. The toy model can explain
mechanisms and guide ensemble design; it cannot by itself establish that a
particular perturbation strategy improves the full atmospheric model.

## Deliverables for the first research iteration

1. A figure showing wind forecast skill versus ug bias for arms A–D/F.
2. Bias-sweep panels for ug_error × K_error, with paired confidence intervals.
3. Sensitivity/posterior contour plots for one-height vs multi-height u/v data.
4. Lead-time plots after the last analysis: state-only versus parameter-only
   versus joint updates.
5. A numerical/convergence appendix and a transparent table of initial conditions,
   priors, update variables, observation geometry, R, seeds and cycle schedule.

## Selected primary references

- Ruckstuhl, Y. M. & Janjić, T. (2018). *Parameter and state estimation with
  ensemble Kalman filter based algorithms for convective-scale applications.*
  QJRMS 144, 826–841. https://doi.org/10.1002/qj.3257.
  Relevant to state–parameter augmentation, physical constraints, global vs
  spatial parameter updating and maintaining parameter spread.
- Bell, Z., Dance, S. L. & Waller, J. A. (2025). *The ensemble transform
  Schmidt–Kalman filter: A novel method to compensate for observation
  uncertainty due to unresolved scales.* Atmospheric Science Letters 26, e1296.
  https://doi.org/10.1002/asl.1296. A related example of accounting for uncertainty
  in a component without estimating it as an ordinary resolved state; its
  unresolved-scale problem is not identical to frozen K here.
- Hunt, B. R., Kostelich, E. J. & Szunyogh, I. (2007). *Efficient data
  assimilation for spatiotemporal chaos: A local ensemble transform Kalman
  filter.* Physica D 230, 112–126. https://doi.org/10.1016/j.physd.2006.11.008;
  author preprint https://arxiv.org/abs/physics/0511236.
  Background for a later deterministic ensemble-transform comparison.

The staged design, formulas applied to this column, parameter ranges and
replicate counts above are proposals for this project; these references do not
validate their results in advance.
