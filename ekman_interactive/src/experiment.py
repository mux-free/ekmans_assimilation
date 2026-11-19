"""Interactive controller: configure -> reset -> run -> observe -> assimilate -> run."""
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass
from threading import Event, Lock
from uuid import uuid4

import numpy as np

from .ekman import EkmanConfig, SimConfig, EkmanEns, EnsembleCreator
from .enkf import EnsembleKalmanFilter, validate_covariance
from .observation_operator import wind_matrix, predict_wind


@dataclass(frozen=True)
class Observation:
    time: float
    values: np.ndarray
    R: np.ndarray
    heights: np.ndarray
    variables: tuple[str, ...]
    id: str


@dataclass
class AnalysisRecord:
    time: float
    variables: tuple[str, ...]
    observation: Observation
    background: object  # xarray.Dataset: complete physical state, before update
    analysis: object
    predicted_before: np.ndarray
    predicted_after: np.ndarray
    innovation: np.ndarray
    gain: np.ndarray  # Rows for K are in log(K) coordinates.
    innovation_covariance: np.ndarray


class EkmanExperiment:
    """Small controller around EkmanEns; all public times are seconds after spin-up.

    Both ug and K are sampled regardless of `state_vars`. An ug-only update
    keeps each member's K fixed. Winds are observations even when they are not
    analysis variables. K is estimated in log space to preserve positivity.

    run_until(t) stops at exactly t and can be called again. request_stop()
    cooperatively stops a running worker after a synchronized model step.
    Mutating operations are serialized; a second operation while busy fails.
    """

    def __init__(self, sim_config, ekman_config=None, seed=14, spinup=True):
        self.sim_config = deepcopy(sim_config)
        self.ekman_config = deepcopy(ekman_config or EkmanConfig())
        self.seed = seed
        self._lock, self._stop = Lock(), Event()
        self.needs_reset = True
        self.reset(spinup=spinup)

    @classmethod
    def from_yaml(cls, path, ekman_config=None, **kwargs):
        return cls(SimConfig.from_yaml(path), ekman_config, **kwargs)

    @contextmanager
    def _exclusive(self):
        if not self._lock.acquire(blocking=False):
            raise RuntimeError("Experiment is busy; request_stop(), then wait for run_until() to return")
        try:
            yield
        finally:
            self._lock.release()

    def _ready(self):
        if self.needs_reset:
            raise RuntimeError("Configuration changed; call reset() to start the new experiment")

    @property
    def time(self):
        return self.ensemble.time

    def set_parameters(self, *, ug_truth=None, K_truth=None, ug_error=None,
                       K_error=None, sigma_ug=None, spread_K=None, save_to=None):
        """Edit initial truth/bias/spread settings. Call reset() before running.

        error = prior center minus truth (signed bias); sigma_ug is an SD,
        spread_K a uniform half-width. Save only when save_to is supplied.
        """
        with self._exclusive():
            config = deepcopy(self.sim_config)
            for key, value in (("ug_truth", ug_truth), ("K_truth", K_truth),
                               ("ug_error", ug_error), ("K_error", K_error)):
                if value is not None:
                    config.parameters[key] = float(value)
            for key, value in (("sigma_ug", sigma_ug), ("spread_K", spread_K)):
                if value is not None:
                    config.ensemble[key] = float(value)
            config.__post_init__()
            EnsembleCreator(config, self.seed)  # Validate the prior support too.
            if save_to is not None:
                config.to_yaml(save_to)
            self.sim_config = config
            self.needs_reset = True
        return self

    def set_errors(self, *, ug_error=None, K_error=None, save_to=None):
        return self.set_parameters(ug_error=ug_error, K_error=K_error, save_to=save_to)

    def save_config(self, path):
        with self._exclusive():
            self.sim_config.to_yaml(path)

    def reset(self, *, spinup=True):
        """Rebuild truth, ensemble and identical no-DA control; reset RNG streams."""
        with self._exclusive():
            config = deepcopy(self.sim_config)
            config.__post_init__()
            # Separate independent streams for priors, synthetic data, and EnKF noise.
            prior_seed, obs_seed, filter_seed = np.random.SeedSequence(self.seed).spawn(3)
            p = config.parameters
            samples = EnsembleCreator(config, prior_seed)()
            truth = EkmanEns(ug=p["ug_truth"], K=p["K_truth"], config=self.ekman_config)
            ensemble = EkmanEns(ug=samples["ug_ens"], K=samples["K_ens"], config=self.ekman_config)
            if spinup:
                truth.spin_up()
                ensemble.spin_up()
            # Commit only after successful construction/spin-up.
            self.truth, self.ensemble = truth, ensemble
            self.control = ensemble.copy()
            self.sim_config = config
            self.obs_rng = np.random.default_rng(obs_seed)
            self.filter = EnsembleKalmanFilter(ensemble.n_members, np.eye(1), random_seed=filter_seed)
            self.analyses = []  # List preserves separate updates at the same physical time.
            self._used_observations = set()
            self._stop.clear()
            self.needs_reset = False
            self.stopped_early = False
        return self

    def request_stop(self):
        """Thread-safe cooperative interrupt; does not rewind or reset anything."""
        self._stop.set()

    interrupt = request_stop

    def run_until(self, until=None, *, on_step=None):
        """Advance all three models on a shared clock; return their histories.

        on_step(experiment), if supplied, runs after each complete synchronized
        step. It may call request_stop(), but cannot mutate/assimilate while busy.
        A UI can run this method in a worker and call request_stop from its Stop
        button. Wait for the worker to finish before assimilating.
        """
        with self._exclusive():
            self._ready()
            until = float(self.sim_config.total_run_time if until is None else until)
            if not np.isfinite(until) or until < self.time:
                raise ValueError("until must be finite and not in the past")
            self._stop.clear()
            models = (self.truth, self.ensemble, self.control)
            interval = self.ekman_config.output_interval
            try:
                while self.time < until - 1e-9 and not self._stop.is_set():
                    t = self.time
                    next_output = (int((t + 1e-9)/interval) + 1)*interval
                    target = min(until, next_output, t+self.ekman_config.dt_max)
                    # Synchronize at short chunk boundaries, but let each model
                    # choose its own stable substeps. Assimilating K must NOT
                    # change the truth/control numerical time integration.
                    for model in models:
                        model._integrate(target, record=False)
                    if abs(self.time-target) < 1e-9:
                        for model in models:
                            model.time = target
                            if abs(target-next_output) < 1e-9 or abs(target-until) < 1e-9:
                                model._record()
                    if on_step is not None:
                        on_step(self)
            finally:
                # Also save an off-output-grid stop requested by the UI.
                for model in models:
                    model._record()
            self.stopped_early = self.time < until - 1e-9
            return self._datasets()

    run = run_until

    def _datasets(self):
        return {"truth": self.truth.dataset(), "ensemble": self.ensemble.dataset(),
                "control": self.control.dataset()}

    def datasets(self):
        with self._exclusive():
            self._ready()
            return self._datasets()

    def observe(self, values=None, *, heights=None, variables=None, std=None,
                R=None, noise=True):
        """Create observations at the CURRENT time, synthetic or manually supplied.

        Ordering is [u at every height, v at every height] for variables=(u,v).
        std is in m/s; R is a covariance in (m/s)^2 and can be correlated.
        With values=None, sample truth; noise=False gives a noise-free diagnostic
        value while retaining the specified assumed measurement uncertainty.
        """
        with self._exclusive():
            self._ready()
            defaults = self.sim_config.observations
            H, heights, variables = wind_matrix(
                self.ensemble.z, defaults["z_obs"] if heights is None else heights,
                defaults["obs_vars"] if variables is None else variables)
            size = len(H)
            if R is not None and std is not None:
                raise ValueError("Specify either std or R, not both")
            if R is None:
                sigma = np.broadcast_to(np.asarray(defaults["obs_error"] if std is None else std, float), (size,))
                if not np.all(np.isfinite(sigma)) or np.any(sigma <= 0):
                    raise ValueError("Observation standard deviations must be finite and positive")
                R = np.diag(sigma**2)
            covariance = validate_covariance(R, size)
            if values is None:
                values = predict_wind(self.truth, H)[0]
                if noise:
                    values = values + self.obs_rng.multivariate_normal(np.zeros(size), covariance)
            values = np.atleast_1d(np.asarray(values, float))
            if values.shape != (size,) or not np.all(np.isfinite(values)):
                raise ValueError(f"Supply {size} finite observation values in variable-major order")
            return Observation(self.time, values.copy(), covariance, heights, variables, uuid4().hex)

    def assimilate(self, observation, *, variables=None):
        """One EnKF update; only selected variables change, then forecast may resume.

        An ug-only update changes the prescribed top u boundary, but leaves
        interior u/v and every member's K unchanged at the analysis instant.
        Parameter effects on interior winds appear through later integration.
        Each observation object can be used only once per experiment branch.
        """
        with self._exclusive():
            self._ready()
            obs = deepcopy(observation)
            if not isinstance(obs, Observation):
                raise TypeError("Use observe() to construct an Observation")
            if not np.isfinite(obs.time) or abs(obs.time-self.time) > 1e-8:
                raise ValueError("Observation time must match the current model time")
            if obs.id in self._used_observations:
                raise ValueError("This observation was already assimilated in this branch")
            selected = self.ensemble._variables(
                self.sim_config.data_assimilation["state_vars"] if variables is None else variables)
            H, _, _ = wind_matrix(self.ensemble.z, obs.heights, obs.variables)
            X = self.ensemble.state_vector(selected)
            Y = predict_wind(self.ensemble, H)
            k_column = None
            if "K" in selected:
                k_column = sum(self.ekman_config.nz-1 if name in ("u", "v") else 1
                               for name in selected[:selected.index("K")])
                X[:, k_column] = np.log(X[:, k_column])
            background = self.ensemble.current()
            # Reject failed analyses without advancing the filter RNG or model.
            filter_before = deepcopy(self.filter)
            try:
                Xa = self.filter.analysis_step(X, Y, obs.values, R=obs.R)
                if k_column is not None:
                    with np.errstate(over="raise", under="raise", invalid="raise"):
                        Xa[:, k_column] = np.exp(Xa[:, k_column])
                self.ensemble.set_state_vector(Xa, selected)
            except Exception:
                self.filter = filter_before
                raise
            record = AnalysisRecord(
                self.time, selected, obs, background, self.ensemble.current(),
                Y.copy(), predict_wind(self.ensemble, H), self.filter.innovation.copy(),
                self.filter.kalman_gain.copy(), self.filter.innovation_covariance.copy())
            self.analyses.append(record)
            self._used_observations.add(obs.id)
            return deepcopy(record)

    enkf_step = assimilate

    def copy(self):
        """Branch at the current checkpoint, including histories and RNG states."""
        with self._exclusive():
            clone = object.__new__(type(self))
            for name, value in self.__dict__.items():
                if name not in ("_lock", "_stop"):
                    setattr(clone, name, deepcopy(value))
            clone._lock, clone._stop = Lock(), Event()
            return clone
