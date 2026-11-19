"""Minimal 1D Ekman model for single runs and parameter ensembles."""

from copy import deepcopy
from dataclasses import dataclass, asdict
from pathlib import Path
import yaml

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt


@dataclass
class EkmanConfig:
    nz: int = 100
    lz: float = 500.0
    fcor: float = 1e-4
    dt_max: float = 5.0
    num_states = 2                      # two states because we have u and v winds
    spinup_seconds: float = 72 * 3600
    output_interval: float = 60.0
    
    def __post_init__(self):
        if not isinstance(self.nz, int) or isinstance(self.nz, bool) or self.nz < 2:
            raise ValueError("nz must be an integer >= 2")
        if not all(np.isfinite(x) for x in (self.lz, self.fcor, self.dt_max,
                                           self.spinup_seconds, self.output_interval)):
            raise ValueError("Model settings must be finite")
        if min(self.lz, self.dt_max, self.output_interval) <= 0 or self.spinup_seconds < 0:
            raise ValueError("Invalid grid or time settings")
        self.state_dim = self.nz+1
        self.dz = self.lz / self.nz


@dataclass
class SimConfig:
    total_run_time: int
    parameters: dict
    ensemble: dict
    data_assimilation: dict
    observations: dict

    def __post_init__(self):
        for name in ("parameters", "ensemble", "data_assimilation", "observations"):
            setattr(self, name, deepcopy(getattr(self, name)))
        if not np.isfinite(self.total_run_time) or self.total_run_time < 0:
            raise ValueError("total_run_time must be finite and nonnegative")
        p, e = self.parameters, self.ensemble
        if not all(np.isfinite(p[k]) for k in ("ug_truth", "K_truth", "ug_error", "K_error")):
            raise ValueError("Truth values and errors must be finite")
        if p["K_truth"] <= 0 or p["K_truth"] + p["K_error"] <= 0:
            raise ValueError("Truth and prior central K must be positive")
        n = e["ensemble_size"]
        if not isinstance(n, int) or isinstance(n, bool) or n < 2:
            raise ValueError("ensemble_size must be an integer >= 2")
        if not all(np.isfinite(e[k]) and e[k] >= 0 for k in ("sigma_ug", "spread_K")):
            raise ValueError("Parameter spreads must be finite and nonnegative")
        self.parameters["ug_sim"] = self.parameters["ug_truth"] + self.parameters["ug_error"]
        self.parameters["K_sim"]  = self.parameters["K_truth"]+ self.parameters["K_error"]

    @classmethod
    def from_yaml(cls, path: str | Path) -> "SimConfig":
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        return cls(**data)

    def to_yaml(self, path):
        """Write this experiment's settings; derived centers are recalculated on load."""
        data = asdict(self)
        data["parameters"].pop("ug_sim", None)
        data["parameters"].pop("K_sim", None)
        Path(path).write_text(yaml.safe_dump(data, sort_keys=False))


# ======================================================================================================================

 
class EnsembleCreator:
    def __init__(
        self,
        sim_config: SimConfig,
        random_seed: int = 14,
        K_ens_distri: str = "uniform" # OR "lognormal"
    ):  

        self.c = sim_config
        self.n_members = sim_config.ensemble["ensemble_size"]
        
        self.params = sim_config.parameters
        self.ug_sim = self.params["ug_truth"] + self.params["ug_error"]
        self.K_sim  = self.params["K_truth"] + self.params["K_error"]
        self.sigma_ug = sim_config.ensemble["sigma_ug"]
        
        self.K_ens_distri = K_ens_distri
        if K_ens_distri == "uniform":
            sigma_K = sim_config.ensemble["spread_K"]
            self.low_K  = max(1e-6, self.K_sim - sigma_K)
            self.high_K = self.K_sim + sigma_K
        else:
            raise ValueError(f"Not implemented parameter ensemble of K with distribution: {K_ens_distri}")
        self.random_seed = random_seed
        self.rng = np.random.default_rng(random_seed)
        if self.high_K < self.low_K:
            raise ValueError("Invalid K prior bounds; adjust K_error/spread_K")

        
    def param_ensemble_ug_gauss(self):
        rng = self.rng
        # -- Normal distribution around a mean
        self.ug_ens = rng.normal(self.ug_sim, self.sigma_ug, self.n_members)
        
    def param_ensemble_K_uniform(self):
        rng = self.rng
        # -- Uniform distirbutioon with specified bounds
        self.K_ens = rng.uniform(self.low_K, self.high_K, self.n_members)  
        
    def param_ensemble_K_lognorm(self, mu_K, sigma_lnK):
        rng = self.rng
        # -- Log-Normal distribution 
        self.K_ens = mu_K * np.exp(sigma_lnK * rng.standard_normal(self.n_members))

    def plot_params_spread(self):
        ncols=1
        ylabel=""
        fig,ax=plt.subplots(figsize=(10,3))
        if hasattr(self, 'ug_ens'):
            ax.scatter(np.arange(1, self.n_members+1, dtype=int), self.ug_ens, color="tab:blue", label="Ug")
            ncols+=1
            ax.axhline(y=self.params["ug_truth"], color="tab:blue", label="truth")
            ylabel+="Ug [m/s]   &   "
            if self.params["ug_truth"] != self.ug_sim:
                ncols+=1
                ax.axhline(y=self.ug_sim, color="tab:blue", linestyle="--", label="model")
            
        if hasattr(self, 'K_ens'):
            ncols+=1
            ax.scatter(np.arange(1,self.n_members+1, dtype=int), self.K_ens, color="tab:orange", label="K")
            ax.axhline(y=self.params["K_truth"], color="tab:orange")
            if self.params["K_truth"] != self.K_sim:
                ax.axhline(y=self.K_sim, color="tab:orange", linestyle="--")
            ylabel+="K [m^2/s]"

        ax.set_xlabel("Member")
        ax.set_ylabel(ylabel)
        ax.set_xticks([int(i) for i in range(1,self.n_members+1)])
        ax.grid()
        ax.legend(bbox_to_anchor=(0.5,1.1), frameon=True, ncols=ncols, loc="center")

    def __call__(self):
        """Sample BOTH parameters, even when the analysis estimates only one."""
        self.param_ensemble_ug_gauss()
        self.param_ensemble_K_uniform()
        return {"ug_ens": self.ug_ens, "K_ens": self.K_ens}


class EkmanEns:
    """One Ekman run or an ensemble; all members share a grid and clock."""

    def __init__(self, ug=10.0, vg=0.0, K=2.0, config=None, random_seed=14):
        self.config = deepcopy(config) if config is not None else EkmanConfig()
        self.random_seed = random_seed
    
        self.z = np.linspace(0, self.config.lz, self.config.nz + 1)
        self.dz = self.config.lz / self.config.nz

        # Scalar inputs make one member; arrays define member-specific parameters.
        self.ug, self.vg, self.K = [a.copy() for a in np.broadcast_arrays(
            *[np.atleast_1d(np.asarray(x, dtype=float)) for x in (ug, vg, K)]
        )]

        self.n_members = len(self.K)
        self.is_ensemble = self.n_members > 1
        
        if any(a.ndim != 1 or not np.all(np.isfinite(a)) for a in (self.ug, self.vg, self.K)) or not self.n_members:
            raise ValueError("Parameters must be finite scalars or nonempty 1D arrays")
        if np.any(self.K <= 0):
            raise ValueError("K must be positive")

        self.u = self.ug[:, None] * self.z[None, :] / self.config.lz
        self.v = np.broadcast_to(self.vg[:, None], self.u.shape).copy()
        self.time = 0.0
        self._boundaries()
        self.history = [self._snapshot()]

    @property
    def dt(self):
        """One stable step for every member, based on the largest K."""
        # Eigenvalues of the interior diffusion/Coriolis operator are -a +/- i*f.
        # Euler is stable iff dt <= 2*a/(a*a + f*f). The minimum over vertical
        # modes occurs at an endpoint; evaluate both for every ensemble member.
        modes = np.array([1, self.config.nz - 1])
        a = 4 * self.K[:, None] / self.dz**2 * np.sin(modes * np.pi / (2*self.config.nz))**2
        limit = np.min(2*a / (a*a + self.config.fcor**2))
        return float(min(0.9 * limit, self.config.dt_max))

    def _boundaries(self):
        self.u[:, 0], self.v[:, 0] = 0.0, 0.0
        self.u[:, -1], self.v[:, -1] = self.ug, self.vg


    def step(self, dt=None):
        """Forward Euler; both tendencies use the old wind fields."""
        dt = self.dt if dt is None else dt
        if not np.isfinite(dt) or dt <= 0 or dt > self.dt * (1 + 1e-12):
            raise ValueError("Time step must be finite, positive and stable")
        Kdz = self.K[:, None] / self.dz**2
        du = Kdz * (self.u[:, 2:] - 2*self.u[:, 1:-1] + self.u[:, :-2]) \
             + self.config.fcor * (self.v[:, 1:-1] - self.vg[:, None])
        dv = Kdz * (self.v[:, 2:] - 2*self.v[:, 1:-1] + self.v[:, :-2]) \
             - self.config.fcor * (self.u[:, 1:-1] - self.ug[:, None])
        self.u[:, 1:-1] += dt * du
        self.v[:, 1:-1] += dt * dv
        self.time += dt

    def _snapshot(self):
        return (self.time, self.u.copy(), self.v.copy(),
                self.ug.copy(), self.vg.copy(), self.K.copy())

    def _record(self):
        state = self._snapshot()
        if abs(self.history[-1][0] - self.time) < 1e-9:
            self.history[-1] = state  # At an analysis time, keep posterior in history.
        else:
            self.history.append(state)

    def _integrate(self, until, record):
        if not np.isfinite(until):
            raise ValueError("End time must be finite")
        if until < self.time - 1e-9:
            raise ValueError("Cannot integrate backwards")
        while self.time < until - 1e-9:
            if record:
                interval = self.config.output_interval
                next_output = (int((self.time + 1e-9) / interval) + 1) * interval
                target = min(until, next_output)
            else:
                target = until
            self.step(min(self.dt, target - self.time))
            if abs(self.time - target) < 1e-9:
                self.time = float(target)
                if record:
                    self._record()

    def spin_up(self):
        """Independently spin up all members, then reset the clock to t0=0."""
        if self.time != 0:
            raise ValueError("Spin-up must precede the forecast")
        self._integrate(self.config.spinup_seconds, record=False)
        self.time = 0.0
        self.history = [self._snapshot()]
        return self

    def run(self, until):
        """Run until absolute time (seconds after spin-up); return history."""
        self._integrate(float(until), record=True)
        return self.dataset()

    def _parameter(self, value):
        return np.broadcast_to(np.asarray(value, float), (self.n_members,)).copy()

    def _profile(self, value):
        """Broadcast a full/interior profile to (ensemble, interior z)."""
        a = np.asarray(value, dtype=float)
        if a.ndim == 1 and a.size in (len(self.z), len(self.z)-2):
            a = a[None, :]
        if a.ndim and a.shape[-1] == len(self.z):
            a = a[..., 1:-1]  # Boundary levels are prescribed, not estimated.
        return np.broadcast_to(a, (self.n_members, len(self.z)-2))

    def update(self, u=None, v=None, ug=None, vg=None, K=None, du=None, dv=None):
        """Replace profiles/parameters or add wind increments at current time."""
        # Validate everything before mutating any part of the model.
        profiles = {n: self._profile(x).copy() for n, x in
                    (("u", u), ("v", v), ("du", du), ("dv", dv)) if x is not None}
        params = {n: self._parameter(x) for n, x in
                  (("ug", ug), ("vg", vg), ("K", K)) if x is not None}
        if not all(np.all(np.isfinite(x)) for x in (*profiles.values(), *params.values())):
            raise ValueError("Updated values must be finite")
        if "K" in params and np.any(params["K"] <= 0):
            raise ValueError("K must be positive")
        new_u, new_v = self.u.copy(), self.v.copy()
        for name, new in (("u", new_u), ("v", new_v)):
            if name in profiles:
                new[:, 1:-1] = profiles[name]
            if "d"+name in profiles:
                new[:, 1:-1] += profiles["d"+name]
        if not np.all(np.isfinite([new_u, new_v])):
            raise ValueError("Wind increment overflow")
        self.u, self.v = new_u, new_v
        for name, value in params.items():
            setattr(self, name, value)
        self._boundaries()
        self._record()
        return self

    update_step = update  # Compatibility with the earlier EkmanEns API.

    @staticmethod
    def _variables(variables):
        variables = (variables,) if isinstance(variables, str) else tuple(variables)
        if not variables or len(set(variables)) != len(variables) or not set(variables) <= {"u", "v", "ug", "vg", "K"}:
            raise ValueError("Choose unique variables from u, v, ug, vg, K")
        return variables

    def state_vector(self, variables=("u", "v", "ug", "vg", "K")):
        """(N, D) matrix for the EnKF; u and v exclude fixed boundary levels."""
        variables = self._variables(variables)
        parts = [getattr(self, v)[:, 1:-1] if v in ("u", "v")
                 else getattr(self, v)[:, None] for v in variables]
        return np.concatenate(parts, axis=1).copy()

    def set_state_vector(self, X, variables=("u", "v", "ug", "vg", "K")):
        """Apply an EnKF posterior shaped (N, D) or (N, *, *)."""
        variables = self._variables(variables)
        X = np.asarray(X, float)
        if X.ndim not in (2, 3) or X.shape[0] != self.n_members:
            raise ValueError("Posterior must have members along its first axis")
        X = X.reshape(self.n_members, -1)
        sizes = [self.config.nz-1 if v in ("u", "v") else 1 for v in variables]
        if X.shape[1] != sum(sizes):
            raise ValueError("Posterior size does not match selected variables")
        values = {}
        i = 0
        for name, size in zip(variables, sizes):
            block = X[:, i:i+size]
            values[name] = block if name in ("u", "v") else block[:, 0]
            i += size
        self.update(**values)

    def copy(self):
        """Independent copy for branching forecasts."""
        return deepcopy(self)

    def current(self):
        """Return the current state (no history)."""
        return xr.Dataset(
            {"u": (("ensemble", "z"), self.u.copy()),
             "v": (("ensemble", "z"), self.v.copy()),
             "u_g": ("ensemble", self.ug.copy()),
             "v_g": ("ensemble", self.vg.copy()),
             "K": ("ensemble", self.K.copy())},
            coords={"ensemble": np.arange(1, self.n_members+1),
                    "z": self.z, "time": self.time},
        )

    def dataset(self):
        """Return saved profiles with one shared 1D time coordinate."""
        ds = xr.Dataset(
            {"u": (("ensemble", "time", "z"), np.stack([h[1] for h in self.history], axis=1)),
             "v": (("ensemble", "time", "z"), np.stack([h[2] for h in self.history], axis=1)),
             "u_g": (("ensemble", "time"), np.stack([h[3] for h in self.history], axis=1)),
             "v_g": (("ensemble", "time"), np.stack([h[4] for h in self.history], axis=1)),
             "K": (("ensemble", "time"), np.stack([h[5] for h in self.history], axis=1))},
            coords={"ensemble": np.arange(1, self.n_members+1),
                    "time": [h[0] for h in self.history], "z": self.z},
        )

        if ds.ensemble.size==1:
            return ds.squeeze("ensemble", drop=True)
        else:
            return ds


def _error(forecast, reference):
    """Require identical coordinates rather than silently aligning by intersection."""
    forecast, reference = xr.align(forecast, reference, join="exact")
    return forecast - reference


def rmse(forecast, reference, dim=None):
    return np.sqrt((_error(forecast, reference)**2).mean(dim=dim))


def bias(forecast, reference, dim=None):
    return _error(forecast, reference).mean(dim=dim)


def mae(forecast, reference, dim=None):
    return abs(_error(forecast, reference)).mean(dim=dim)


def vector_rmse(forecast, reference, dim=None):
    return np.sqrt((_error(forecast.u, reference.u)**2 +
                    _error(forecast.v, reference.v)**2).mean(dim=dim))
