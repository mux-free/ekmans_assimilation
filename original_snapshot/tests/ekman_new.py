"""A small, restartable 1D Ekman model for assimilation experiments."""

from copy import deepcopy
from dataclasses import dataclass
from math import floor, inf, isfinite

import numpy as np
import xarray as xr


@dataclass(frozen=True)
class EkmanConfig:
    nz: int = 100
    lz: float = 500.0
    fcor: float = 1e-4
    dt_max: float = 5.0
    spinup_seconds: float = 60 * 3600
    output_interval: float = 60.0


@dataclass
class ModelState:
    u: np.ndarray
    v: np.ndarray
    time: float
    timestep: int
    ug: float
    vg: float
    K: float


class Ekman:
    """Integrates the Ekman equations; knows nothing about assimilation or I/O."""

    def __init__(self, ug=10.0, vg=0.0, K=2.0, config=None):
        self.config = config or EkmanConfig()
        c = self.config
        if c.nz < 2 or c.lz <= 0 or c.dt_max <= 0 or c.spinup_seconds < 0:
            raise ValueError("Invalid grid or time configuration")
        if c.output_interval <= 0 or not all(isfinite(x) for x in (ug, vg, K, c.fcor, c.lz, c.dt_max, c.output_interval)) or K <= 0:
            raise ValueError("Invalid wind, diffusivity, or model configuration")
        self.z = np.linspace(0, c.lz, c.nz + 1)
        self.dz = c.lz / c.nz
        self.state = ModelState(
            u=ug * self.z / c.lz,
            v=np.full(c.nz + 1, vg, dtype=float),
            time=0.0, timestep=0, ug=float(ug), vg=float(vg), K=float(K),
        )
        self.apply_boundaries()

    @property
    def dt(self):
        return min(0.45 * self.dz**2 / self.state.K, self.config.dt_max)

    def apply_boundaries(self):
        s = self.state
        s.u[0], s.v[0] = 0.0, 0.0
        s.u[-1], s.v[-1] = s.ug, s.vg

    def tendencies(self):
        s = self.state
        diffusion_u = s.K * (s.u[2:] - 2 * s.u[1:-1] + s.u[:-2]) / self.dz**2
        diffusion_v = s.K * (s.v[2:] - 2 * s.v[1:-1] + s.v[:-2]) / self.dz**2
        du = diffusion_u + self.config.fcor * (s.v[1:-1] - s.vg)
        dv = diffusion_v - self.config.fcor * (s.u[1:-1] - s.ug)
        return du, dv

    def step(self, dt=None):
        dt = self.dt if dt is None else dt
        if dt <= 0 or dt > self.dt * (1 + 1e-12):
            raise ValueError("Time step must be positive and stable")
        du, dv = self.tendencies()  # Both use the OLD velocity fields.
        self.state.u[1:-1] += dt * du
        self.state.v[1:-1] += dt * dv
        self.state.time += dt
        self.state.timestep += 1

    def integrate(self, until):
        if until < self.state.time:
            raise ValueError("Cannot integrate backwards")
        while self.state.time < until - 1e-10:
            self.step(min(self.dt, until - self.state.time))
        self.state.time = float(until)

    def spin_up(self):
        """Spin up the model, then reset the clock to time zero."""
        if self.state.time != 0:
            raise ValueError("Spin-up must precede the forecast")
        self.integrate(self.config.spinup_seconds)
        self.state.time = 0.0
        self.state.timestep = 0

    def snapshot(self):
        return deepcopy(self.state)

    def restore(self, state):
        self.state = deepcopy(state)
        self.apply_boundaries()

    def copy(self):
        return deepcopy(self)


@dataclass
class StateUpdater:
    """Apply prescribed wind increments and/or replace model parameters."""

    ug: float | None = None
    vg: float | None = None
    K: float | None = None
    u: np.ndarray | None = None  # New full or interior u profile
    v: np.ndarray | None = None  # New full or interior v profile
    du: float = 0.0
    dv: float = 0.0
    h_obs: float | None = None
    sigma: float | None = None

    def update_step(self, model):
        """Replace profiles/parameters, then optionally add Gaussian increments."""
        s = model.state
        if self.K is not None and (not isfinite(self.K) or self.K <= 0):
            raise ValueError("Updated K must be positive and finite")
        if not all(isfinite(x) for x in (self.du, self.dv)):
            raise ValueError("Wind increments must be finite")
        if self.ug is not None and not isfinite(self.ug):
            raise ValueError("Updated ug must be finite")
        if self.vg is not None and not isfinite(self.vg):
            raise ValueError("Updated vg must be finite")

        # Either full profiles (including boundaries) or interior profiles.
        def interior(profile):
            if profile is None:
                return None
            values = np.asarray(profile, dtype=float)
            if values.shape == (model.config.nz + 1,):
                values = values[1:-1]
            if values.shape != (model.config.nz - 1,) or not np.all(np.isfinite(values)):
                raise ValueError("Updated profile must have nz-1 or nz+1 finite values")
            return values

        u_new, v_new = interior(self.u), interior(self.v)
        weights = None
        if self.du != 0 or self.dv != 0:
            if self.h_obs is None or self.sigma is None or not isfinite(self.sigma) or self.sigma <= 0:
                raise ValueError("Wind increments require h_obs and positive sigma")
            if not 0 <= self.h_obs <= model.config.lz:
                raise ValueError("h_obs must be within the model domain")
            weights = np.exp(-0.5 * ((model.z[1:-1] - self.h_obs) / self.sigma)**2)

        if u_new is not None:
            s.u[1:-1] = u_new
        if v_new is not None:
            s.v[1:-1] = v_new
        if weights is not None:
            s.u[1:-1] += self.du * weights
            s.v[1:-1] += self.dv * weights
        if self.ug is not None:
            s.ug = float(self.ug)
        if self.vg is not None:
            s.vg = float(self.vg)
        if self.K is not None:
            s.K = float(self.K)
        model.apply_boundaries()


class Simulation:
    """Schedule updates, record profiles, and branch independent forecasts."""

    def __init__(self, model):
        self.model = model
        self.events = {}       # {time: [StateUpdater, ...]}
        self.history = [model.snapshot()]
        self.analyses = {}     # {time: {'background': state, 'analysis': state}}

    def schedule(self, time, update):
        if time < self.model.state.time - 1e-9:
            raise ValueError("Cannot schedule an update in the past")
        self.events.setdefault(float(time), []).append(update)
        return self

    def copy(self):
        """Independent branch including checkpoint, history, and planned updates."""
        return deepcopy(self)

    def _record(self):
        state = self.model.snapshot()
        if abs(self.history[-1].time - state.time) < 1e-9:
            self.history[-1] = state   # analysis replaces prior at the same time
        else:
            self.history.append(state)

    def update_step(self, update):
        """Immediately apply an update; save both background and analysis."""
        t = self.model.state.time
        # Preserve the earliest background if several updates occur at one time.
        prior = self.analyses[t]["background"] if t in self.analyses else self.model.snapshot()
        update.update_step(self.model)
        self.analyses[t] = {"background": prior, "analysis": self.model.snapshot()}
        self._record()
        return self

    def _apply_updates(self):
        t = self.model.state.time
        events = [when for when in self.events if abs(when - t) < 1e-9]
        for when in events:
            for update in self.events.pop(when):
                self.update_step(update)
        return bool(events)

    def run(self, until):
        """Integrate to `until`, recording full profiles on exact output times."""
        if not isfinite(until) or until < self.model.state.time - 1e-9:
            raise ValueError("End time must be finite and not in the past")
        self._apply_updates()
        interval = self.model.config.output_interval
        while self.model.state.time < until - 1e-9:
            t = self.model.state.time
            next_output = (floor((t + 1e-9) / interval) + 1) * interval
            next_event = min((x for x in self.events if x > t + 1e-9), default=inf)
            target = min(float(until), next_output, next_event)
            dt = min(self.model.dt, target - t)
            self.model.step(dt)
            if abs(self.model.state.time - target) < 1e-9:
                self.model.state.time = target
            updated = self._apply_updates()
            if updated or abs(self.model.state.time - next_output) < 1e-9 or abs(self.model.state.time - until) < 1e-9:
                self._record()
        return self.dataset()

    def dataset(self):
        """Return full velocity profiles as an xarray.Dataset."""
        states = self.history
        coords = {
            "time": ("time", [s.time for s in states]),
            "z": ("z", self.model.z),
            "timestep": ("time", [s.timestep for s in states]),
        }
        for name, field in (("u_g", "ug"), ("v_g", "vg"), ("K", "K")):
            values = np.array([getattr(s, field) for s in states])
            coords[name] = float(values[0]) if np.all(values == values[0]) else ("time", values)
        ds = xr.Dataset(
            {"u": (("time", "z"), np.stack([s.u for s in states])),
             "v": (("time", "z"), np.stack([s.v for s in states]))},
            coords=coords,
            attrs={"fcor": self.model.config.fcor,
                   "output_interval_seconds": self.model.config.output_interval,
                   "spinup_seconds": self.model.config.spinup_seconds},
        )
        for name, units in {"time": "s", "z": "m", "u": "m s-1", "v": "m s-1",
                            "u_g": "m s-1", "v_g": "m s-1", "K": "m2 s-1"}.items():
            ds[name].attrs["units"] = units
        return ds


class EkmanEns:
    """Ensemble of independent Ekman models with shared output times."""

    def __init__(self, ug, vg=0.0, K=2.0, config=None):
        # Scalars are broadcast; equal-length arrays are paired memberwise.
        arrays = [np.atleast_1d(np.asarray(x, dtype=float)) for x in (ug, vg, K)]
        n = max(map(len, arrays))
        if any(len(a) not in (1, n) for a in arrays):
            raise ValueError("ug, vg, K must be scalars or arrays of equal length")
        ug_vals, vg_vals, K_vals = [np.repeat(a, n) if len(a) == 1 else a for a in arrays]
        self.models = [Ekman(u, v, k, config) for u, v, k in zip(ug_vals, vg_vals, K_vals)]
        self.simulations = None

    @property
    def n_members(self):
        return len(self.models)

    def spin_up(self):
        """Spin up members independently; forecast time starts at t0=0."""
        if self.simulations is not None:
            raise ValueError("Ensemble was already initialized")
        for model in self.models:
            model.spin_up()
        self.simulations = [Simulation(m) for m in self.models]
        return self

    def _sims(self):
        if self.simulations is None:
            raise ValueError("Call ens.spin_up() before integrating or updating")
        times = [m.state.time for m in self.models]
        if not np.allclose(times, times[0], rtol=0, atol=1e-8):
            raise ValueError("Ensemble members must be at the same physical time")
        return self.simulations

    def _updates(self, updates):
        if isinstance(updates, StateUpdater):
            return [updates] * self.n_members
        if len(updates) != self.n_members or not all(isinstance(x, StateUpdater) for x in updates):
            raise ValueError("Supply one StateUpdater or one per ensemble member")
        return updates

    def schedule(self, time, update):
        """Schedule the same update, or member-specific updates, at one time."""
        for sim, item in zip(self._sims(), self._updates(update)):
            sim.schedule(time, item)
        return self

    def update_step(self, update):
        """Apply an immediate update to all members and record the analysis."""
        for sim, item in zip(self._sims(), self._updates(update)):
            sim.update_step(item)
        return self

    def run(self, until):
        """Integrate each member to the same physical time."""
        for sim in self._sims():
            sim.run(until)
        return self.dataset()

    def current(self):
        """Current profiles, parameters and scalar time (no history)."""
        self._sims()
        return xr.Dataset(
            {
                "u": (("ensemble", "z"), np.stack([m.state.u for m in self.models])),
                "v": (("ensemble", "z"), np.stack([m.state.v for m in self.models])),
                "u_g": ("ensemble", [m.state.ug for m in self.models]),
                "v_g": ("ensemble", [m.state.vg for m in self.models]),
                "K": ("ensemble", [m.state.K for m in self.models]),
            },
            coords={"ensemble": np.arange(1, self.n_members + 1),
                    "z": self.models[0].z,
                    "time": self.models[0].state.time},
        )

    def dataset(self):
        """Recorded histories, with one shared 1D time coordinate."""
        dss = [sim.dataset() for sim in self._sims()]
        times = dss[0].time.values
        if any(not np.array_equal(ds.time.values, times) for ds in dss[1:]):
            raise ValueError("Member output times differ; use the same output and event times")
        return xr.concat(dss, dim=xr.IndexVariable("ensemble", np.arange(1, self.n_members + 1)),
                         join="exact", coords="all", compat="override")

    @staticmethod
    def _variables(variables):
        if isinstance(variables, str):
            variables = (variables,)
        variables = tuple(variables)
        if not variables or len(set(variables)) != len(variables) or not set(variables) <= {"u", "v", "ug", "vg", "K"}:
            raise ValueError("Choose unique variables from u, v, ug, vg, K")
        return variables

    def state_vector(self, variables=("u", "v", "ug", "vg", "K")):
        """Return (N, D) matrix; u and v include ONLY interior grid levels."""
        self._sims()
        variables = self._variables(variables)
        return np.stack([
            np.concatenate([
                getattr(m.state, name)[1:-1].copy() if name in ("u", "v")
                else np.array([getattr(m.state, name)], dtype=float)
                for name in variables
            ])
            for m in self.models
        ])

    def set_state_vector(self, X, variables=("u", "v", "ug", "vg", "K")):
        """Write EnKF posterior (N,D) or (N,*,*) back into model interiors."""
        self._sims()
        variables = self._variables(variables)
        X = np.asarray(X, dtype=float)
        dofs = sum(self.models[0].config.nz - 1 if v in ("u", "v") else 1 for v in variables)
        if X.ndim not in (2, 3) or X.shape[0] != self.n_members or X.size != self.n_members * dofs:
            raise ValueError(f"Expected posterior with {self.n_members} members and {dofs} values each")
        X = X.reshape(self.n_members, dofs)
        if not np.all(np.isfinite(X)):
            raise ValueError("Posterior must contain finite values")
        if "K" in variables:
            k_pos = sum(self.models[0].config.nz - 1 if v in ("u", "v") else 1
                        for v in variables[:variables.index("K")])
            if np.any(X[:, k_pos] <= 0):
                raise ValueError("Posterior K must be positive")

        updates = []
        for row in X:
            values = {}
            start = 0
            for name in variables:
                size = self.models[0].config.nz - 1 if name in ("u", "v") else 1
                values[name] = row[start:start + size].copy() if name in ("u", "v") else float(row[start])
                start += size
            updates.append(StateUpdater(**values))
        self.update_step(updates)
        return self

    def copy(self):
        """Independent ensemble branch with recorded histories and pending events."""
        return deepcopy(self)


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
