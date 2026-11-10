
import argparse
import math
from pathlib import Path

import jax.numpy as jnp
import xarray as xr




class Ekman:
    def __init__(
        self,
        nz: int,
        lz: float,
        fcor: float = 1e-4,
        spinup_seconds: int = 60*3600,
        dt_write: int = 60 # 
    ):

        NZ = nz 
        LZ = lz 
        FCOR = fcor
        spinup_seconds = spinup_seconds     # SpinUp period in seconds
        dt_write = dt_write                 # Write outputs every 60 seconds


    def weighted_kick(self, z, h_obs, sigma):
        """Gaussian perturbation centered at h_obs."""
        return np.exp(-0.5 * ((z - h_obs) / sigma) ** 2)





NZ = 100
LZ = 500.0
FCOR = 1.0e-4
VG = 0.0

SPINUP_SECONDS = 60 * 3600
TIMESERIES_INTERVAL = 60.0
DT_MAX = 5.0


def weighted_kick(z, h_obs, sigma):
    """Gaussian perturbation centered at h_obs."""
    return np.exp(-0.5 * ((z - h_obs) / sigma) ** 2)


def advance(u, v, dt, K, dz, ug, vg, fcor):
    """Advance both velocity components one time step."""


    du = (
        K * (u[2:] - 2 * u[1:-1] + u[:-2]) / dz**2 + fcor * (v[1:-1] - vg)
        )

    dv = (
        K * (v[2:] - 2 * v[1:-1] + v[:-2]) / dz**2 - fcor * (u[1:-1] - ug)
    )

    u[1:-1] += dt * du
    v[1:-1] += dt * dv


def run(
    expnr: int,
    duration: float,
    ug: float,
    K: float,
    ukick: float,
    vkick: float,
    h_obs: float,
    sigma: float,
    config_dict: dict | None = None,
    save_data: bool = True,
    vg: float | None = None,
) -> xr.Dataset:

    # ---------------------------------
    # 1. Configuration
    # ---------------------------------

    cfg = {
        "NZ": NZ,
        "LZ": LZ,
        "FCOR": FCOR,
        "VG": VG,
        "SPINUP_SECONDS": SPINUP_SECONDS,
        "TIMESERIES_INTERVAL": TIMESERIES_INTERVAL,
        "DT_MAX": DT_MAX,
    }

    if config_dict is not None:
        cfg.update(config_dict)

    nz = cfg["NZ"]
    lz = float(cfg["LZ"])
    fcor = float(cfg["FCOR"])
    vg = float(cfg["VG"] if vg is None else vg)
    spinup = float(cfg["SPINUP_SECONDS"])
    output_interval = float(cfg["TIMESERIES_INTERVAL"])
    dt_max = float(cfg["DT_MAX"])

    # ---------------------------------
    # 2. Validation
    # ---------------------------------

    if not isinstance(nz, (int, np.integer)) or isinstance(nz, bool) or nz < 2:
        raise ValueError("NZ must be an integer >= 2")
    if duration < 0 or not np.isfinite(duration):
        raise ValueError("duration must be finite and nonnegative")
    if K <= 0 or not np.isfinite(K):
        raise ValueError("K must be finite and positive")
    if sigma <= 0 or not np.isfinite(sigma):
        raise ValueError("sigma must be finite and positive")
    if lz <= 0 or not np.isfinite(lz):
        raise ValueError("LZ must be finite and positive")
    if not np.isfinite(h_obs) or not 0 <= h_obs <= lz:
        raise ValueError(f"h_obs must be between 0 and {lz}")
    if output_interval <= 0 or not np.isfinite(output_interval):
        raise ValueError("TIMESERIES_INTERVAL must be positive")
    if dt_max <= 0 or not np.isfinite(dt_max):
        raise ValueError("DT_MAX must be positive")
    if spinup < 0 or not np.isfinite(spinup):
        raise ValueError("SPINUP_SECONDS must be nonnegative")

    if not all(np.isfinite(x) for x in (ug, vg, ukick, vkick, fcor)):
        raise ValueError("Wind parameters and FCOR must be finite")

    # ---------------------------------
    # 3. Grid and time step
    # ---------------------------------

    dz = lz / nz
    z = np.arange(nz + 1) * dz
    dt = min(0.45 * dz**2 / K, dt_max)

    # ---------------------------------
    # 4. Initial conditions
    # ---------------------------------

    u = ug * z / lz
    v = np.full(nz + 1, vg)

    u[0], v[0] = 0.0, 0.0
    u[-1], v[-1] = ug, vg

    # ---------------------------------
    # 5. Spin-up
    # ---------------------------------

    t = 0.0

    while t < spinup:
        step = min(dt, spinup - t)
        advance(u, v, step, K, dz, ug, vg, fcor)
        t += step
    
    # ---------------------------------
    # 6. Gaussian perturbation
    # ---------------------------------

    weights = weighted_kick(z[1:-1], h_obs, sigma)

    u[1:-1] += ukick * weights
    v[1:-1] += vkick * weights

    # ---------------------------------
    # 7. Main simulation
    # ---------------------------------

    saved_steps = [0]
    saved_times = [0.0]
    saved_u = [u.copy()]
    saved_v = [v.copy()]
    next_output = output_interval

    t = 0.0
    timestep = 0

    while t < duration:
        step = min(dt, duration - t)

        advance(u, v, step, K, dz, ug, vg, fcor)

        t += step
        timestep += 1

        if t >= next_output - 1e-9 or t >= duration:
            saved_steps.append(timestep)
            saved_times.append(t)
            saved_u.append(u.copy())
            saved_v.append(v.copy())

            while next_output <= t + 1e-9:
                next_output += output_interval

    # ---------------------------------
    # 8. Create xarray Dataset
    # ---------------------------------

    steps = np.asarray(saved_steps, dtype=np.int64)

    ds = xr.Dataset(
        data_vars={
            "u": (
                ("time", "z"),
                np.stack(saved_u)
            ),
            "v": (
                ("time", "z"),
                np.stack(saved_v)
            ),
        },
        coords={
            "time": ("time", np.asarray(saved_times)),
            "z": ("z", z),
            "timestep": ("time", np.asarray(saved_steps, dtype=np.int64)),
            "u_g": float(ug),
            "v_g": float(vg),
            "K": float(K),
        },
        attrs={
            "description": "1D Ekman model",
            "expnr": int(expnr),
            "dt": float(dt),
            "dz": float(dz),
            "fcor": fcor,
            "spinup_seconds": spinup,
            "output_interval_seconds": output_interval,
            "ukick": float(ukick),
            "vkick": float(vkick),
            "h_obs": float(h_obs),
            "sigma": float(sigma),
        },
    )

    # Variable metadata
    ds["time"].attrs["units"] = "s"
    ds["z"].attrs["units"] = "m"
    ds["u"].attrs["units"] = "m s-1"
    ds["v"].attrs["units"] = "m s-1"
    ds["u_g"].attrs["units"] = "m s-1"
    ds["v_g"].attrs["units"] = "m s-1"
    ds["K"].attrs["units"] = "m2 s-1"

    # ---------------------------------
    # 9. Save NetCDF (optional)
    # ---------------------------------

    if save_data:
        data_dir = Path(__file__).resolve().parent.parent / "data"
        data_dir.mkdir(parents=True, exist_ok=True)

        output_path = data_dir / f"ekman.{expnr:03d}.nc"

        ds.to_netcdf(output_path)
        print(f"Saved {output_path}")

    return ds


def main(argv: list[str] | None = None) -> xr.Dataset:

    parser = argparse.ArgumentParser()

    parameters = {
        "expnr": int,
        "duration_in_seconds": float,
        "ug": float,
        "K": float,
        "ukick": float,
        "vkick": float,
        "h_obs": float,
        "sigma": float,
    }

    for name, dtype in parameters.items():
        parser.add_argument(name, nargs="?", type=dtype)

        flags = [f"--{name}"]

        if name == "duration_in_seconds":
            flags.insert(0, "--duration")

        parser.add_argument(
            *flags,
            dest=f"{name}_option",
            type=dtype,
        )

    parser.add_argument("--vg", type=float, default=None)
    parser.add_argument("--no-save", action="store_true")

    args = parser.parse_args(argv)
    values = {}

    for name in parameters:
        positional = getattr(args, name)
        named = getattr(args, f"{name}_option")

        if positional is not None and named is not None:
            parser.error(f"{name} provided twice")

        if positional is None and named is None:
            parser.error(f"missing required argument: {name}")

        values[name] = named if named is not None else positional

    return run(
        expnr=values["expnr"],
        duration=values["duration_in_seconds"],
        ug=values["ug"],
        K=values["K"],
        ukick=values["ukick"],
        vkick=values["vkick"],
        h_obs=values["h_obs"],
        sigma=values["sigma"],
        vg=args.vg,
        save_data=not args.no_save,
    )


if __name__ == "__main__":
    main()
