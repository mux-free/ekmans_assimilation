
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation



def plot_rmse(rmse_dict, ax=None, figsize=(8, 4)):
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    for label, da in rmse_dict.items():
        color = ax._get_lines.get_next_color()

        if "ensemble" in da.dims:
            ax.plot(da.time, da.T, color=color, lw=0.7, alpha=0.25)
            ax.plot([], [], color=color, lw=2, label=label)
        else:
            ax.plot(da.time, da, color=color, lw=2, label=label)

    ax.set(xlabel="Time [s]", ylabel="RMSE")
    ax.grid(alpha=0.25)
    ax.legend()
    
    ax.set_ylim(0,None)

    rmse_da_tot = np.sqrt((rmse_dict['EnsMean']**2).mean())
    ax.set_title(f"Total rmse (DA): {rmse_da_tot:.2f}")

    return ax


def plot_timeseries(da, da_true, da_ctrl, zlev="avg", var="u",
                    ax=None, figsize=(8, 4)):
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    # Select depth or average over depth
    if zlev in ("mean", "avg"):
        da = da.mean(dim="z")
        da_true = da_true.mean(dim="z")
        da_ctrl = da_ctrl.mean(dim="z")
        ylabel = f"{var}_avg [m/s]"
    elif isinstance(zlev, (int, float)):
        da = da.sel(z=zlev)
        da_true = da_true.sel(z=zlev)
        da_ctrl = da_ctrl.sel(z=zlev)
        ylabel = f"{var}(z={da.z.item():.0f}m) [m/s]"

    # Select variable if inputs are Datasets
    if isinstance(da, xr.Dataset):
        da = da[var]
        da_true = da_true[var]
        da_ctrl = da_ctrl[var]

    # Ensemble members
    ax.plot(da.time, da.T, c="k", lw=0.5, alpha=0.5)
    ax.plot([], [], c="k", lw=0.5, alpha=0.5, label="EnsMember")

    # Ensemble mean, truth, and control
    ax.plot(da.time, da.mean("ensemble"), c="tab:orange",
            lw=2, label="EnsMean")
    ax.plot(da_true.time, da_true, c="tab:green",
            lw=2, label="Truth")
    ax.plot(da_ctrl.time, da_ctrl.mean("ensemble"), c="k",
            lw=2, ls="--", label="Ctrl")

    ax.set(xlabel="Time [s]", ylabel=ylabel)
    ax.grid(alpha=0.25)
    ax.legend()

    return ax



def animate_ekman(
    data: dict[str, xr.DataArray],
    observations: tuple[np.ndarray, np.ndarray] | None = None,
    obs_time: float = 3600,
    step: int = 30,
    interval: int = 100,
    figsize: tuple = (7, 7),
    suptitle: str|None = None,
    styles: dict[str, dict] | None = None,
):
    """
    Animate vertical profiles of multiple DataArrays.

    Parameters
    ----------
    data : dict[str, xr.DataArray]
        Mapping from legend labels to DataArrays.
        Expected dimensions:
            (time, z)
            (time, ensemble, z)

        Arrays with an ensemble dimension show individual
        members and the ensemble mean.

    observations : tuple, optional
        (u_obs, obs_heights), plotted at obs_time.

    obs_time : float
        Time at which observations are shown, in seconds.

    step : int
        Frame stride.

    interval : int
        Delay between frames in milliseconds.

    styles : dict, optional
        Matplotlib line styles per dataset label.
        Example:
            {"Truth": {"color": "tab:green", "lw": 3}}
    """

    if not data:
        raise ValueError("data must contain at least one DataArray")

    styles = styles or {}

    fig, ax = plt.subplots(figsize=figsize)

    # Use the first DataArray as time reference
    reference = next(iter(data.values()))
    times = reference.time.values

    # Automatic color assignment
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    artists = {}
    all_lines = []

    for idx, (label, da) in enumerate(data.items()):

        color = colors[idx % len(colors)]
        style = {
            "color": color,
            "lw": 2.5,
            **styles.get(label, {}),
        }

        if "ensemble" in da.dims:
            # Individual ensemble members
            members = [
                ax.plot(
                    [], [],
                    color=style["color"],
                    lw=0.7,
                    alpha=0.25,
                    label=label if j == 0 else "_nolegend_",
                    zorder=1,
                )[0]
                for j in range(da.sizes["ensemble"])
            ]

            artists[label] = {
                "data": da,
                "members": members,
            }

            all_lines.extend(members)

        else:
            line, = ax.plot(
                [], [],
                **style,
                label=label,
                zorder=10,
            )

            artists[label] = {
                "data": da,
                "line": line,
            }

            all_lines.append(line)

    # Observations
    obs_line = None
    obs_idx = None

    if observations is not None:
        u_obs, obs_heights = observations

        obs_idx = int(np.argmin(np.abs(times - obs_time)))

        obs_line, = ax.plot(
            [], [], "o",
            color="tab:green",
            markersize=10,
            markeredgecolor="k",
            markeredgewidth=0.8,
            label="Observations",
            zorder=20,
        )
        all_lines.append(obs_line)

    # Axis limits
    xmin = min(float(da.min()) for da in data.values())
    xmax = max(float(da.max()) for da in data.values())

    zmin = min(float(da.z.min()) for da in data.values())
    zmax = max(float(da.z.max()) for da in data.values())

    if observations is not None:
        xmin = min(xmin, float(np.min(u_obs)))
        xmax = max(xmax, float(np.max(u_obs)))
        zmin = min(zmin, float(np.min(obs_heights)))
        zmax = max(zmax, float(np.max(obs_heights)))

    ax.set(
        xlabel="u [m/s]",
        ylabel="z [m]",
        xlim=(xmin, xmax),
        ylim=(zmin, zmax),
    )

    ax.grid(alpha=0.25)
    ax.legend(loc="upper left")

    def update(i):

        for label, item in artists.items():

            da = item["data"]

            if "members" in item:
                profile = da.isel(time=i)

                for j, line in enumerate(item["members"]):
                    line.set_data(
                        profile.isel(ensemble=j).values,
                        da.z.values,
                    )

            else:
                item["line"].set_data(
                    da.isel(time=i).values,
                    da.z.values,
                )

        if obs_line is not None:
            if i == obs_idx:
                obs_line.set_data(u_obs, obs_heights)
            else:
                obs_line.set_data([], [])

        ax.set_title(f"Time = {times[i] / 3600:.1f} h")

        return all_lines

    frames = set(range(0, len(times), step))

    if obs_idx is not None:
        frames.add(obs_idx)

    ani = FuncAnimation(
        fig,
        update,
        frames=sorted(frames),
        interval=interval,
    )

    if suptitle is not None:
        fig.suptitle(suptitle, ha="left", fontfamily="monospace", x=0.1)

    plt.close(fig)
    return ani
