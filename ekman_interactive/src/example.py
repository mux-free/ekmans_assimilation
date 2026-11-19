"""Reproducible six-cycle example: python -m src.example (from project root)."""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from .experiment import EkmanExperiment
from .ekman import EkmanConfig


def main():
    experiment = EkmanExperiment.from_yaml("configs/config.yaml", EkmanConfig(), seed=14)
    initial_K = experiment.ensemble.K.copy()
    for time in (3600, 5400, 7200, 9000, 10800, 12600):
        experiment.run_until(time)
        observation = experiment.observe(heights=[100, 300], variables=("u", "v"))
        experiment.assimilate(observation, variables=("ug",))
        np.testing.assert_array_equal(experiment.ensemble.K, initial_K)
    experiment.run_until(16200)
    data = experiment.datasets()
    output = Path("outputs")
    output.mkdir(exist_ok=True)
    for name, dataset in data.items():
        dataset.to_netcdf(output/f"{name}.nc", engine="scipy")
    # Plot without requiring the optional widget dependency.
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), layout="constrained")
    for name, color, style in (("ensemble", "#1c6eab", "-"), ("control", "#666666", "--")):
        ds, truth = data[name], data["truth"]
        mean = ds.mean("ensemble")
        interior = dict(z=slice(1, -1))
        error = np.sqrt(((mean.u-truth.u)**2 + (mean.v-truth.v)**2).isel(**interior).mean("z"))
        axes[0].plot(ds.time/3600, error, style, color=color, label=name)
        axes[1].plot(ds.time/3600, mean.u_g, style, color=color)
        axes[2].plot(ds.time/3600, mean.K, style, color=color)
    axes[1].axhline(experiment.truth.ug[0], color="#23855b", label="truth")
    axes[2].axhline(experiment.truth.K[0], color="#23855b")
    for ax, ylabel in zip(axes, ("Interior vector RMSE [m/s]", "Mean ug [m/s]", "Mean K [m²/s]")):
        for record in experiment.analyses:
            ax.axvline(record.time/3600, color="#bbbbbb", lw=0.6)
        ax.set(xlabel="Hours after spin-up", ylabel=ylabel)
        ax.grid(alpha=0.15)
    axes[0].legend()
    axes[1].legend()
    fig.suptitle("One seeded illustration: uncertain ug and K, updating ug only")
    fig.savefig(output/"ug_only_demo.png", dpi=160)
    plt.close(fig)
    print(f"Completed {len(experiment.analyses)} cycles to t={experiment.time:g} s; K unchanged memberwise.")
    print(f"Initial/final ug means: {float(data['ensemble'].u_g.isel(time=0).mean()):.4f} / {float(experiment.ensemble.ug.mean()):.4f}")
    print("Saved outputs under", output.resolve())


if __name__ == "__main__":
    main()
