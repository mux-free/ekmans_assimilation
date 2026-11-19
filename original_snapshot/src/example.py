"""Run independent forecasts from the same assimilation checkpoint."""
from pathlib import Path

from ekman import Ekman, EkmanConfig, Simulation, StateUpdater, rmse

config = EkmanConfig(output_interval=60)
model = Ekman(ug=10, vg=0, K=2, config=config)
model.spin_up()
StateUpdater(du=1, dv=0.5, h_obs=100, sigma=25).apply(model)

sim = Simulation(model)
background = sim.run(until=1800)         # Background forecast to assimilation

control = sim.copy()                     # Independent branches
analysis = sim.copy()
analysis.schedule(1800, StateUpdater(K=5, ug=12))

control_ds = control.run(until=7200)
analysis_ds = analysis.run(until=7200)

print("Background/analysis at assimilation:")
record = analysis.analyses[1800]
print("K:", record["background"].K, "->", record["analysis"].K)
print("RMSE vs. control:", rmse(analysis_ds.u, control_ds.u).item(), "m/s")

out = Path(__file__).resolve().parent / "data"
out.mkdir(exist_ok=True)
control_ds.to_netcdf(out / "control.nc")
analysis_ds.to_netcdf(out / "analysis.nc")
print("Saved output to", out)
