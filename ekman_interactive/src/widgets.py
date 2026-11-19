"""Optional Jupyter controls. Install with: pip install -e '.[notebook]'"""
import asyncio
import numpy as np
import matplotlib.pyplot as plt
import ipywidgets as widgets
from IPython.display import display, clear_output


def plot_experiment(experiment, observation=None):
    """Current profiles and evolving parameter ensembles, with the no-DA control."""
    e = experiment
    ds = e.datasets()
    fig, axes = plt.subplots(1, 4, figsize=(15, 4), layout="constrained")
    for ax, variable in zip(axes[:2], ("u", "v")):
        ax.plot(getattr(e.ensemble, variable).T, e.ensemble.z, color="#4782b8", alpha=0.18, lw=0.7)
        ax.plot(getattr(e.ensemble, variable).mean(axis=0), e.ensemble.z, color="#1464a0", label="Analysis ensemble")
        ax.plot(getattr(e.truth, variable)[0], e.truth.z, color="#18855e", label="Truth")
        ax.plot(getattr(e.control, variable).mean(axis=0), e.control.z, "--", color="#555555", label="No DA")
        if observation is not None and variable in observation.variables and abs(observation.time-e.time)<1e-8:
            start = observation.variables.index(variable)*len(observation.heights)
            ax.scatter(observation.values[start:start+len(observation.heights)], observation.heights,
                       color="#d66a1e", marker="x", zorder=4, label="Observation")
        ax.set(xlabel=f"{variable} [m/s]", ylabel="Height [m]")
        ax.grid(alpha=0.2)
    axes[0].legend(fontsize=7)
    for ax, variable, label in zip(axes[2:], ("u_g", "K"), ("ug [m/s]", "K [m²/s]")):
        ens, truth, control = [ds[name][variable] for name in ("ensemble", "truth", "control")]
        time = ds["ensemble"].time.values/3600
        ax.plot(time, ens.T, color="#4782b8", alpha=0.18, lw=0.7)
        ax.plot(time, ens.mean("ensemble"), color="#1464a0")
        ax.plot(time, truth, color="#18855e")
        ax.plot(time, control.mean("ensemble"), "--", color="#555555")
        for record in e.analyses:
            ax.axvline(record.time/3600, color="#aaaaaa", lw=0.6)
        ax.set(xlabel="Hours after spin-up", ylabel=label)
        ax.grid(alpha=0.2)
    fig.suptitle(f"t = {e.time/3600:.3f} h  |  {len(e.analyses)} analysis steps")
    return fig


class ExperimentPanel:
    """Thin UI over the controller; numerical logic stays in EkmanExperiment."""
    def __init__(self, experiment):
        self.experiment = experiment
        self.pending = None
        self.task = None
        p, en = experiment.sim_config.parameters, experiment.sim_config.ensemble
        self.settings = {name: widgets.FloatText(value=value, description=name)
                         for name, value in {**{k: p[k] for k in ("ug_truth", "K_truth", "ug_error", "K_error")},
                                             **{k: en[k] for k in ("sigma_ug", "spread_K")}}.items()}
        self.until = widgets.FloatText(value=1, description="Until [h]")
        self.heights = widgets.Text(value="100, 200, 300, 400", description="Heights [m]")
        self.obs_vars = widgets.SelectMultiple(options=["u", "v"], value=("u", "v"), description="Observe", rows=2)
        self.analysis_vars = widgets.SelectMultiple(options=["ug", "K", "u", "v"], value=("ug",), description="Update", rows=4)
        self.std = widgets.FloatText(value=0.25, description="Obs SD [m/s]")
        self.values = widgets.Text(value="", description="Values", placeholder="u at each height, then v at each height",
                                   layout=widgets.Layout(width="650px"))
        self.reset_button = widgets.Button(description="Apply & reset")
        self.run_button = widgets.Button(description="Run to time", button_style="primary")
        self.stop_button = widgets.Button(description="Stop run", disabled=True)
        self.sample_button = widgets.Button(description="Sample truth")
        self.manual_button = widgets.Button(description="Use entered values")
        self.assim_button = widgets.Button(description="Assimilate", button_style="success", disabled=True)
        self.status = widgets.HTML(value="Ready. Times are hours after spin-up.")
        self.output = widgets.Output()
        self._buttons = [self.reset_button, self.run_button, self.sample_button, self.manual_button, self.assim_button]
        self._inputs = [*self.settings.values(), self.until, self.heights, self.obs_vars, self.analysis_vars, self.std, self.values]
        self.reset_button.on_click(lambda _: self._launch(self._reset))
        self.run_button.on_click(lambda _: self._launch(lambda: experiment.run_until(self.until.value*3600), allow_stop=True))
        self.stop_button.on_click(lambda _: experiment.request_stop())
        self.sample_button.on_click(lambda _: self._prepare(synthetic=True))
        self.manual_button.on_click(lambda _: self._prepare(synthetic=False))
        self.assim_button.on_click(lambda _: self._assimilate())
        self.widget = widgets.VBox([
            widgets.HTML("<b>Ekman parameter and wind assimilation</b>"),
            widgets.HBox(list(self.settings.values())[:3]), widgets.HBox(list(self.settings.values())[3:]),
            self.reset_button, widgets.HBox([self.until, self.run_button, self.stop_button]),
            widgets.HBox([self.heights, self.std, self.obs_vars, self.analysis_vars]), self.values,
            widgets.HBox([self.sample_button, self.manual_button, self.assim_button]), self.status, self.output])

    def _reset(self):
        self.experiment.set_parameters(**{k: v.value for k, v in self.settings.items()})
        self.experiment.reset()

    def _launch(self, operation, allow_stop=False):
        async def execute():
            for item in (*self._buttons, *self._inputs):
                item.disabled = True
            self.stop_button.disabled = not allow_stop
            self.status.value = "Running…"
            try:
                await asyncio.to_thread(operation)
                self.pending = None
                self.refresh()
                self.status.value = f"Paused at {self.experiment.time/3600:.4f} h. Ready for an observation."
            except Exception as exc:
                self.status.value = f"Error: {str(exc)}"
            finally:
                for item in (*self._buttons, *self._inputs):
                    item.disabled = False
                self.stop_button.disabled = True
                self.assim_button.disabled = self.pending is None
        if self.task is not None and not self.task.done():
            return
        self.task = asyncio.get_running_loop().create_task(execute())

    def _prepare(self, synthetic):
        try:
            heights = [float(s.strip()) for s in self.heights.value.split(",")]
            values = None if synthetic else [float(s.strip()) for s in self.values.value.split(",")]
            self.pending = self.experiment.observe(values, heights=heights,
                variables=self.obs_vars.value, std=self.std.value)
            self.values.value = ", ".join(f"{v:.8g}" for v in self.pending.values)
            self.assim_button.disabled = False
            self.status.value = "Observation prepared. Editing values requires ‘Use entered values’ before assimilating."
            self.refresh()
        except Exception as exc:
            self.pending = None
            self.assim_button.disabled = True
            self.status.value = f"Error: {str(exc)}"

    def _assimilate(self):
        try:
            if self.pending is None:
                raise ValueError("Prepare a new independent observation first")
            self.experiment.assimilate(self.pending, variables=self.analysis_vars.value)
            self.pending = None
            self.assim_button.disabled = True
            self.status.value = "Assimilated. Continue the forecast or prepare a new independent observation."
            self.refresh()
        except Exception as exc:
            self.status.value = f"Error: {str(exc)}"

    def refresh(self):
        with self.output:
            clear_output(wait=True)
            fig = plot_experiment(self.experiment, self.pending)
            display(fig)
            plt.close(fig)

    def display(self):
        display(self.widget)
        self.refresh()
        return self
