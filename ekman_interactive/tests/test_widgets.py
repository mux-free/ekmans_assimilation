import asyncio
import pytest

pytest.importorskip("ipywidgets")
from src.widgets import ExperimentPanel
from src import EkmanExperiment, EkmanConfig


def make_panel():
    e = EkmanExperiment.from_yaml("configs/config.yaml", EkmanConfig(nz=10, spinup_seconds=0))
    return ExperimentPanel(e)


def test_widget_prepare_assimilate_and_render():
    panel = make_panel()
    panel._prepare(synthetic=True)
    assert panel.pending is not None
    assert not panel.assim_button.disabled
    panel._assimilate()
    assert len(panel.experiment.analyses) == 1
    assert panel.assim_button.disabled
    panel.values.value = "1, 2, 3, 4, 5, 6, 7, 8"
    panel._prepare(synthetic=False)
    assert panel.pending.values.tolist() == [1, 2, 3, 4, 5, 6, 7, 8]


def test_widget_async_run():
    async def run():
        panel = make_panel()
        panel.until.value = 0.01
        panel.run_button.click()
        await panel.task
        assert panel.experiment.time == 36
        assert panel.stop_button.disabled
        assert "Paused" in panel.status.value
    asyncio.run(run())
