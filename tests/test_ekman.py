"""Small tests: run with `pytest -q` from the project directory."""
import numpy as np
import pytest

from ekman import Ekman, EkmanConfig, Simulation, StateUpdater, bias, rmse, vector_rmse


def make_sim():
    model = Ekman(config=EkmanConfig(nz=20, lz=500, spinup_seconds=0, output_interval=7))
    return Simulation(model)


def test_initial_and_final_profiles():
    sim = make_sim()
    ds = sim.run(22)
    assert ds.sizes["z"] == 21
    np.testing.assert_allclose(ds.time.values, [0, 7, 14, 21, 22])
    np.testing.assert_allclose(ds.u.isel(z=0), 0)
    np.testing.assert_allclose(ds.u.isel(z=-1), 10)
    assert ds.K.ndim == 0


def test_branch_and_parameter_update():
    sim = make_sim()
    sim.run(7)
    reference = sim.copy()
    modified = sim.copy()
    modified.schedule(7, StateUpdater(K=4, ug=12, du=1, h_obs=100, sigma=25))
    ds_mod = modified.run(21)
    ds_ref = reference.run(21)
    assert sim.model.state.K == 2
    assert reference.model.state.K == 2
    assert modified.model.state.K == 4
    assert ds_mod.K.dims == ("time",)
    assert modified.analyses[7]["background"].ug == 10
    assert modified.analyses[7]["analysis"].ug == 12
    np.testing.assert_allclose(ds_mod.u.sel(time=7).isel(z=-1), 12)
    assert ds_ref.u.sel(time=7).isel(z=-1).item() == 10


def test_restart_without_update():
    direct = make_sim()
    ds_direct = direct.run(21)
    split = make_sim()
    split.run(7)
    ds_split = split.copy().run(21)
    np.testing.assert_allclose(ds_direct.u, ds_split.u, atol=1e-12)
    np.testing.assert_array_equal(ds_direct.time, ds_split.time)


def test_metrics():
    a = make_sim().run(7)
    b = a.copy(deep=True)
    b["u"] = b.u + 2
    np.testing.assert_allclose(rmse(b.u, a.u), 2)
    np.testing.assert_allclose(bias(b.u, a.u), 2)
    np.testing.assert_allclose(vector_rmse(b, a), 2)


def test_invalid_diffusivity():
    with pytest.raises(ValueError):
        Ekman(K=0)
    sim = make_sim()
    sim.schedule(5, StateUpdater(K=-1))
    with pytest.raises(ValueError):
        sim.run(10)


def test_off_grid_assimilation_and_netcdf(tmp_path):
    sim = make_sim()
    sim.schedule(17.3, StateUpdater(K=5, vg=2))
    ds = sim.run(22.7)
    np.testing.assert_allclose(ds.time.values, [0, 7, 14, 17.3, 21, 22.7])
    assert ds.v_g.sel(time=17.3).item() == 2
    assert sim.model.dt == min(0.45 * sim.model.dz**2 / 5, sim.model.config.dt_max)
    path = tmp_path / "test.nc"
    ds.to_netcdf(path)
    import xarray as xr
    with xr.open_dataset(path) as ds_loaded:
        np.testing.assert_allclose(ds_loaded.u, ds.u)
        np.testing.assert_allclose(ds_loaded.time, ds.time)


def test_metric_rejects_different_coordinates():
    a = make_sim().run(7).u
    b = a.assign_coords(time=a.time + 1)
    with pytest.raises(ValueError):
        rmse(a, b)
