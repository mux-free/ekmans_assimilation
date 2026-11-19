from copy import deepcopy
from pathlib import Path
import numpy as np
import pytest
from scipy.linalg import expm

from src import EkmanConfig, SimConfig, EkmanEns, EnsembleCreator, EkmanExperiment
from src.enkf import EnsembleKalmanFilter
from src.observation_operator import wind_matrix, predict_wind

CONFIG = Path(__file__).resolve().parents[1]/"configs/config.yaml"


def make_experiment():
    return EkmanExperiment.from_yaml(CONFIG, EkmanConfig(nz=20, spinup_seconds=0,
                                                        output_interval=7), seed=42)


def test_prior_uncertainty_independent_of_analysis_selection():
    c = SimConfig.from_yaml(CONFIG)
    c.data_assimilation["state_vars"] = ["ug"]
    one = EnsembleCreator(c, 99)()
    c.data_assimilation["state_vars"] = ["K", "u", "v"]
    two = EnsembleCreator(c, 99)()
    assert np.std(one["K_ens"]) > 0
    for key in one:
        np.testing.assert_array_equal(one[key], two[key])


def test_pause_resume_and_identical_control():
    direct, split = make_experiment(), make_experiment()
    direct.run_until(21)
    split.run_until(7)
    split.run_until(21)
    for name in ("truth", "ensemble", "control"):
        np.testing.assert_array_equal(getattr(direct, name).u, getattr(split, name).u)
    np.testing.assert_array_equal(split.ensemble.u, split.control.u)
    split.run_until(23.7)
    assert split.time == split.truth.time == split.control.time == 23.7
    for ds in split.datasets().values():
        np.testing.assert_allclose(ds.time, [0, 7, 14, 21, 23.7])


def test_cooperative_stop_and_resume():
    e = make_experiment()
    e.run_until(100, on_step=lambda exp: exp.request_stop() if exp.time >= 5 else None)
    assert e.stopped_early and e.time == 5
    assert e.truth.time == e.control.time == 5
    e.run_until(17.3)
    assert e.time == 17.3 and not e.stopped_early


def test_K_analysis_cannot_change_truth_or_control_numerics():
    e = make_experiment()
    branch = e.copy()
    branch.ensemble.update(K=100)
    e.run_until(30)
    branch.run_until(30)
    np.testing.assert_array_equal(e.truth.u, branch.truth.u)
    np.testing.assert_array_equal(e.control.u, branch.control.u)


def test_ug_only_preserves_nuisance_K_and_interior_winds():
    e = make_experiment()
    e.run_until(50)
    K, u, v = [getattr(e.ensemble, k).copy() for k in ("K", "u", "v")]
    record = e.assimilate(e.observe(heights=[100], noise=False), variables=("ug",))
    np.testing.assert_array_equal(e.ensemble.K, K)
    np.testing.assert_array_equal(e.ensemble.u[:, 1:-1], u[:, 1:-1])
    np.testing.assert_array_equal(e.ensemble.v, v)
    assert not np.array_equal(record.background.u_g, record.analysis.u_g)
    np.testing.assert_array_equal(e.ensemble.u[:, -1], e.ensemble.ug)
    np.testing.assert_array_equal(record.predicted_before, record.predicted_after)
    e.run_until(100)
    assert not np.array_equal(e.ensemble.u, e.control.u)


def test_wind_only_keeps_parameters_fixed():
    e = make_experiment()
    e.run_until(100)
    ug, K = e.ensemble.ug.copy(), e.ensemble.K.copy()
    before = e.ensemble.u.copy()
    e.assimilate(e.observe(heights=[100], noise=False), variables=("u", "v"))
    np.testing.assert_array_equal(e.ensemble.ug, ug)
    np.testing.assert_array_equal(e.ensemble.K, K)
    assert not np.array_equal(e.ensemble.u[:, 1:-1], before[:, 1:-1])


def test_repeated_assimilation_logK_and_history():
    e = make_experiment()
    for time in (60, 120, 180):
        e.run_until(time)
        e.assimilate(e.observe(heights=[100, 200]), variables=("ug", "K", "u", "v"))
        assert np.all(e.ensemble.K > 0)
    assert len(e.analyses) == 3
    assert e.analyses[0].time == 60
    assert e.analyses[0].gain.shape == (40, 4)
    e.ensemble.ug[0] += 1  # saved records are independent copies
    assert e.analyses[-1].analysis.u_g[0] != e.ensemble.ug[0]


def test_observation_order_interpolation_and_R():
    e = make_experiment()
    e.truth.update(v=np.arange(19))
    H, _, _ = wind_matrix(e.truth.z, [112.5, 237.5], ("v", "u"))
    expected = [np.interp(h, e.truth.z, getattr(e.truth, v)[0])
                for v in ("v", "u") for h in (112.5, 237.5)]
    np.testing.assert_allclose(predict_wind(e.truth, H)[0], expected)
    obs = e.observe([1, 2, 3, 4], heights=[112.5, 237.5], variables=("v", "u"), std=0.5)
    np.testing.assert_array_equal(obs.R, np.eye(4)*0.25)
    with pytest.raises(ValueError):
        e.observe(heights=[501])


def test_reuse_and_stale_observations_rejected_but_independent_same_time_allowed():
    e = make_experiment()
    obs = e.observe()
    e.assimilate(obs, variables=("ug",))
    with pytest.raises(ValueError, match="already"):
        e.assimilate(obs)
    e.assimilate(e.observe(), variables=("ug",))
    assert len(e.analyses) == 2
    old = e.observe()
    e.run_until(5)
    with pytest.raises(ValueError, match="time"):
        e.assimilate(old)


def test_config_edit_save_and_explicit_reset(tmp_path):
    e = make_experiment()
    path = tmp_path/"edited.yaml"
    e.set_parameters(ug_truth=8, K_truth=2, ug_error=-1, K_error=0.5,
                     sigma_ug=0, spread_K=0, save_to=path)
    c = SimConfig.from_yaml(path)
    assert c.parameters["ug_sim"] == 7
    assert c.parameters["K_sim"] == 2.5
    with pytest.raises(RuntimeError, match="reset"):
        e.run_until(5)
    e.reset(spinup=False)
    np.testing.assert_array_equal(e.ensemble.ug, np.full(20, 7))
    np.testing.assert_array_equal(e.ensemble.K, np.full(20, 2.5))
    assert e.truth.ug[0] == 8


def test_copy_and_reproducible_independent_noise_streams():
    e = make_experiment()
    e.run_until(5)
    branch = e.copy()
    obs, obs_branch = e.observe(), branch.observe()
    np.testing.assert_array_equal(obs.values, obs_branch.values)
    obs2 = e.observe()
    assert not np.array_equal(obs.values, obs2.values)
    e.assimilate(obs, variables=("ug",))
    branch.assimilate(obs, variables=("ug",))
    np.testing.assert_array_equal(e.ensemble.ug, branch.ensemble.ug)
    e.ensemble.ug[0] += 1
    assert e.ensemble.ug[0] != branch.ensemble.ug[0]


def test_filter_scalar_mean_matches_exact_kalman_update_and_fresh_noise():
    X = np.arange(20.)[:, None]
    Y = 2*X
    y = np.array([3.])
    f = EnsembleKalmanFilter(20, np.array([[0.25]]))
    gain = 2*np.var(X, ddof=1)/(4*np.var(X, ddof=1)+0.25)
    expected = X.mean()+gain*(3-Y.mean())
    first = f.analysis_step(X, Y, y)
    np.testing.assert_allclose(first.mean(), expected)
    second = f.analysis_step(X, Y, y)
    assert not np.array_equal(first, second)
    with pytest.raises(NotImplementedError):
        EnsembleKalmanFilter(20, np.eye(1), inflation_factor=1.1)
    with pytest.raises(NotImplementedError):
        EnsembleKalmanFilter(20, np.eye(1), localization_distance=5)


def test_invalid_analysis_is_atomic_including_rng():
    e = make_experiment()
    before = e.copy()
    obs = e.observe()
    obs.R[0, 0] = -1
    with pytest.raises(ValueError):
        e.assimilate(obs)
    np.testing.assert_array_equal(e.ensemble.state_vector(), before.ensemble.state_vector())
    assert e.filter.rng.bit_generator.state == before.filter.rng.bit_generator.state
    assert not e.analyses


def test_atomic_model_update_and_interior_state_vector():
    m = EkmanEns(ug=[8, 10], config=EkmanConfig(nz=20, spinup_seconds=0))
    X = m.state_vector(("u", "v", "K"))
    assert X.shape == (2, 39)
    before = m.state_vector()
    with pytest.raises(ValueError):
        m.update(u=5, v=np.arange(8))
    np.testing.assert_array_equal(m.state_vector(), before)
    with pytest.raises(ValueError):
        m.update(K=np.nan)
    m.set_state_vector(X, ("u", "v", "K"))
    np.testing.assert_array_equal(m.state_vector(), before)
    with pytest.raises(ValueError):
        m.set_state_vector(X.T, ("u", "v", "K"))


def test_singleton_history_dimensions_and_netcdf(tmp_path):
    m = EkmanEns()
    assert m.dataset().u.dims == ("time", "z")
    m.run(3.7)
    p = tmp_path/"result.nc"
    m.dataset().to_netcdf(p, engine="scipy")
    import xarray as xr
    with xr.open_dataset(p, engine="scipy") as ds:
        np.testing.assert_array_equal(ds.u, m.dataset().u)


@pytest.mark.parametrize("K", [0.0001, 0.1, 1, 100])
def test_euler_step_stable_for_combined_coriolis_diffusion(K):
    m = EkmanEns(K=K, config=EkmanConfig())
    a = 4*K/m.dz**2*np.sin(np.arange(1, 100)*np.pi/200)**2
    assert np.max(abs(1+m.dt*(-a+1j*m.config.fcor))) <= 1
    with pytest.raises(ValueError):
        m.step(-1)
    with pytest.raises(ValueError):
        m.run(float("nan"))


def test_euler_convergence_against_exact_semidiscrete_solution():
    # Zero boundary/forcing permits an exact matrix-exponential reference.
    n, dz, K, f = 9, 10, 2, 1e-4
    L = (np.diag(np.full(n, -2.))+np.diag(np.ones(n-1), 1)+np.diag(np.ones(n-1), -1))*K/dz**2
    A = np.block([[L, f*np.eye(n)], [-f*np.eye(n), L]])
    u0 = np.sin(np.arange(1, n+1)*np.pi/(n+1))
    x0 = np.r_[u0, u0/2]
    reference = expm(A*100) @ x0
    errors = []
    for dt in (1, 0.5):
        m = EkmanEns(ug=0, K=K, config=EkmanConfig(nz=10, lz=100, dt_max=dt,
                                                 spinup_seconds=0, output_interval=100))
        m.update(u=u0, v=u0/2)
        m.run(100)
        errors.append(np.linalg.norm(m.state_vector(("u", "v"))[0]-reference))
    assert 1.95 < errors[0]/errors[1] < 2.05


def test_long_run_approaches_finite_domain_steady_solution():
    # Compare to the exact steady solution of the SAME discrete equations.
    m = EkmanEns(config=EkmanConfig(nz=10, lz=100, dt_max=5,
                                  spinup_seconds=50000, output_interval=100))
    m.spin_up()
    n = 9
    L = (np.diag(np.full(n, -2.))+np.diag(np.ones(n-1), 1)+np.diag(np.ones(n-1), -1))*2/100
    A = L.astype(complex)-1j*1e-4*np.eye(n)
    forcing = np.full(n, 1j*1e-4*10)
    forcing[-1] += 2/100*10
    steady = np.linalg.solve(A, -forcing)
    np.testing.assert_allclose(m.u[0, 1:-1]+1j*m.v[0, 1:-1], steady, atol=1e-9)
    assert m.time == 0
