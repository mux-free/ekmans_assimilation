import numpy as np 
import xarray as xr

from typing import  Sequence, Union
from pathlib import Path
from dataclasses import dataclass

from src.ekman import EkmanEns, EkmanConfig, SimConfig, EnsembleCreator, rmse
from src.enkf import EnsembleKalmanFilter
from src.observation_operator import ObservationOperator, get_obs_matrix, get_height_indices



@dataclass
class AssimilationState:
    Xb: np.ndarray
    Xa: np.ndarray
    X_true: np.ndarray
    kalman_gain: np.ndarray

@dataclass
class ObservationState:
    H: np.ndarray
    obs_vars: Sequence
    obs_std: float
    ds_obs: xr.Dataset

@dataclass
class ExpState:
    ds_true: xr.Dataset
    ds_ctrl: xr.Dataset
    ds_assim: xr.Dataset
    rmse_dict: dict
    assim_state: AssimilationState | Sequence[AssimilationState]
    obs_state: ObservationState | Sequence[ObservationState]




class EkmanRun:
    def __init__(self, ekman_config: EkmanConfig, sim_config: SimConfig):
        self.ekman_config = ekman_config
        self.sim_config = sim_config
        
        # --- short-hands
        self.obs_std = sim_config.observations["obs_error"]
        self.obs_vars = sim_config.observations["obs_vars"]
        self.state_vars = sim_config.data_assimilation["state_vars"]


    def perturb_ug_K(self):
        enscr = EnsembleCreator(self.sim_config)
        ens_dict = enscr()
        
        ug_sim = self.sim_config.parameters["ug_truth"] + self.sim_config.parameters["ug_error"]
        K_sim = self.sim_config.parameters["K_truth"] + self.sim_config.parameters["K_error"]
        
        self.ug_ens = ens_dict["ug_ens"] if "ug" in self.state_vars else ug_sim
        self.K_ens  = ens_dict["K_ens"]  if "K"  in self.state_vars else K_sim


    def setup_simulation(self, ug, K):
        """ 
        SetUp a simulation. If ug and K are single values, one single run is setup. 
        If either ug or K is a array of numbers, an ensemble of runs is setup
        """
        sim = EkmanEns(ug=ug, vg=0, K=K, config=self.ekman_config)
        return sim

    def spin_up(self, sim: EkmanEns):
        """ Run the spin-up as specified in config file """
        sim.spin_up()

    def copy_state(self, sim: EkmanEns):
        return sim.copy()

    def integrate(self, sim: EkmanEns, t_stop: int, return_dataset: bool = True):
        """ Integrate the model state from its current time until t_stop in seconds """
        ds = sim.run(until=t_stop)
        if return_dataset:
            return ds


    def get_synthetic_obs(
        self, 
        sim_truth, 
        obs_heights: Sequence|None = None
        ) -> ObservationState:
        
        # --- At what heights are they observed
        if obs_heights is None:
            obs_heights = self.sim_config.observations["z_obs"]
        
        # Map  heights to model grid
        z = np.asarray(sim_truth.z)
        obs_indices = np.array([np.abs(z - h).argmin() for h in obs_heights])
        if len(np.unique(obs_indices)) != len(obs_indices):
            raise ValueError("Multiple observation heights map to the same grid index.")

        # --- Get Observation matrix 
        obs_operator =  ObservationOperator(
            obs_states=np.array([0, 1]),  # u and v
            obs_indices=obs_indices,
            state_dim=self.ekman_config.state_dim,
            num_states=self.ekman_config.num_states,
            obs_std=self.obs_std,
        )
        H = obs_operator.obs_matrix
        y_true = (sim_truth.state_vector(("u", "v")) @ H.T)[0]
        y_obs = obs_operator.synthetic_obs(y_true, self.obs_std)
        
        # Unstack observations into an xarray Dataset
        values = np.asarray(y_obs).reshape(len(self.obs_vars), len(obs_indices))

        ds_obs = xr.Dataset(
            {var: ("zi", values[i]) for i, var in enumerate(self.obs_vars)},
            coords={
                "zi": obs_indices,
                "z": ("zi", z[obs_indices]),
            },
        )
        ds_obs["z"].attrs["units"] = "m"
        ds_obs["zi"].attrs["units"] = "column_index"

        return ObservationState(
            H=H,
            obs_vars=self.obs_vars,
            obs_std=self.obs_std,
            ds_obs=ds_obs,
        )


    def postprocess_K(self, Xa, Xb):
        # --- 5) Post-process K, s.t. it is NOT negative
        if "K" in self.state_vars:
            if self.state_vars[1] == "K":
                # If K is negative, reject!
                Xa[:,1] = np.where(Xa[:,1]>=0, Xa[:,1], Xb[:,1])
            else:
                raise ValueError("If K is in state-vector, it should be second column in state vector")
        return Xa


    def assimilate(
        self, 
        sim_prior: EkmanEns, 
        sim_truth: EkmanEns,
        observation_state: ObservationState, 
        ) -> AssimilationState:
        
        # --- 1) Retireve syntehtic obs as stacked array
        y_obs = np.concatenate([observation_state.ds_obs[var].values for var in self.obs_vars])

        # --- 2) Define StateVector & Predicted Obs
        H = observation_state.H
        Xb = sim_prior.state_vector(self.state_vars)
        Yb = sim_prior.state_vector(self.obs_vars) @ H.T

        # --- 3) Define ObsError CovMatrix
        R = self.obs_std**2 * np.eye(len(y_obs))

        # --- 4) Perform analysis
        enkf = EnsembleKalmanFilter(
            ensemble_size = Xb.shape[0],
            R=R,
            state_dim=self.ekman_config.state_dim,
            num_states=self.ekman_config.num_states,
            )
        Xa = enkf.analysis_step(Xb, Yb, y_obs)
        Xa = self.postprocess_K(Xa, Xb)

        # --- 5) Update Ensemble State of Ekman Model ensemble
        sim_prior.set_state_vector(Xa, self.state_vars)

        return AssimilationState(
            Xb=Xb,
            Xa=Xa,
            kalman_gain=enkf.kalman_gain,
            X_true=sim_truth.state_vector(self.state_vars),
        )


    def create_exp_title(self, assim_state: AssimilationState):
        exp_title = []
        for i, name in enumerate(self.state_vars):
            if name in ["ug","K"]:
                tit = (
                    f"{name:<3}:  "
                    f"true = {assim_state.X_true[:,i].mean():4.1f}  |  "
                    f"prior = {assim_state.Xb[:, i].mean():5.2f}  |  "
                    f"post = {assim_state.Xa[:, i].mean():5.2f}"
                    )
                exp_title.append(tit)
        exp_title = "\n".join(exp_title)
        return exp_title

    
    def compute_rmse(da_sim: xr.DataArray, da_true: xr.DataArray, dim: str|list[str]):
        return rmse(da_sim, da_true, dim)


    def __call__(self):
        """ Run Experiment with single assimilation step """

        # --- 1) Create perturbed parameters (NOTE: only if they are in state-vectr)
        self.perturb_ug_K()

        # --- 2) Initialise and SpinUp Truth and Ensemble
        sim_truth = self.setup_simulation(ug=self.sim_config.parameters["ug_truth"], K=self.sim_config.parameters["K_truth"])
        sim_ens = self.setup_simulation(ug=self.ug_ens, K=self.K_ens)
        self.spin_up(sim_truth)
        self.spin_up(sim_ens)
        sim_ctrl = self.copy_state(sim_ens)

        # --- 3) Integrate States until DA-step
        t_assim = self.sim_config.data_assimilation["t0_assim"]
        self.integrate(sim_truth, t_stop=t_assim, return_dataset=False)
        self.integrate(sim_ens,   t_stop=t_assim, return_dataset=False)

        # --- 4) DA step
        obs_state   = self.get_synthetic_obs(sim_truth)
        assim_state = self.assimilate(sim_prior=sim_ens, observation_state=obs_state, sim_truth=sim_truth)

        # --- 5) Integrate until end
        tstop = self.sim_config.total_run_time
        ds_true = self.integrate(sim_truth, t_stop=tstop)
        ds_da   = self.integrate(sim_ens,   t_stop=tstop)
        ds_ctrl = self.integrate(sim_ctrl,  t_stop=tstop)

        # --------------------------------------
        rmse_dict = {
            "Control": rmse(ds_ctrl, ds_true, dim=["z","ensemble"])["u"],
            "Members": rmse(ds_da, ds_true, dim="z")["u"],
            "EnsMean": rmse(ds_da, ds_true, dim=["z","ensemble"])["u"]
            }

        return ExpState(
            ds_true=ds_true,
            ds_ctrl=ds_ctrl,
            ds_assim=ds_da,
            rmse_dict=rmse_dict,
            assim_state=assim_state,
            obs_state=obs_state,
        )
