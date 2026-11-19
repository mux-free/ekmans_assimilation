import numpy as np
from typing import Union, Sequence

ObsIndicesLike = Union[
    Sequence[int], np.ndarray, Sequence[Sequence[int]], Sequence[np.ndarray]
]



def _normalize_obs_indices(
    obs_indices: ObsIndicesLike,
    num_obs_states: int,
) -> list[np.ndarray]:
    """Return one index array per observed state.

    Accepts either a flat 1D index array (broadcast to every observed state) or
    a sequence of per-state index arrays (one entry per element in obs_states).
    Works with plain Python sequences, numpy arrays.
    """
    items = list(obs_indices)
    if len(items) > 0 and np.ndim(items[0]) >= 1:
        per_state = [np.asarray(idx) for idx in items]
        if len(per_state) != num_obs_states:
            raise ValueError(
                f"obs_indices has {len(per_state)} per-state arrays but "
                f"obs_states has {num_obs_states} entries"
            )
        return per_state
    flat = np.asarray(items)
    return [flat for _ in range(num_obs_states)]



def get_obs_matrix(
    obs_states: np.ndarray,
    obs_indices_per_state: Sequence[np.ndarray],
    num_states: int,
    state_dim: int,
) -> np.ndarray:
    """
    Create observation matrix H that maps flattened state to observations.

    - num_states: int   Is the number of variables (in Ekman case this is u and v wind)
    - state_dim: int    Is the length of domain size (Ekman the height of the domain)


    Each entry of `obs_states` is paired with the corresponding entry of
    `obs_indices_per_state`, allowing different spatial sampling per observed
    state. 
    
    The flattening order of the input state is
    state0_dim0, state0_dim1, ..., state1_dim0, ... and observations are
    returned in (obs_state, dim_index) row-major order.

    Returns:
        H of shape [num_obs, num_states * state_dim] where
        num_obs = sum(len(idx) for idx in obs_indices_per_state).
    """
    obs_states = np.asarray(obs_states)

    if len(obs_states) != len(obs_indices_per_state):
        raise ValueError(
            f"obs_states has length {len(obs_states)} but obs_indices_per_state "
            f"has length {len(obs_indices_per_state)}"
        )

    total_obs = sum(len(idx) for idx in obs_indices_per_state)
    total_state_size = num_states * state_dim

    H = np.zeros((total_state_size, total_obs))

    obs_col = 0
    for state_idx, dim_indices in zip(obs_states, obs_indices_per_state):
        for dim_idx in np.asarray(dim_indices):
            flattened_idx = state_idx * state_dim + dim_idx
            H[flattened_idx, obs_col] = 1
            obs_col += 1

    return H.T


def get_height_indices(z_obs, z_model):
    if isinstance(z_obs, int):
        z_obs = [z_obs]
    if not isinstance(z_obs, list) and all(isinstance(item, int) for item in z_obs):
        raise ValueError(f"z_obs must be either list of ints, a single int. Not {type(z_obs)}")

    return [np.argmin(np.abs(z_model-zobs)).item() for zobs in z_obs]

class ObservationOperator:
    """ 
    obs_states:     Which variables are obsered? 
    obs_indices:    Where are observations made? Indices along spatial dimension
    state_dim:      Phyiscal domain size
    num_states:     Number of observed states
    """
    def __init__(
        self,
        obs_states: np.ndarray,
        obs_indices: ObsIndicesLike,
        state_dim: int,
        obs_std: float,
        num_states: int = 2,
    ):
            
        self.obs_states = obs_states
        self.obs_indices = obs_indices
        self.state_dim = state_dim
        self.num_states = num_states

        self.obs_indices_per_state = _normalize_obs_indices(
            self.obs_indices, len(obs_states)
        )

        self.num_obs = sum(len(idx) for idx in self.obs_indices_per_state)

        self.is_linear = True

        self.obs_matrix = get_obs_matrix(
            obs_states, self.obs_indices_per_state, self.num_states, self.state_dim
        )
        # self.obs_matrix = sparse.BCOO.fromdense(self.obs_matrix)

    def grad_obs_operator(self, x: np.ndarray) -> np.ndarray:
        """Gradient of the observation operator."""
        return self.obs_matrix.T

    def _obs_operator(self, x: np.ndarray) -> np.ndarray:
        """Apply the observation operator to a single state."""
        return self.obs_matrix @ x.flatten()


    def synthetic_obs(self, obs_true, obs_std, random_seed=14):
        rng = np.random.default_rng(random_seed)
        pred_obs = obs_true + rng.normal(0, obs_std, len(obs_true))
        return pred_obs
