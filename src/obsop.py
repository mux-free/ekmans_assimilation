import abc
from abc import abstractmethod
from typing import Callable, Optional, Sequence, Union

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import sparse
from numpy.typing import NDArray

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
    Works with plain Python sequences, numpy arrays, and OmegaConf ListConfig.
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


class ObservationOperator:
    """Observation operator for selecting specific states and indices from a 1D state array """

    def __init__(
        self,
        is_linear: bool = True,
    ):
        self.is_linear = is_linear

    @abstractmethod
    def _obs_operator(self, x: jnp.ndarray) -> jnp.ndarray:
        """Apply the observation operator to a single state."""
        raise NotImplementedError

    @abstractmethod
    def grad_obs_operator(self, x: jnp.ndarray) -> jnp.ndarray:
        """Gradient of the observation operator."""
        raise NotImplementedError

    def __call__(self, ensemble: jnp.ndarray) -> jnp.ndarray:
        """Apply the observation operator to the ensemble."""
        return jax.vmap(self._obs_operator)(ensemble)

class LinearObservationOperator(ObservationOperator):
    """Observation operator for selecting specific states and indices from a 3D state array.

    The state array is expected to have shape [ensemble_size, num_states, state_dim].
    This operator selects:
    - Specific states from the num_states dimension (via obs_states)
    - Specific indices from the state_dim dimension (via obs_indices)

    The output is flattened to shape [ensemble_size, len(obs_states) * len(obs_indices)].
    """

    def __init__(
        self,
        obs_states: np.ndarray,
        obs_indices: ObsIndicesLike,
        state_dim: int,
        num_states: int = 1,
    ):
        """
        Initialize the observation operator.

        Args:
            obs_states: Sequence of state indices to observe (from num_states dimension).
                       Example: [0, 1] to observe states at indices 0 and 1.
            obs_indices: Either a single 1D index array applied to every observed
                       state, or a sequence of per-state index arrays — one per
                       entry of obs_states — when each state has its own spatial
                       sampling pattern.
                       Example (uniform):  np.arange(0, 50, 5)
                       Example (per-state): [np.arange(0, 50, 45),
                                             np.arange(0, 50, 15)]
        """
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
        self.obs_matrix = sparse.BCOO.fromdense(self.obs_matrix)

    def grad_obs_operator(self, x: jnp.ndarray) -> jnp.ndarray:
        """Gradient of the observation operator."""
        return self.obs_matrix.T

    def _obs_operator(self, x: jnp.ndarray) -> jnp.ndarray:
        """Apply the observation operator to a single state."""
        return self.obs_matrix @ x.flatten()


def get_obs_matrix(
    obs_states: np.ndarray,
    obs_indices_per_state: Sequence[np.ndarray],
    num_states: int,
    state_dim: int,
) -> np.ndarray:
    """Create an observation matrix H mapping flattened state to observations.

    Each entry of `obs_states` is paired with the corresponding entry of
    `obs_indices_per_state`, allowing different spatial sampling per observed
    state. The flattening order of the input state is
    state0_dim0, state0_dim1, ..., state1_dim0, ... and observations are
    emitted in (obs_state, dim_index) row-major order.

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

