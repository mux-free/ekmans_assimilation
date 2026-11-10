import jax.numpy as jnp


def distance_based_localization(
    r_influ: int,
    state_dim: int,
    cov_prior: jnp.ndarray,
    num_states: int | None = None,
    periodic: bool = False,
) -> jnp.ndarray:
    """
    Apply localization to the covariance matrix.

    Args:
    r_influ (int): The radius of influence for localization -- grid cells.
    state_dim (int): The dimension of the state vector.
    cov_prior (jax.numpy.array): The prior covariance matrix.
            with shape: [num_states * state_dim, num_states * state_dim]

    Returns:
    jax.numpy.array: Localized covariance matrix.

    Notes:
    The taper is the Gaspari-Cohn (1999, eq. 4.10) 5th-order piecewise-rational
    correlation function, which is compactly supported AND positive
    semi-definite by construction. This guarantees the Schur (elementwise)
    product with the sample covariance stays PSD (unlike a hard-truncated
    Gaussian, which is not a valid correlation function).

    ``r_influ`` is interpreted as the CUTOFF radius: the taper is exactly 0
    for distances > r_influ. Internally the Gaspari-Cohn half-width is
    ``c = r_influ / 2`` so that the support ``d <= 2c`` equals ``r_influ``.
    """

    # -- Check assumption of num_state=1 is correct if that if no num_state is passed!!
    if num_states is None:
        if cov_prior.shape[0] % state_dim != 0:
            raise ValueError(
                "Num_state was None (assumed to be 1), however "
                f"cov_prior shape {cov_prior.shape} is incompatible with state_dim={state_dim}"
            )
        num_states = cov_prior.shape[0] // state_dim
    # -- Final check covariance matrix has expected shape || expected square atrix with dim: num_states * state_dim
    expected_dim = num_states * state_dim
    if cov_prior.shape != (expected_dim, expected_dim):
        raise ValueError(
            f"Expected covariance shape {(expected_dim, expected_dim)}, "
            f"got {cov_prior.shape}. num_states={num_states}, state_dim={state_dim}."
        )

    # --- Make a matrix that shows distance from diagonal
    idx = jnp.arange(state_dim)
    dist = jnp.abs(idx[:, None] - idx[None, :])

    # --- If periodic, wrap around corners of matrix
    if periodic:
        dist = jnp.minimum(dist, state_dim - dist)

    # --- Create mask with the Gaspari-Cohn 5th-order correlation function.
    # r_influ is the cutoff radius; support is d <= r_influ (== 2c), so c=r_influ/2.
    dist = dist.astype(jnp.float32)
    c = r_influ / 2.0
    z = dist / c

    # Safe z for the 1<z<=2 branch to avoid nan/inf at z=0 poisoning the where.
    z_safe = jnp.where(dist > 0, z, 1.0)

    gc_near = -0.25 * z**5 + 0.5 * z**4 + (5.0 / 8.0) * z**3 - (5.0 / 3.0) * z**2 + 1.0
    gc_far = (
        (1.0 / 12.0) * z**5
        - 0.5 * z**4
        + (5.0 / 8.0) * z**3
        + (5.0 / 3.0) * z**2
        - 5.0 * z
        + 4.0
        - 2.0 / (3.0 * z_safe)
    )

    mask = jnp.where(z <= 1.0, gc_near, jnp.where(z <= 2.0, gc_far, 0.0))
    # Clamp tiny negative round-off to keep entries in [0, 1].
    mask = jnp.clip(mask, 0.0, 1.0)

    # Apply the localization mask to the prior covariance matrix.
    # Each (state_dim x state_dim) block is multiplied by the same mask, so
    # tiling the mask across the full matrix lets us do it in one operation.
    full_mask = jnp.tile(mask, (num_states, num_states))
    cov_prior_loc = cov_prior * full_mask

    return cov_prior_loc


def make_localization_fn(
    localization_distance: Optional[int],
    state_dim: int,
    num_states: int,
    periodic: bool,
) -> Callable[[jnp.ndarray], jnp.ndarray]:
    """Build the covariance localization closure shared by all filters.

    When ``localization_distance`` is ``None`` the returned function is the
    identity; otherwise it applies distance-based localization to the prior
    covariance. Kept in one place so EnKF, AGMF and PFF cannot diverge.
    """
    if localization_distance is None:
        return lambda x: x

    return lambda x: distance_based_localization(
        r_influ=localization_distance,  # type: ignore[arg-type]
        state_dim=state_dim,
        cov_prior=x,
        num_states=num_states,
        periodic=periodic,
    )


class EnsembleKalmanFilter():
    """Ensemble Kalman Filter."""

    def __init__(
        self,
        ensemble_size: int,
        R: np.ndarray,
        obs_operator: ObservationOperator,
        name: str = "enkf",
        inflation_factor: float = 1.0,
        localization_distance: Optional[int] = None,
        periodic: bool = False,
    ) -> None:
        """
        Initialize the Ensemble Kalman Filter.
        Args:
        ensemble_size (int): Number of ensemble members.
        R (numpy.array): Observation error covariance matrix.
        inflation_factor (float): Inflation factor.
        obs_operator (Callable[[np.ndarray], np.ndarray]): Observation operator.
        localization_distance (int): Localization distance.
        """

        self.name = name
        self.obs_operator = obs_operator
        self.ensemble_size = ensemble_size
        self.inflation_factor = inflation_factor

        self.dofs = self.num_states * self.state_dim
        self.R = R
        self.localization_distance = localization_distance
        self.periodic = periodic

        self.localization = make_localization_fn(
            self.localization_distance,
            self.state_dim,
            self.num_states,
            self.periodic,
        )

    def _analysis_step(
        self,
        prior_ensemble: np.ndarray,
        obs_vect: np.ndarray,
        rng_key: jax.random.PRNGKey,
        **kwargs: Any,
    ) -> np.ndarray:
        """Analysis step of the Ensemble Kalman Filter.

        Args:
        mem (int): Number of ensemble members.
        nx (int): The size of the state vector.
        ensemble (numpy.array): Ensemble of state estimates.
        obs_vect (numpy.array): Observation vector.
        R (numpy.array): Observation error covariance matrix.
        rng_key (jax.random.PRNGKey): RNG key.

        Returns:
        dict: A dictionary containing the posterior ensemble, Kalman gain, innovation,
            mean and covariance of the posterior.
        """

        # Prepare the prior state vector (ensemble matrix) -> shape [dofs, N]
        prior_ensemble = prior_ensemble.reshape(self.ensemble_size, -1).T

        # Standard multiplicative prior inflation: inflate the forecast ensemble
        # ANOMALIES about their mean by inflation_factor (lambda). This scales the
        # standard deviation by lambda (covariance by lambda^2) AND, because the
        # analysis update below starts from these inflated members, it actually
        # increases the posterior spread -- unlike scaling only cov_prior, which
        # would inflate the gain and shrink the posterior spread (the opposite of
        # the intended effect).
        mean = jnp.mean(prior_ensemble, axis=1, keepdims=True)
        prior_ensemble = mean + self.inflation_factor * (prior_ensemble - mean)

        # Calculate the covariance of the (inflated) prior
        cov_prior = jnp.cov(prior_ensemble)
        cov_prior = self.localization(cov_prior)

        # Filter and perturb the observation vector
        rng_key, key = jax.random.split(rng_key)
        perturb = jax.random.multivariate_normal(
            key, jnp.zeros(self.obs_operator.num_obs), self.R, shape=(self.ensemble_size,)  # type: ignore[attr-defined]
        )
        # Center the perturbations exactly across the ensemble axis. The raw draws
        # carry an O(1/sqrt(N)) sample mean that would otherwise add a spurious
        # shift to the analysis mean; subtracting the ensemble mean removes it.
        perturb = perturb - perturb.mean(axis=0, keepdims=True)
        obs_vect_perturbed = obs_vect + perturb
        obs_vect_perturbed = obs_vect_perturbed.T

        # Observation operator matrix
        obs_matrix = self.obs_operator.obs_matrix  # type: ignore[attr-defined]

        # Calculate the Kalman gain
        k_left = cov_prior @ obs_matrix.T
        k_right = obs_matrix @ cov_prior @ obs_matrix.T + self.R

        # K_left * K_right^-1
        self.kalman_gain = jnp.linalg.solve(k_right, k_left.T).T

        # Calculate the innovation
        self.innovation = obs_vect_perturbed - obs_matrix @ prior_ensemble

        # Calculate the posterior ensemble
        posterior_ensemble = prior_ensemble + self.kalman_gain @ self.innovation
        posterior_ensemble = posterior_ensemble.T

        posterior_ensemble = posterior_ensemble.reshape(
            self.ensemble_size, self.num_states, self.state_dim
        )

        return posterior_ensemble