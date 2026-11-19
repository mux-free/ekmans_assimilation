"""Small stochastic EnKF: members in rows, variables/observations in columns."""
import numpy as np


def validate_covariance(R, size):
    R = np.asarray(R, dtype=float)
    if R.shape != (size, size) or not np.all(np.isfinite(R)):
        raise ValueError(f"R must be a finite ({size}, {size}) covariance matrix")
    if not np.allclose(R, R.T, rtol=1e-12, atol=1e-14):
        raise ValueError("R must be symmetric")
    try:
        np.linalg.cholesky(R)
    except np.linalg.LinAlgError as exc:
        raise ValueError("R must be positive definite") from exc
    return R.copy()


class EnsembleKalmanFilter:
    """Stochastic, unlocalized EnKF with persistent observation-perturbation RNG.

    Keep inflation=1 for the initial experiments. Unsupported options fail
    explicitly instead of silently doing nothing. Global parameters require
    a deliberate localization design; a height taper cannot just be applied
    to the flattened augmented vector.
    """

    def __init__(self, ensemble_size, R, num_states=2, state_dim=101,
                 name="enkf", inflation_factor=1.0, localization_distance=None,
                 periodic=False, random_seed=14):
        if not isinstance(ensemble_size, int) or ensemble_size < 2:
            raise ValueError("At least two ensemble members are required")
        if inflation_factor != 1.0:
            raise NotImplementedError("Inflation is not implemented in this minimal filter")
        if localization_distance is not None or periodic:
            raise NotImplementedError("Localization is not implemented in this minimal filter")
        self.ensemble_size = ensemble_size
        self.name = name
        self.R = np.asarray(R, float).copy()
        self.rng = np.random.default_rng(random_seed)

    def analysis_step(self, X, Y, y, R=None):
        """X: (N,D), Y: (N,M), y: (M,). R contains VARIANCES/covariances."""
        X, Y, y = [np.asarray(a, dtype=float) for a in (X, Y, y)]
        if X.ndim != 2 or Y.ndim != 2 or X.shape[0] != self.ensemble_size or Y.shape[0] != self.ensemble_size:
            raise ValueError("X and Y must be 2D arrays with members in rows")
        if not X.shape[1] or not Y.shape[1] or y.shape != (Y.shape[1],):
            raise ValueError("Observation vector must match columns of Y")
        if not all(np.all(np.isfinite(a)) for a in (X, Y, y)):
            raise ValueError("Ensembles and observations must be finite")
        covariance = validate_covariance(self.R if R is None else R, len(y))
        Xp, Yp = X - X.mean(axis=0), Y - Y.mean(axis=0)
        Pxy = Xp.T @ Yp / (self.ensemble_size - 1)
        Pyy = Yp.T @ Yp / (self.ensemble_size - 1)
        self.kalman_gain = np.linalg.solve(Pyy + covariance, Pxy.T).T
        noise = self.rng.multivariate_normal(np.zeros(len(y)), covariance, size=self.ensemble_size)
        noise -= noise.mean(axis=0, keepdims=True)
        self.perturbations = noise
        self.innovation = y - Y.mean(axis=0)
        self.innovation_covariance = Pyy + covariance
        Xa = X + (y + noise - Y) @ self.kalman_gain.T
        if not np.all(np.isfinite(Xa)):
            raise ValueError("Non-finite analysis")
        return Xa
