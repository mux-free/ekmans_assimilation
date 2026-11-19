"""Wind observations on the FULL physical grid, independent of analysis variables."""
import numpy as np


def wind_matrix(z_model, heights, variables=("u", "v")):
    """Linear interpolation, with rows [all u heights, all v heights]."""
    z = np.asarray(z_model, float)
    h = np.atleast_1d(np.asarray(heights, float))
    variables = (variables,) if isinstance(variables, str) else tuple(variables)
    if z.ndim != 1 or len(z) < 2 or not np.all(np.isfinite(z)) or np.any(np.diff(z) <= 0):
        raise ValueError("Model heights must be finite and strictly increasing")
    if h.ndim != 1 or not len(h) or not np.all(np.isfinite(h)) or np.any((h < z[0]) | (h > z[-1])):
        raise ValueError("Observation heights must lie within the model domain")
    if len(np.unique(h)) != len(h):
        raise ValueError("Duplicate observation heights")
    if not variables or len(set(variables)) != len(variables) or not set(variables) <= {"u", "v"}:
        raise ValueError("Observe u and/or v, without duplicates")
    H = np.zeros((len(h)*len(variables), 2*len(z)))
    for k, variable in enumerate(variables):
        offset = 0 if variable == "u" else len(z)
        for j, height in enumerate(h):
            left = int(np.clip(np.searchsorted(z, height, side="right") - 1, 0, len(z)-2))
            weight = (height-z[left])/(z[left+1]-z[left])
            H[k*len(h)+j, offset+left] = 1-weight
            H[k*len(h)+j, offset+left+1] = weight
    return H, h.copy(), variables


def predict_wind(model, H):
    return np.concatenate((model.u, model.v), axis=1) @ H.T
