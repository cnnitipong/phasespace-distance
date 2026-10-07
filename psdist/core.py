"""Phase-space distance between two signals: CPST and percentile Hausdorff distance.

Both indices compare a test signal with a reference signal after each has been
turned into a trajectory by time-delay embedding (Takens, 1981):

    X(n) = [x(n), x(n + tau), ..., x(n + (m - 1) tau)]

CPST (Change in Phase Space Topology)
    For every test point s_i, take the centroid of its k nearest reference
    points and measure the distance E_i = || s_i - centroid ||.
    CPST = mean(E_i). It summarises the *average* distortion of the trajectory.

Percentile Hausdorff distance d_H
    For every point, take the distance to the nearest point of the other set,
    in both directions, keep the q-th percentile (q = 95 by default) of each
    direction and return the larger one. q = 100 gives the classical
    (worst-case) Hausdorff distance. It summarises the *largest local* gap.

Both values are 0 for identical signals and are expressed in the units of the
signal (for example volts).
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

__all__ = [
    "embed", "average_mutual_information", "first_minimum",
    "false_nearest_neighbours", "cpst", "hausdorff", "compare",
]


# --------------------------------------------------------------------------
# 1. Phase-space reconstruction
# --------------------------------------------------------------------------
def embed(x, tau: int, m: int) -> np.ndarray:
    """Time-delay embedding of a 1-D signal.

    Returns an array of shape (N - (m - 1) * tau, m), one row per point X(n).
    """
    x = np.asarray(x, dtype=float).ravel()
    n = x.size - (m - 1) * tau
    if tau < 1 or m < 1:
        raise ValueError("tau and m must be positive integers")
    if n < 1:
        raise ValueError(f"signal too short ({x.size} samples) for tau={tau}, m={m}")
    return np.column_stack([x[i * tau:i * tau + n] for i in range(m)])


def _as_trajectory(x, tau, m):
    """Accept either a 1-D signal (embedded here) or an already embedded array."""
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        return embed(x, tau, m)
    if x.ndim == 2:
        return x
    raise ValueError("input must be a 1-D signal or a 2-D trajectory")


def average_mutual_information(x, max_lag: int, bins: int = 16) -> np.ndarray:
    """Average mutual information I(lag) for lag = 0..max_lag (histogram estimate)."""
    x = np.asarray(x, dtype=float).ravel()
    edges = np.histogram_bin_edges(x, bins=bins)
    out = np.empty(max_lag + 1)
    for lag in range(max_lag + 1):
        a, b = x[:x.size - lag], x[lag:]
        pxy, _, _ = np.histogram2d(a, b, bins=[edges, edges])
        pxy /= pxy.sum()
        px, py = pxy.sum(1), pxy.sum(0)
        nz = pxy > 0
        out[lag] = np.sum(pxy[nz] * np.log(pxy[nz] / np.outer(px, py)[nz]))
    return out


def first_minimum(curve, smooth: int = 1) -> int:
    """Index of the first local minimum of a curve (last index if none).

    smooth > 1 applies a moving average of that width first, so that small
    ripples from noise are not mistaken for the first prominent minimum.
    """
    c = np.asarray(curve, dtype=float)
    if smooth > 1:
        w = int(smooth) | 1  # force odd width
        c = np.convolve(np.pad(c, w // 2, mode="edge"), np.ones(w) / w, mode="valid")
    for i in range(1, c.size - 1):
        if c[i] < c[i - 1] and c[i] <= c[i + 1]:
            return i
    return int(c.size - 1)


def false_nearest_neighbours(x, tau: int, max_m: int = 6, rtol: float = 10.0) -> np.ndarray:
    """Fraction of false nearest neighbours for m = 1..max_m (Kennel et al., 1992)."""
    x = np.asarray(x, dtype=float).ravel()
    frac = np.full(max_m, np.nan)
    for m in range(1, max_m + 1):
        try:
            nxt = embed(x, tau, m + 1)
        except ValueError:
            break
        cur = nxt[:, :m]
        dist, idx = cKDTree(cur).query(cur, k=2)
        d, j = dist[:, 1], idx[:, 1]
        extra = np.abs(nxt[:, m] - nxt[j, m])
        ok = d > 0
        frac[m - 1] = np.mean(extra[ok] / d[ok] > rtol) if ok.any() else 0.0
    return frac


# --------------------------------------------------------------------------
# 2. CPST
# --------------------------------------------------------------------------
def cpst(reference, test, tau: int = 1, m: int = 3, k: int = 5, return_details: bool = False):
    """Change in Phase Space Topology between a reference and a test signal.

    Parameters
    ----------
    reference, test : 1-D signals (embedded with tau, m) or 2-D trajectories.
    k : number of nearest reference neighbours averaged for each test point.
    return_details : if True also return a dict with E_i, centroids and indices.
    """
    R = _as_trajectory(reference, tau, m)
    S = _as_trajectory(test, tau, m)
    k = int(min(k, len(R)))
    _, idx = cKDTree(R).query(S, k=k)
    idx = idx.reshape(len(S), k)
    centroids = R[idx].mean(axis=1)
    E = np.linalg.norm(S - centroids, axis=1)
    value = float(E.mean())
    if return_details:
        return value, {"E": E, "centroids": centroids, "neighbours": idx, "R": R, "S": S}
    return value


# --------------------------------------------------------------------------
# 3. Percentile Hausdorff distance
# --------------------------------------------------------------------------
def hausdorff(reference, test, tau: int = 1, m: int = 3, q: float = 95.0, return_details: bool = False):
    """Percentile Hausdorff distance between two signals (q = 100: classical).

    d_H = max( Q_q( min_j ||s_i - r_j|| ),  Q_q( min_i ||r_j - s_i|| ) )
    """
    R = _as_trajectory(reference, tau, m)
    S = _as_trajectory(test, tau, m)
    d_sr, j_sr = cKDTree(R).query(S)   # every test point -> nearest reference point
    d_rs, i_rs = cKDTree(S).query(R)   # every reference point -> nearest test point
    h_sr, h_rs = np.percentile(d_sr, q), np.percentile(d_rs, q)
    value = float(max(h_sr, h_rs))
    if return_details:
        return value, {"d_sr": d_sr, "d_rs": d_rs, "nearest_r": j_sr, "nearest_s": i_rs,
                       "h_sr": float(h_sr), "h_rs": float(h_rs), "R": R, "S": S}
    return value


def compare(reference, test, tau: int, m: int = 3, k: int = 5, q: float = 95.0) -> dict:
    """Return both indices (and the classical Hausdorff distance) in one call."""
    return {
        "cpst": cpst(reference, test, tau, m, k),
        "hausdorff_q": hausdorff(reference, test, tau, m, q),
        "hausdorff_classical": hausdorff(reference, test, tau, m, 100.0),
        "tau": tau, "m": m, "k": k, "q": q,
    }
