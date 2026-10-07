"""Mock signal pairs that make the two indices easy to see.

The base signal is a two-tone wave, x(t) = sin(t) + 0.45 sin(3t + 0.6), sampled
over six periods. Embedded with tau = 8 samples (the first minimum of the
average mutual information) and m = 3 it draws a clean closed knot in 3-D, so
any change to the signal is visible as a change of that loop.
The same generator is used by the web demo (web/app.js).
"""
from __future__ import annotations

import numpy as np

KINDS = ("shape", "global", "local", "frequency", "noise")
N_DEFAULT, PERIODS = 600, 6
TAU_DEFAULT = 8                               # AMI first minimum for the base signal


def base(n: int = N_DEFAULT) -> tuple[np.ndarray, np.ndarray]:
    t = np.linspace(0.0, 2 * np.pi * PERIODS, n, endpoint=False)
    return t, np.sin(t) + 0.45 * np.sin(3 * t + 0.6)


def make_pair(kind: str = "shape", severity: float = 0.3, noise: float = 0.0,
              n: int = N_DEFAULT, seed: int = 0):
    """Return (t, reference, test) for one of KINDS.

    shape      the third harmonic grows by `severity` (systematic change of shape)
    global     the whole signal shrinks by a factor (1 - severity)
    local      a short burst of height `severity` appears once, at two thirds of the record
    frequency  the base frequency rises by `severity` x 10 %
    noise      no change at all; only noise of std `noise` (healthy pair)
    """
    if kind not in KINDS:
        raise ValueError(f"kind must be one of {KINDS}")
    t, ref = base(n)
    if kind == "shape":
        test = np.sin(t) + (0.45 + severity) * np.sin(3 * t + 0.6)
    elif kind == "global":
        test = (1.0 - severity) * ref
    elif kind == "local":
        t0, w = t[int(2 * n / 3)], 0.6
        test = ref + 2.0 * severity * np.exp(-((t - t0) / w) ** 2)
    elif kind == "frequency":
        f = 1.0 + 0.1 * severity
        test = np.sin(f * t) + 0.45 * np.sin(3 * f * t + 0.6)
    else:
        test = ref.copy()
    rng = np.random.default_rng(seed)
    if noise > 0:
        ref = ref + noise * rng.standard_normal(n)
        test = test + noise * rng.standard_normal(n)
    return t, ref, test
