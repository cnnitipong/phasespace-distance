"""One-call visual comparison of two signals in phase space."""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from .core import cpst, hausdorff

REF_COLOR = "#B38450"   # reference: tan, solid
TEST_COLOR = "#683817"  # test: dark brown, dashed
ACCENT = "#DEA608"      # highlights: gold


def plot_comparison(reference, test, tau: int, m: int = 3, k: int = 5, q: float = 95.0,
                    t=None, labels=("reference", "test"), title=None):
    """Four-panel figure: signals, trajectories, CPST deviations and Hausdorff distances.

    Returns (fig, results) where results holds the CPST and Hausdorff values.
    """
    reference = np.asarray(reference, float).ravel()
    test = np.asarray(test, float).ravel()
    t = np.arange(reference.size) if t is None else np.asarray(t)
    c_val, c = cpst(reference, test, tau, m, k, return_details=True)
    h_val, h = hausdorff(reference, test, tau, m, q, return_details=True)
    R, S = c["R"], c["S"]

    fig, ax = plt.subplots(2, 2, figsize=(11, 8))
    a = ax[0, 0]
    a.plot(t, reference, "-", color=REF_COLOR, lw=2, label=labels[0])
    a.plot(t[:test.size], test, "--", color=TEST_COLOR, lw=2, label=labels[1])
    a.set(title="(a) Signals in time", xlabel="time / sample", ylabel="amplitude")
    a.legend(frameon=False)

    a = ax[0, 1]
    a.plot(R[:, 0], R[:, 1], "-", color=REF_COLOR, lw=2, label=labels[0])
    a.plot(S[:, 0], S[:, 1], "--", color=TEST_COLOR, lw=2, label=labels[1])
    a.set(title=fr"(b) Trajectories, $\tau$={tau}, m={m}", xlabel="x(n)", ylabel=r"x(n+$\tau$)")
    a.legend(frameon=False)

    a = ax[1, 0]
    a.plot(c["E"], color=TEST_COLOR, lw=1.5)
    a.axhline(c_val, color=ACCENT, lw=2.5, ls="--", label=f"CPST = mean = {c_val:.4g}")
    a.set(title=fr"(c) CPST local deviation $E_i$ (k={k})", xlabel="test point i",
          ylabel=r"$E_i$")
    a.legend(frameon=False)

    a = ax[1, 1]
    for d, ls, lab in ((h["d_sr"], "--", "test to reference"), (h["d_rs"], "-", "reference to test")):
        ds = np.sort(d)
        a.plot(ds, np.arange(1, ds.size + 1) / ds.size, ls, color=TEST_COLOR if ls == "--" else REF_COLOR,
               lw=2, label=lab)
    a.axhline(q / 100, color="0.55", lw=1, ls=":")
    a.axvline(h_val, color=ACCENT, lw=2.5, ls="--", label=fr"$d_H$ (Q{q:g}) = {h_val:.4g}")
    a.set(title="(d) Nearest-neighbour distances (Hausdorff)", xlabel="distance to the other set",
          ylabel="cumulative fraction")
    a.legend(frameon=False, loc="upper left")

    for a in ax.flat:
        a.grid(alpha=0.3)
    if title:
        fig.suptitle(title, fontweight="bold")
    fig.tight_layout()
    return fig, {"cpst": c_val, "hausdorff_q": h_val}
