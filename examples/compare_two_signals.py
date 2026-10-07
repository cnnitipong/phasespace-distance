"""Compare two signals with CPST and the percentile Hausdorff distance.

Usage
-----
    python examples/compare_two_signals.py                      # built-in demo signals
    python examples/compare_two_signals.py ref.csv test.csv --tau 64 --m 3

Each CSV holds one signal: either a single column of amplitudes, or two
columns (time, amplitude). A header row is allowed.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))          # use the package without installing it

import psdist                                 # noqa: E402
from psdist.plotting import plot_comparison   # noqa: E402


def demo_signals(n: int = 200, seed: int = 0):
    """Two synthetic pulsed-eddy-current-like transients (defect-free and 'cracked')."""
    t = np.arange(n) * 2.0e-5                            # 50 kS/s, 4 ms window
    t_off = 2.1e-3                                       # excitation switched off here

    def pect(amp, t_rise, t_decay, undershoot, t_under):
        on = amp * (1.0 - np.exp(-t / t_rise))
        v_off = amp * (1.0 - np.exp(-t_off / t_rise))
        s = t - t_off
        off = v_off * ((1 + undershoot) * np.exp(-s / t_decay) - undershoot * np.exp(-s / t_under))
        return np.where(t < t_off, on, off)

    healthy = pect(0.2185, 1.9e-4, 3.2e-4, 0.14, 1.6e-3)
    # the 'crack' lowers the plateau, slows the rise and the decay, and reduces the undershoot
    cracked = pect(0.1920, 2.6e-4, 4.6e-4, 0.09, 1.6e-3)
    rng = np.random.default_rng(seed)
    noise = 0.0008
    return t, healthy + noise * rng.standard_normal(n), cracked + noise * rng.standard_normal(n)


def load_signal(path: Path):
    first = path.read_text().splitlines()[0].split(",")[0]
    try:
        float(first)
        skip = 0
    except ValueError:      # header row present
        skip = 1
    data = np.loadtxt(path, delimiter=",", skiprows=skip, ndmin=2)
    return (data[:, 0], data[:, 1]) if data.shape[1] >= 2 else (None, data[:, 0])


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("reference", nargs="?", type=Path, help="CSV of the reference (defect-free) signal")
    p.add_argument("test", nargs="?", type=Path, help="CSV of the test signal")
    p.add_argument("--tau", type=int, default=None, help="embedding delay in samples (default: AMI first minimum)")
    p.add_argument("--m", type=int, default=3, help="embedding dimension (default 3)")
    p.add_argument("--k", type=int, default=5, help="CPST neighbours (default 5)")
    p.add_argument("--q", type=float, default=95.0, help="Hausdorff percentile (default 95)")
    p.add_argument("--out", type=Path, default=HERE / "comparison.png", help="output figure")
    a = p.parse_args()

    if a.reference and a.test:
        t, ref = load_signal(a.reference)
        _, test = load_signal(a.test)
    else:
        t, ref, test = demo_signals()
        np.savetxt(HERE / "data" / "reference.csv", np.column_stack([t, ref]), delimiter=",",
                   header="time_s,amplitude", comments="")
        np.savetxt(HERE / "data" / "test.csv", np.column_stack([t, test]), delimiter=",",
                   header="time_s,amplitude", comments="")

    if a.tau is None:   # same tau for both signals, chosen from the reference
        ami = psdist.average_mutual_information(ref, max_lag=len(ref) // 3)
        a.tau = max(1, psdist.first_minimum(ami, smooth=9))
        print(f"tau from AMI first minimum: {a.tau} samples")

    res = psdist.compare(ref, test, tau=a.tau, m=a.m, k=a.k, q=a.q)
    for key, val in res.items():
        print(f"{key:>20s}: {val:.6g}" if isinstance(val, float) else f"{key:>20s}: {val}")

    fig, _ = plot_comparison(ref, test, tau=a.tau, m=a.m, k=a.k, q=a.q,
                             t=None if t is None else t * 1e3,
                             labels=("reference", "test"),
                             title="CPST and percentile Hausdorff distance")
    if t is not None:
        fig.axes[0].set_xlabel("time (ms)")
    fig.savefig(a.out, dpi=150)
    print(f"figure saved to {a.out}")


if __name__ == "__main__":
    main()
