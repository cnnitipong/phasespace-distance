import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import psdist  # noqa: E402


def pulse(n=200, decay=40.0, amp=1.0):
    t = np.arange(n, dtype=float)
    return amp * (1 - np.exp(-t / 6.0)) * np.exp(-t / decay)


def test_embed_shape_and_values():
    x = np.arange(10.0)
    X = psdist.embed(x, tau=2, m=3)
    assert X.shape == (6, 3)
    assert np.allclose(X[0], [0, 2, 4])


def test_identical_signals():
    x = pulse()
    # k = 1: each point is its own nearest neighbour, so CPST is exactly zero
    assert psdist.cpst(x, x, tau=8, m=3, k=1) == pytest.approx(0.0, abs=1e-12)
    # k > 1: the neighbour centroid is slightly off the point, a small positive floor
    floor = psdist.cpst(x, x, tau=8, m=3, k=5)
    assert 0.0 < floor < psdist.cpst(x, pulse(decay=30.0), tau=8, m=3, k=5)
    assert psdist.hausdorff(x, x, tau=8, m=3) == pytest.approx(0.0, abs=1e-12)


def test_hausdorff_is_symmetric():
    a, b = pulse(), pulse(decay=30.0)
    assert psdist.hausdorff(a, b, 8, 3) == pytest.approx(psdist.hausdorff(b, a, 8, 3))


def test_indices_grow_with_difference():
    ref = pulse()
    small, large = pulse(decay=36.0), pulse(decay=25.0)
    assert psdist.cpst(ref, small, 8, 3) < psdist.cpst(ref, large, 8, 3)
    assert psdist.hausdorff(ref, small, 8, 3) < psdist.hausdorff(ref, large, 8, 3)


def test_percentile_ignores_single_outlier_but_classical_does_not():
    ref = pulse()
    spiky = ref.copy()
    spiky[100] += 0.5                       # one isolated spike
    classical = psdist.hausdorff(ref, spiky, 8, 3, q=100)
    robust = psdist.hausdorff(ref, spiky, 8, 3, q=95)
    assert classical > 0.4
    assert robust < 0.05 * classical


def test_classical_hausdorff_known_value():
    # two constant signals offset by c: every point is shifted by c in each of m axes
    c, m = 0.3, 3
    d = psdist.hausdorff(np.zeros(50), np.full(50, c), tau=2, m=m, q=100)
    assert d == pytest.approx(c * np.sqrt(m))


def test_compare_returns_all_keys():
    r = psdist.compare(pulse(), pulse(decay=30.0), tau=8)
    assert {"cpst", "hausdorff_q", "hausdorff_classical"} <= set(r)
    assert r["hausdorff_q"] <= r["hausdorff_classical"]


def test_fnn_drops_to_zero_for_smooth_pulse():
    f = psdist.false_nearest_neighbours(pulse(), tau=8, max_m=4)
    assert f[2] == pytest.approx(0.0, abs=0.05)


def test_mock_pairs_reproduce_documented_values():
    from psdist.mock import make_pair
    _, ref, test = make_pair("shape", 0.3)
    assert psdist.cpst(ref, test, tau=8, m=3, k=5) == pytest.approx(0.3355, abs=5e-4)
    assert psdist.hausdorff(ref, test, tau=8, m=3, q=95) == pytest.approx(0.4183, abs=5e-4)
    _, r0, t0 = make_pair("noise", 0.3, 0.0)
    assert psdist.hausdorff(r0, t0, tau=8, m=3) == pytest.approx(0.0, abs=1e-12)
