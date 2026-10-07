"""Regenerate every figure in docs/figures from the mock signals (psdist.mock).

    python examples/make_readme_figures.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import psdist as P                                         # noqa: E402
from psdist.mock import make_pair, TAU_DEFAULT, KINDS      # noqa: E402

OUT = ROOT / "docs" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
BG, INK, PINK, MUTED, LINE, SURF = "#fbfbf9", "#141414", "#DE5C8E", "#6b6b6b", "#d9d9d6", "#f4f4f2"
CMAP = LinearSegmentedColormap.from_list("np", [SURF, "#efb4ca", PINK, "#7a1f45", INK])
TIME = LinearSegmentedColormap.from_list("t", ["#c9c9c6", MUTED, INK])
TAU, M, K, Q = TAU_DEFAULT, 3, 5, 95
plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG, "font.family": "DejaVu Sans",
    "font.size": 11, "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": LINE,
    "grid.linewidth": 0.6, "legend.frameon": False, "lines.linewidth": 1.8, "savefig.dpi": 160,
})


def ax3d(fig, pos, title):
    a = fig.add_subplot(pos, projection="3d")
    a.set_facecolor(BG)
    for axis in (a.xaxis, a.yaxis, a.zaxis):
        axis.pane.set_facecolor(SURF); axis.pane.set_edgecolor(LINE)
        axis._axinfo["grid"]["color"] = LINE
        axis.set_tick_params(labelsize=8, colors=MUTED)
    a.set_xlabel("x(n)", fontsize=9, labelpad=-2); a.set_ylabel("x(n+τ)", fontsize=9, labelpad=-2)
    a.set_zlabel("x(n+2τ)", fontsize=9, labelpad=-2)
    a.set_title(title, loc="left", fontsize=12, fontweight="bold", pad=0)
    a.view_init(24, -52)
    return a


def save(fig, name):
    fig.savefig(OUT / f"{name}.png", bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    print("saved", name)


t, ref, test = make_pair("shape", 0.3)
R, S = P.embed(ref, TAU, M), P.embed(test, TAU, M)

# ---------------------------------------------------------------- banner
fig = plt.figure(figsize=(13, 3.6))
fig.text(0.03, 0.70, "TOOL  ·  PHASE-SPACE DISTANCE", family="DejaVu Sans Mono", fontsize=10, color=MUTED)
fig.text(0.03, 0.42, "phasespace-distance", family="DejaVu Serif", fontsize=34, color=INK)
fig.text(0.03, 0.24, "How different are two signals?  Compare their trajectories,\nnot single numbers: CPST and the percentile Hausdorff distance.",
         fontsize=11.5, color=MUTED, va="top")
a = fig.add_axes([0.53, -0.05, 0.47, 1.12], projection="3d")
a.plot(*R.T, color=INK, lw=1.4); a.plot(*S.T, color=PINK, lw=1.4)
idx = np.arange(0, len(S), 6)
_, d = P.cpst(R, S, k=K, return_details=True)
for i in idx:
    a.plot(*np.c_[S[i], d["centroids"][i]], color=PINK, lw=0.5, alpha=0.6)
a.set_axis_off(); a.view_init(24, -52)
save(fig, "banner")

# ---------------------------------------------------------------- 01 signal to trajectory
n0 = 40
fig = plt.figure(figsize=(14, 4.3))
a = fig.add_subplot(131)
a.plot(t, ref, color=INK)
for j, mk in enumerate("osD"):
    n = n0 + j * TAU
    a.plot(t[n], ref[n], mk, ms=10, color=PINK, mec="white", zorder=5)
    a.annotate(["x(n)", "x(n+τ)", "x(n+2τ)"][j], (t[n], ref[n]), xytext=(4, 10), textcoords="offset points", color=PINK)
a.set_xlim(0, t[200]); a.set(xlabel="time", ylabel="x(t)")
a.set_title("(a) Read three samples, τ apart")
b = ax3d(fig, 132, "(b) ...they are one point in 3-D")
b.plot(*R.T, color=LINE, lw=1); b.scatter(*R[n0], s=90, color=PINK, edgecolor="white", depthshade=False)
c = ax3d(fig, 133, "(c) Sliding n draws the trajectory")
c.scatter(*R.T, c=np.arange(len(R)), cmap=TIME, s=6, depthshade=False)
fig.tight_layout(); save(fig, "01_signal_to_trajectory")

# ---------------------------------------------------------------- 02 tau and m
ami = P.average_mutual_information(ref, 60)
fnn = P.false_nearest_neighbours(ref, TAU, 5)
fig, ax = plt.subplots(1, 2, figsize=(12, 3.8), gridspec_kw={"width_ratios": [1.6, 1]})
ax[0].plot(ami, color=INK); ax[0].axvline(TAU, color=PINK, ls="--")
ax[0].annotate(f"first minimum: τ = {TAU}", (TAU, ami[TAU]), xytext=(TAU + 4, ami.max() * 0.75), color=PINK,
               arrowprops=dict(arrowstyle="-|>", color=PINK))
ax[0].set(xlabel="delay τ (samples)", ylabel="average mutual information"); ax[0].set_title("(a) Delay τ")
ax[1].bar(np.arange(1, 6), fnn * 100, color=[MUTED, MUTED, PINK, MUTED, MUTED], width=0.6)
for x_, v in zip(range(1, 6), fnn * 100):
    ax[1].text(x_, v + 1.5, f"{v:.1f}%", ha="center", fontsize=9)
ax[1].set(xlabel="embedding dimension m", ylabel="false nearest neighbours (%)", xticks=range(1, 6), ylim=(0, fnn.max() * 125))
ax[1].set_title("(b) Dimension m")
fig.tight_layout(); save(fig, "02_choosing_tau_m")

# ---------------------------------------------------------------- 03 CPST
cv, d = P.cpst(R, S, k=K, return_details=True)
i0 = int(np.argmax(d["E"]))
fig = plt.figure(figsize=(15, 4.6))
a = ax3d(fig, 131, "(a) One test point, k = 5 neighbours")
a.plot(*R.T, color=INK, lw=0.8, alpha=0.5); a.plot(*S.T, color=PINK, lw=0.8, alpha=0.5)
nb = R[d["neighbours"][i0]]; cen = d["centroids"][i0]; s0 = S[i0]
for q in nb:
    a.plot(*np.c_[s0, q], color=MUTED, lw=0.8, ls=":")
a.scatter(*nb.T, s=40, color=INK, depthshade=False, label="5 nearest reference points")
a.scatter(*cen, s=110, marker="D", color="white", edgecolor=INK, linewidth=1.5, depthshade=False, label="centroid r̄ᵢ")
a.scatter(*s0, s=110, color=PINK, edgecolor="white", depthshade=False, label="test point sᵢ")
a.plot(*np.c_[s0, cen], color=PINK, lw=2.5)
a.legend(loc="upper left", fontsize=8.5)
b = ax3d(fig, 132, "(b) Every test point, coloured by Eᵢ")
b.plot(*R.T, color=INK, lw=0.8, alpha=0.4)
sc = b.scatter(*S.T, c=d["E"], cmap=CMAP, s=10, depthshade=False)
fig.colorbar(sc, ax=b, shrink=0.6, pad=0.16).set_label("Eᵢ")
c = fig.add_subplot(133)
c.fill_between(np.arange(len(d["E"])), d["E"], color=PINK, alpha=0.15, lw=0)
c.plot(d["E"], color=PINK, lw=1.2); c.axhline(cv, color=INK, ls="--")
c.text(len(d["E"]) * 0.02, cv * 1.06, f"CPST = mean(Eᵢ) = {cv:.3f}", color=INK, fontweight="bold")
c.set(xlabel="test point i", ylabel="local deviation Eᵢ"); c.set_title("(c) CPST is the average")
fig.tight_layout(); save(fig, "03_cpst")

# ---------------------------------------------------------------- 04 Hausdorff
hv, h = P.hausdorff(R, S, q=Q, return_details=True)
hc = P.hausdorff(R, S, q=100)
fig = plt.figure(figsize=(13, 4.6))
a = ax3d(fig, 121, "(a) Nearest-neighbour gaps, both ways")
a.plot(*R.T, color=INK, lw=1); a.plot(*S.T, color=PINK, lw=1)
for i in range(0, len(S), 4):
    a.plot(*np.c_[S[i], R[h["nearest_r"][i]]], color=PINK, lw=0.5, alpha=0.7)
for j in range(2, len(R), 4):
    a.plot(*np.c_[R[j], S[h["nearest_s"][j]]], color=MUTED, lw=0.5, alpha=0.7)
iw = int(np.argmin(np.abs(h["d_sr"] - h["h_sr"])))
a.plot(*np.c_[S[iw], R[h["nearest_r"][iw]]], color=PINK, lw=3.5)
b = fig.add_subplot(122)
for dist, col, ls, lab in ((h["d_sr"], PINK, "-", "test → reference"), (h["d_rs"], INK, "--", "reference → test")):
    ds = np.sort(dist); b.plot(ds, np.arange(1, ds.size + 1) / ds.size, ls, color=col, label=lab)
b.axhline(Q / 100, color=MUTED, ls=":", lw=1); b.axvline(hv, color=PINK, lw=2)
b.axvline(hc, color=MUTED, lw=1.2, ls="-.")
b.text(hv, 0.25, f"d_H (Q95) = {hv:.3f}  ", color=PINK, fontweight="bold", ha="right")
b.text(hv, 0.12, f"classical = {hc:.3f}  ", color=MUTED, ha="right")
b.set(xlabel="distance to the nearest point of the other trajectory", ylabel="cumulative fraction", ylim=(0, 1.03))
b.set_title("(b) Keep the 95th percentile, take the larger direction"); b.legend(loc="upper left")
fig.tight_layout(); save(fig, "04_hausdorff")

# ---------------------------------------------------------------- 05 five cases
fig = plt.figure(figsize=(17, 6.6))
titles = {"shape": "Shape change", "global": "Global shrink", "local": "Local burst",
          "frequency": "Frequency shift", "noise": "Noise only"}
for j, kind in enumerate(KINDS):
    tt, r_, s_ = make_pair(kind, 0.3, 0.05 if kind == "noise" else 0.0, seed=2)
    c_, h_ = P.cpst(r_, s_, TAU, M, K), P.hausdorff(r_, s_, TAU, M, Q)
    a = fig.add_subplot(2, 5, j + 1)
    a.plot(tt, r_, color=INK, lw=1.3); a.plot(tt, s_, color=PINK, lw=1.3)
    a.set_xlim(tt[300], tt[500]); a.set_xticks([]); a.set_ylim(-1.9, 1.9)
    a.set_title(titles[kind])
    b = fig.add_subplot(2, 5, j + 6, projection="3d")
    b.plot(*P.embed(r_, TAU, M).T, color=INK, lw=0.9); b.plot(*P.embed(s_, TAU, M).T, color=PINK, lw=0.9)
    b.set_axis_off(); b.view_init(24, -52)
    b.set_title(f"CPST {c_:.3f}   ·   d_H {h_:.3f}", fontsize=10.5, color=INK, pad=-4)
fig.text(0.5, 0.005, "black: reference   ·   pink: test   ·   τ = 8, m = 3, k = 5, q = 95", ha="center", color=MUTED)
fig.tight_layout(); save(fig, "05_five_cases")

# ---------------------------------------------------------------- 06 severity response
sev = np.linspace(0, 0.5, 11)
fig, ax = plt.subplots(1, 4, figsize=(17, 3.9))
for a, kind in zip(ax[:3], ("shape", "global", "local")):
    cc, hh = [], []
    for s in sev:
        _, r_, s_ = make_pair(kind, s)
        cc.append(P.cpst(r_, s_, TAU, M, K)); hh.append(P.hausdorff(r_, s_, TAU, M, Q))
    a.plot(sev, cc, "o-", color=INK, ms=4, label="CPST"); a.plot(sev, hh, "s-", color=PINK, ms=4, label="d_H (Q95)")
    a.set(xlabel="severity"); a.set_title(titles[kind])
ax[0].legend(); ax[0].set_ylabel("index value")
noise = np.linspace(0, 0.2, 11); rng_c, rng_h = [], []
for sd in noise:
    vals = [(P.cpst(*make_pair("noise", 0, sd, seed=s)[1:], TAU, M, K),
             P.hausdorff(*make_pair("noise", 0, sd, seed=s)[1:], TAU, M, Q)) for s in range(10)]
    rng_c.append(np.mean([v[0] for v in vals])); rng_h.append(np.mean([v[1] for v in vals]))
ax[3].plot(noise, rng_c, "o-", color=INK, ms=4); ax[3].plot(noise, rng_h, "s-", color=PINK, ms=4)
ax[3].set(xlabel="noise std"); ax[3].set_title("Noise only: the healthy floor")
fig.tight_layout(); save(fig, "06_severity_response")

# ---------------------------------------------------------------- 07 outlier
_, r_, s_ = make_pair("noise", 0, 0.02, seed=4)
spk = s_.copy(); spk[333] += 2.5
vals = {name: (P.hausdorff(r_, x_, TAU, M, 100), P.hausdorff(r_, x_, TAU, M, Q), P.cpst(r_, x_, TAU, M, K))
        for name, x_ in (("clean", s_), ("spike", spk))}
fig, ax = plt.subplots(1, 2, figsize=(13, 3.9), gridspec_kw={"width_ratios": [1.4, 1]})
ax[0].plot(t, r_, color=INK, lw=1.2); ax[0].plot(t, spk, color=PINK, lw=1.2)
ax[0].annotate("one bad sample", (t[333], spk[333]), xytext=(t[333] + 2.5, 2.9), color=PINK,
               arrowprops=dict(arrowstyle="-|>", color=PINK))
ax[0].set(xlabel="time", ylabel="x(t)", ylim=(-2, 3.9)); ax[0].set_title("(a) Two healthy signals, one spike")
lab = ["classical\nHausdorff", "percentile\nHausdorff", "CPST"]
x = np.arange(3); w = 0.36
ax[1].bar(x - w / 2, vals["clean"], w, color=MUTED, label="clean")
ax[1].bar(x + w / 2, vals["spike"], w, color=PINK, label="with spike")
for xi, (a_, b_) in enumerate(zip(vals["clean"], vals["spike"])):
    ax[1].text(xi + w / 2, b_ + 0.04, f"{b_:.2f}", ha="center", fontsize=9, color=PINK)
ax[1].set_xticks(x); ax[1].set_xticklabels(lab); ax[1].legend(loc="upper right")
ax[1].set_title("(b) Only the classical form jumps")
fig.tight_layout(); save(fig, "07_outlier")
