#!/usr/bin/env python
"""Render the MFM validation plots from the profile dumps test_mfm writes to plots/.

    ./test_mfm && ./make_test_plots.py       # -> plots/sod.png, plots/soundwave.png

Two series everywhere -- computed (points) vs exact (line) -- so identity never rides on
color alone: the exact solution is always the dark line, the data always discrete markers,
and each panel carries its own legend.
"""
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
PLOTS = os.path.join(HERE, "plots")

# two fixed roles, never cycled: data = mid blue, exact = near-black
C_DATA, C_EXACT = "#3B6FB6", "#222222"

plt.rcParams.update({
    "figure.dpi": 130, "savefig.dpi": 130, "font.size": 9,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.5,
    "axes.spines.top": False, "axes.spines.right": False,
})


def sod():
    prof = np.loadtxt(os.path.join(PLOTS, "sod_profile.txt"))
    ex = np.loadtxt(os.path.join(PLOTS, "sod_exact.txt"))
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.2), constrained_layout=True)
    for ax, col, name in zip(axes, (1, 2, 3), (r"$\rho$", r"$v_x$", r"$P$")):
        ax.plot(prof[:, 0], prof[:, col], ".", ms=1.5, color=C_DATA, alpha=0.35,
                label="MFM (particles)", rasterized=True)
        ax.plot(ex[:, 0], ex[:, col], "-", lw=1.6, color=C_EXACT, label="exact")
        ax.set_xlabel("x")
        ax.set_ylabel(name)
        ax.set_xlim(0.1, 0.9)
        # the scored window: outside it the periodic wrap's inverted discontinuity intrudes
        for xw in (0.2, 0.8):
            ax.axvline(xw, color="0.55", lw=0.8, ls=":")
    axes[0].legend(loc="upper right", frameon=False, fontsize=8, markerscale=6)
    fig.suptitle("Sod shocktube, t = 0.1 (147k cells; dotted = scored window)", fontsize=10)
    fig.savefig(os.path.join(PLOTS, "sod.png"))
    plt.close(fig)


def soundwave():
    conv = np.loadtxt(os.path.join(PLOTS, "soundwave_convergence.txt"))
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(8.2, 3.2), constrained_layout=True)

    w = np.loadtxt(os.path.join(PLOTS, "soundwave_n32.txt"))
    o = np.argsort(w[:, 0])
    a1.plot(w[o, 0], w[o, 1], ".", ms=1.5, color=C_DATA, alpha=0.35,
            label="MFM $32^3$", rasterized=True)
    a1.plot(w[o, 0], w[o, 2], "-", lw=1.6, color=C_EXACT, label="exact")
    a1.set_xlabel("x")
    a1.set_ylabel(r"$v_x / A$")
    a1.set_title("wave after half period", fontsize=9)
    a1.legend(loc="upper right", frameon=False, fontsize=8, markerscale=6)

    n, l1 = conv[:, 0], conv[:, 1]
    a2.loglog(n, l1, "o-", ms=5, lw=1.4, color=C_DATA, label="measured")
    for p, ls in ((1, ":"), (2, "--")):
        a2.loglog(n, l1[0] * (n / n[0]) ** (-p), ls, lw=1.0, color="0.55",
                  label=f"$\\propto N^{{-{p}}}$")
    a2.set_xlabel(r"$N^{1/3}$")
    a2.set_ylabel(r"$L_1(v)/A$")
    a2.set_title("convergence", fontsize=9)
    a2.set_xticks(n)
    a2.set_xticklabels([f"{int(v)}" for v in n])
    a2.legend(frameon=False, fontsize=8)
    fig.suptitle("Travelling sound wave, A = $10^{-4}$", fontsize=10)
    fig.savefig(os.path.join(PLOTS, "soundwave.png"))
    plt.close(fig)


if __name__ == "__main__":
    sod()
    soundwave()
    print(f"wrote {PLOTS}/sod.png and {PLOTS}/soundwave.png")
