#!/usr/bin/env python3
"""What the optimizer sees, drawn from one real search.

Dr. Yao's audience will largely not know Bayesian optimization, and the flow
chart on its own does not show the one thing that makes it work: a model of
the objective that gets sharper where it has looked, and stays uncertain
where it has not. This fits the same GP class the pipeline uses
(SingleTaskGP, Matern 5/2) to the aperture-versus-contrast slice of search
arm r4 at two points in its life, and shows where the evaluations went.

This is ONE axis of a six-dimensional, five-objective model. It is an
illustration of the mechanism, not the model itself, and the figure says so.

Usage:  python make_search_figure.py [--out search_process.png]
"""
import argparse
import os
import re
import warnings

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from gpytorch.mlls import ExactMarginalLogLikelihood

UBB, TEAL, CORAL = "#005BBB", "#007681", "#E56D54"
INK, MUTED, GRID = "#1A1A1A", "#5A5A5A", "#D8D7D2"

HERE = os.path.dirname(os.path.abspath(__file__))
ARCHIVE = os.path.join(HERE, "..", "archive_backup", "h2h",
                       "results_h2h_5obj_r4.csv")
SEED_PREFIX = "lhs6d"          # the space-filling start, before any modelling


def load_arm(path):
    """Evaluations in the order the search ran them: seeds first, then each
    proposal by its index in the config name."""
    d = pd.read_csv(path).dropna(subset=["aperture_diam_mm", "cnr_sector_mean"])
    idx = d["config"].str.extract(r"_(\d{4})_")[0]
    d = d.assign(
        is_seed=d["config"].str.startswith(SEED_PREFIX),
        prop_idx=pd.to_numeric(idx, errors="coerce"))
    seeds = d[d.is_seed]
    props = d[~d.is_seed].sort_values("prop_idx")
    if seeds.empty or props.empty:
        raise SystemExit(f"ERROR: {path} has {len(seeds)} seed rows and "
                         f"{len(props)} proposal rows; expected both.")
    return pd.concat([seeds, props], ignore_index=True), len(seeds)


def fit_gp(x, y):
    """Same model class and kernel the search config names."""
    X = torch.tensor(x, dtype=torch.double).unsqueeze(-1)
    Y = torch.tensor(y, dtype=torch.double).unsqueeze(-1)
    Xn = (X - X.min()) / (X.max() - X.min())
    Yn = (Y - Y.mean()) / Y.std()
    gp = SingleTaskGP(Xn, Yn)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit_gpytorch_mll(ExactMarginalLogLikelihood(gp.likelihood, gp))
    return gp, X, Y


def posterior(gp, X, Y, grid):
    G = torch.tensor(grid, dtype=torch.double).unsqueeze(-1)
    Gn = (G - X.min()) / (X.max() - X.min())
    with torch.no_grad():
        p = gp.posterior(Gn)
        m = p.mean.squeeze(-1).numpy() * Y.std().item() + Y.mean().item()
        s = p.variance.sqrt().squeeze(-1).numpy() * Y.std().item()
    return m, s


def panel(ax, d, n, grid, title):
    sub = d.iloc[:n]
    gp, X, Y = fit_gp(sub.aperture_diam_mm.values, sub.cnr_sector_mean.values)
    m, s = posterior(gp, X, Y, grid)
    ax.fill_between(grid, m - 2 * s, m + 2 * s, color=UBB, alpha=0.16, lw=0,
                    label="model uncertainty", zorder=2)
    ax.plot(grid, m, color=UBB, lw=2.2, label="model's best guess", zorder=3)
    ax.scatter(sub.aperture_diam_mm, sub.cnr_sector_mean, s=26, color=INK,
               alpha=0.75, zorder=4, label=f"designs simulated ({n})")
    ax.set_title(title, fontsize=12.5, color=INK, pad=8)
    ax.set_xlabel("aperture diameter (mm)", fontsize=10.5, color=MUTED)
    ax.tick_params(colors=MUTED, labelsize=9.5)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(GRID)
    ax.grid(axis="y", color=GRID, lw=0.7)
    ax.set_axisbelow(True)
    return m, s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="search_process.png")
    ap.add_argument("--dpi", type=int, default=200)
    a = ap.parse_args()

    d, n_seed = load_arm(ARCHIVE)
    grid = np.linspace(d.aperture_diam_mm.min(), d.aperture_diam_mm.max(), 300)

    fig, axes = plt.subplots(1, 3, figsize=(15.2, 4.5))
    fig.patch.set_facecolor("white")
    for ax in axes:
        ax.set_facecolor("white")

    m0, s0 = panel(axes[0], d, n_seed, grid,
                   f"(a)  after {n_seed} space-filling designs\n"
                   f"a peak is already visible, loosely placed")
    m, s = panel(axes[1], d, len(d), grid,
                 f"(b)  after all {len(d)} designs\n"
                 f"tight where it kept looking, loose where it stopped")
    axes[0].set_ylabel("reconstructed contrast (CNR)", fontsize=10.5, color=MUTED)
    lo = min(ax.get_ylim()[0] for ax in axes[:2])
    hi = max(ax.get_ylim()[1] for ax in axes[:2])
    for ax in axes[:2]:
        ax.set_ylim(lo, hi)
    axes[1].legend(frameon=False, fontsize=9.5, loc="lower center")

    # (c) where the budget went
    ax = axes[2]
    order = np.arange(len(d))
    ax.scatter(order[:n_seed], d.aperture_diam_mm[:n_seed], s=26, color=MUTED,
               alpha=0.8, label="space-filling start")
    ax.scatter(order[n_seed:], d.aperture_diam_mm[n_seed:], s=26, color=TEAL,
               alpha=0.85, label="model-proposed")
    best = d.aperture_diam_mm[d.cnr_sector_mean.idxmax()]
    ax.axhline(best, color=CORAL, lw=1.6, ls="--")
    ax.text(len(d) * 0.02, best - 0.035, f"best found: {best:.2f} mm", color=CORAL,
            fontsize=10, va="top", ha="left", fontweight="bold")
    n_dense = int(((d.aperture_diam_mm > 0.2) & (d.aperture_diam_mm < 0.45)).sum())
    ax.set_title(f"(c)  where the simulation budget went\n{n_dense} of {len(d)} "
                 f"designs fell between 0.20 and 0.45 mm", fontsize=12.5,
                 color=INK, pad=8)
    ax.set_xlabel("evaluation number", fontsize=10.5, color=MUTED)
    ax.set_ylabel("aperture diameter (mm)", fontsize=10.5, color=MUTED)
    ax.tick_params(colors=MUTED, labelsize=9.5)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(GRID)
    ax.grid(axis="y", color=GRID, lw=0.7)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=9.5, loc="upper right")

    fig.text(0.5, 0.005,
             "One slice of a six-parameter, five-objective model, shown to "
             "explain the mechanism. The search itself never looks at a single "
             "axis this way.",
             ha="center", fontsize=9.5, color=MUTED, style="italic")
    fig.tight_layout(rect=(0, 0.035, 1, 1))
    fig.savefig(a.out, dpi=a.dpi, facecolor="white")

    # The claim the slide makes is that uncertainty collapses where the search
    # looked. Check it rather than assert it in a caption.
    dense = (d.aperture_diam_mm > 0.2) & (d.aperture_diam_mm < 0.45)
    near = (grid > 0.2) & (grid < 0.45)
    far = grid > 0.7
    if not s[near].mean() < s[far].mean():
        raise SystemExit(f"ERROR: uncertainty is not lower where the search "
                         f"concentrated ({s[near].mean():.3f} vs "
                         f"{s[far].mean():.3f}); the caption would be wrong.")
    if not s[near].mean() < s0[near].mean():
        raise SystemExit(f"ERROR: panel (b) is not tighter than panel (a) in the "
                         f"region the search worked ({s[near].mean():.3f} vs "
                         f"{s0[near].mean():.3f}); caption (b) would be wrong.")
    print(f"wrote {a.out}")
    print(f"  {n_seed} seeds + {len(d) - n_seed} proposals")
    print(f"  {dense.sum()} of {len(d)} evaluations in 0.20-0.45 mm")
    print(f"  posterior sd there {s[near].mean():.3f} vs {s[far].mean():.3f} above 0.7 mm")


if __name__ == "__main__":
    main()
