#!/usr/bin/env python3
"""What the optimizer sees, drawn from one real search.

Dr. Yao's audience will largely not know Bayesian optimization, and the flow
chart on its own does not show the one thing that makes it work: a model of
the objective that gets sharper where it has looked, and stays uncertain
where it has not. This fits the same GP class the pipeline uses
(SingleTaskGP, Matern 5/2) to the aperture-versus-CNR slice of search
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

# Math as on the slides (Dr. Yao, 22 Oct; QA, 24 Oct): symbols in italic Times New Roman.
plt.rcParams.update({"mathtext.fontset": "custom", "mathtext.rm": "Times New Roman",
                     "mathtext.it": "Times New Roman:italic",
                     "mathtext.bf": "Times New Roman:bold",
                     "mathtext.bfit": "Times New Roman:bold:italic"})

UBB, CORAL = "#005BBB", "#E56D54"
INK, MUTED, GRID = "#1A1A1A", "#5A5A5A", "#D8D7D2"

HERE = os.path.dirname(os.path.abspath(__file__))
ARCHIVE = os.path.join(HERE, "..", "archive_backup", "h2h",
                       "results_h2h_5obj_r4.csv")
SEED_PREFIX = "lhs6d"          # the space-filling start, before any modelling
# Layout knobs, set from the checker: room above the data for legends.
HEADROOM = 0.15
C_TOP = 1.35


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


def style(ax):
    ax.tick_params(colors=MUTED, labelsize=12)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(GRID)
    ax.grid(axis="y", color=GRID, lw=0.7)
    ax.set_axisbelow(True)


def panel(ax, d, n, grid, title):
    sub = d.iloc[:n]
    gp, X, Y = fit_gp(sub.aperture_diam_mm.values, sub.cnr_sector_mean.values)
    m, s = posterior(gp, X, Y, grid)
    # s is the posterior SD of the fitted function (no measurement noise), so
    # the band is exactly mean +/- 2 SD.
    ax.fill_between(grid, m - 2 * s, m + 2 * s, color=UBB, alpha=0.16, lw=0,
                    label="model uncertainty (±2$\\sigma$)", zorder=2)
    ax.plot(grid, m, color=UBB, lw=2.2, label="model’s best guess", zorder=3)
    ax.scatter(sub.aperture_diam_mm, sub.cnr_sector_mean, s=30, color=INK,
               alpha=0.75, zorder=4, label=f"designs simulated ({n})")
    ax.set_title(title, fontsize=14, color=INK, pad=6, linespacing=1.12)
    ax.set_xlabel("aperture diameter (mm)", fontsize=13, color=MUTED)
    style(ax)
    return m, s


def in_band(a):
    """0.20 to 0.45 mm, both ends included: three designs sit exactly on the
    0.20 mm search bound, and a strict test silently dropped them."""
    return (a >= 0.20) & (a <= 0.45)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="search_process.png")
    ap.add_argument("--dpi", type=int, default=300)
    a = ap.parse_args()

    d, n_seed = load_arm(ARCHIVE)
    grid = np.linspace(d.aperture_diam_mm.min(), d.aperture_diam_mm.max(), 300)

    # Slide 19 box is 11.89 x 4.3 in, so this size is shown at scale 1 and the
    # font sizes are slide points. Panels are placed in inches: (c) needs a
    # wider gap for its own y label.
    FW, FH = 11.89, 4.3
    fig = plt.figure(figsize=(FW, FH))
    fig.patch.set_facecolor("white")
    LM, G1, G2, RM = 0.68, 0.48, 0.80, 0.38
    BOT, TOPM = 1.06, 0.86
    PW = (FW - LM - G1 - G2 - RM) / 3
    PH = FH - BOT - TOPM
    lefts = [LM, LM + PW + G1, LM + 2 * PW + G1 + G2]
    axes = [fig.add_axes([l / FW, BOT / FH, PW / FW, PH / FH]) for l in lefts]
    for ax in axes:
        ax.set_facecolor("white")

    m0, s0 = panel(axes[0], d, n_seed, grid,
                   f"(a)  after {n_seed} starting designs:\n"
                   f"a peak is already visible,\nloosely placed")
    m, s = panel(axes[1], d, len(d), grid,
                 f"(b)  after all {len(d)} designs:\n"
                 f"tight where it kept looking,\nloose where it stopped")
    axes[0].set_ylabel("CNR", fontsize=13, color=MUTED)
    lo = min(ax.get_ylim()[0] for ax in axes[:2])
    hi = max(ax.get_ylim()[1] for ax in axes[:2])
    for ax in axes[:2]:
        ax.set_ylim(lo, hi + HEADROOM)
    # Two legends: at slide size one three-entry legend has no free corner.
    # The model entries go in the empty upper right of (a), the data entry in
    # the upper right of (b), each clear of the points and the band.
    h, l = axes[0].get_legend_handles_labels()
    axes[0].legend(h[:2], ["model uncertainty\n(±2$\\sigma$)", l[1]],
                   frameon=False, fontsize=12, loc="upper right",
                   handlelength=1.4, handletextpad=0.5, borderaxespad=0.2)
    h, l = axes[1].get_legend_handles_labels()
    axes[1].legend(h[2:], l[2:], frameon=False, fontsize=12, loc="upper right",
                   handlelength=1.0, handletextpad=0.4, borderaxespad=0.2)

    # (c) where the budget went
    ax = axes[2]
    order = np.arange(len(d))
    ax.scatter(order[:n_seed], d.aperture_diam_mm[:n_seed], s=30, color=MUTED,
               alpha=0.8, label="starting designs")
    ax.scatter(order[n_seed:], d.aperture_diam_mm[n_seed:], s=30, color=UBB,
               alpha=0.85, label="chosen by the model")
    best = d.aperture_diam_mm[d.cnr_sector_mean.idxmax()]
    # A legend entry rather than a text label: every free spot near the
    # line is occupied by data points.
    ax.axhline(best, color=CORAL, lw=1.6, ls="--",
               label=f"best in this search:\n{best:.2f} mm")
    n_dense = int(in_band(d.aperture_diam_mm).sum())
    if n_dense != 80:
        raise SystemExit(f"ERROR: {n_dense} designs fall in 0.20 to 0.45 mm "
                         f"(inclusive), expected 80.")
    ax.set_title(f"(c)  where the simulation budget went:\n{n_dense} of {len(d)} "
                 f"designs fell\nbetween 0.20 and 0.45 mm", fontsize=14,
                 color=INK, pad=6, linespacing=1.12)
    ax.set_xlabel("evaluation number", fontsize=13, color=MUTED)
    ax.set_ylabel("aperture diameter (mm)", fontsize=13, color=MUTED)
    style(ax)
    ax.set_xlim(-5, 104)
    # Room above the data for the legend, clear of the one late proposal at
    # 0.77 mm; ticks stop at the 1.0 mm search bound.
    ax.set_ylim(0.15, C_TOP)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.legend(frameon=False, fontsize=12, loc="upper right",
              handlelength=1.4, handletextpad=0.5, borderaxespad=0.2)

    fig.text(0.5, 0.012,
             "A one-parameter Gaussian process, the same kind the search uses, "
             "refit to one search’s aperture and CNR data.\nThe search "
             "itself models all six parameters together.",
             ha="center", va="bottom", fontsize=12, color=MUTED, style="italic",
             linespacing=1.25)
    fig.savefig(a.out, dpi=a.dpi, facecolor="white")

    # The claim the slide makes is that uncertainty collapses where the search
    # looked. Check it rather than assert it in a caption.
    dense = in_band(d.aperture_diam_mm)
    near = in_band(grid)
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
    print(f"  {dense.sum()} of {len(d)} evaluations in 0.20-0.45 mm (inclusive)")
    print(f"  best in this search: {best:.4f} mm")
    print(f"  posterior sd there {s[near].mean():.3f} vs {s[far].mean():.3f} above 0.7 mm")


if __name__ == "__main__":
    main()
