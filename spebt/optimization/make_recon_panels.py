#!/usr/bin/env python3
"""Slide 12's two reconstructions with a shared colour bar, as click-build layers.

Dr. Yao, R11-rY (30 Oct 2026): "Better to have color palette bar to be consistent with
other slides." One noise realization (seed 0) of each design through the same
150-iteration ML-EM, one display window for both (to the 99.5th percentile of the two),
the colour bar vertical at the right as on slides 13 to 15. The arrays are drawn as
stored (x on the first index, down the page), as the phantom on slides 6 and 20.
Layer 1 is the benchmark with the bar (click 1), layer 2 the MOBO-optimized design.

Inputs: cnr_repeat_seed0/recon_mlem_T8.npz of results/tmi_reference_000 and of
results/nd2fine_960 (mobo_0296's parameters re-simulated; its seed-0 CNR results equal
mobo_0296's, whose own reconstruction was purged), copied from CCR.

  python make_recon_panels.py --bench BENCH.npz --mobo MOBO.npz [--outdir ../../deck_figures]
"""
import argparse
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from make_mpxi_optimum_figure import write_layers, INK, INK_MUTED, GRID

FIG_W, FIG_H, DPI = 5.82, 2.75, 300
SIDE, GAP = 2.55, 0.12                       # in: each panel, and between the two
CX = (SIDE / 2, SIDE + GAP + SIDE / 2)       # panel centres, in figure inches (the slide labels use them)
FS_TICK, FS_LABEL = 11, 12                   # as the bars on slides 13 to 15


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--bench", required=True)
    ap.add_argument("--mobo", required=True)
    ap.add_argument("--outdir", default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                     "..", "..", "deck_figures"))
    a = ap.parse_args()
    imgs = [np.load(p)["final"] for p in (a.bench, a.mobo)]
    vmax = float(np.percentile(np.concatenate([i.ravel() for i in imgs]), 99.5))
    fig = plt.figure(figsize=(FIG_W, FIG_H))
    y0 = (FIG_H - SIDE) / 2
    axes = [fig.add_axes([(cx - SIDE / 2) / FIG_W, y0 / FIG_H, SIDE / FIG_W, SIDE / FIG_H]) for cx in CX]
    for ax, img in zip(axes, imgs):
        im = ax.imshow(img, cmap="hot", vmin=0, vmax=vmax, interpolation="nearest")
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_color(GRID)
    cbar = fig.add_axes([(2 * SIDE + GAP + 0.1) / FIG_W, y0 / FIG_H, 0.09 / FIG_W, SIDE / FIG_H])
    cb = fig.colorbar(im, cax=cbar)
    cb.ax.tick_params(labelsize=FS_TICK - 1, colors=INK_MUTED, length=2, pad=1)
    cb.outline.set_edgecolor(GRID)
    cb.set_label("reconstructed counts per pixel", fontsize=FS_LABEL - 1, color=INK, labelpad=4)
    layers = {"benchmark": [axes[0], cbar], "mobo": [axes[1]]}
    write_layers(fig, axes[0], axes + [cbar], layers, os.path.join(a.outdir, "recon_panels.png"), DPI)
    print(f"display window 0 to {vmax:.1f} counts per pixel")


if __name__ == "__main__":
    main()
