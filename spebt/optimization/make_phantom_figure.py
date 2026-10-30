#!/usr/bin/env python3
"""Slide 6's hot-rod phantom, each sector marked with its rod diameter.

Dr. Yao, R11-rY (30 Oct 2026): "Since later diameters of rods are used, better to
introduce diameters here. You may mark the diameter numbers next to each sector."
The pipeline's own phantom (the one every CNR was measured on), black rods on white,
drawn as the array is stored (x on the first index, down the page), as slides 12 and
20 draw it. The diameters are the phantom's nominal ones (twice its rod radii); the
pipeline thresholds the anti-aliased image at 0.1, so the simulated rods are 6-16%
wider (see mobo_verified_results).

Usage:  python make_phantom_figure.py [--out ../../deck_figures/hotrod_phantom_d.png]
"""
import argparse
import os

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import ndimage

HERE = os.path.dirname(os.path.abspath(__file__))
PHANTOM = os.path.join(HERE, "..", "data", "sai_10mm", "hot_rods_phantom_10.0_mm_x_10.0_mm.pt")
RADII = (0.100, 0.125, 0.150, 0.175, 0.200, 0.225)      # mm, the phantom's six sectors
MM_PER_PX = 0.05
INK, MUTED = "#1A1A1A", "#5A5A5A"
FIG_W, FIG_H = 1.96, 1.42                                 # in, as placed on slide 6 (below its PPDS symbols)
FS = 9.5
VIEW_V = 108                                               # px above and below the centre: rods (80) plus labels


def sectors(img):
    """Rod groups by size: {radius: (rod centroids (row, col) in px)}."""
    lab, n = ndimage.label(img > 0.1)
    idx = range(1, n + 1)
    act = np.asarray(ndimage.sum(img, lab, idx))            # activity = nominal area in px
    r_mm = np.sqrt(act / np.pi) * MM_PER_PX
    cen = np.asarray(ndimage.center_of_mass(img, lab, idx))
    out = {}
    for r in RADII:
        k = np.abs(r_mm - r) < 0.0125
        out[r] = cen[k]
    assert sum(len(v) for v in out.values()) == n == 58, "rods not all assigned to a sector"
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default=os.path.join(HERE, "..", "..", "deck_figures", "hotrod_phantom_d.png"))
    a = ap.parse_args()
    d = torch.load(PHANTOM, weights_only=True, map_location="cpu")
    img = (d["Phantom tensor"] if isinstance(d, dict) else d).numpy().astype(float)
    groups = sectors(img)
    c = (np.array(img.shape) - 1) / 2.0
    fig = plt.figure(figsize=(FIG_W, FIG_H))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(1 - np.clip(img, 0, 1), cmap="gray", vmin=0, vmax=1, interpolation="antialiased")
    view_h = VIEW_V * FIG_W / FIG_H
    ax.set_xlim(c[1] - view_h, c[1] + view_h)
    ax.set_ylim(c[0] + VIEW_V, c[0] - VIEW_V)
    ax.axis("off")
    for r, cen in groups.items():
        v = cen.mean(axis=0) - c                             # (row, col) from the centre
        u = v / np.linalg.norm(v)
        reach = np.max(np.linalg.norm(cen - c, axis=1)) + r / MM_PER_PX
        # clear of the rods by 8 px plus the label's half-extent along u (about 22 px wide, 9 tall)
        row, col = c + u * (reach + 8 + 22 * abs(u[1]) + 9 * abs(u[0]))
        ax.text(col, row, f"{2 * r:.2f}", ha="center", va="center", fontsize=FS,
                color=INK, fontweight="bold")
    fig.savefig(a.out, dpi=600, facecolor="white")
    print("wrote", a.out, "| sector diameters:", ", ".join(f"{2 * r:.2f}" for r in RADII))


if __name__ == "__main__":
    main()
