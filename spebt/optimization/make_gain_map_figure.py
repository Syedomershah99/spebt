#!/usr/bin/env python3
"""Where in the phantom the contrast gain actually landed.

Slide 10 carried the per-rod gains as a column of numbers beside the bar
chart, which repeats what the bars already say. This puts the same numbers
on the object instead, so the audience sees that the improvement sits in the
smallest rod groups rather than spread evenly.

Rod radii and both CNR series are imported from make_reference_figure so
there is one source of truth for the numbers on the slide.

Usage:  python make_gain_map_figure.py [--out gain_map.png]
"""
import argparse
import os

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from scipy import ndimage

from make_reference_figure import ROD_RADII, REF_MEAN, OURS_MEAN

PHANTOM = ("/Users/Omer/Desktop/RA/pyrecon/custom_phantoms/"
           "hot_rods_phantom_10.0_mm_x_10.0_mm.pt")
INK, MUTED = "#1A1A1A", "#5A5A5A"


def load_phantom(path):
    d = torch.load(path, weights_only=False)
    if hasattr(d, "item"):          # some phantoms are saved as a 0-d object array
        d = d.item()
    img = d["Phantom tensor"].numpy()
    radii = d["Metadata"]["rods radii in mm"]
    mm_per_px = d["Metadata"]["mm per pixel"][0]
    if len(radii) != len(ROD_RADII) or not np.allclose(radii, ROD_RADII, atol=1e-5):
        raise SystemExit(f"ERROR: phantom radii {radii} do not match the radii "
                         f"the CNR numbers were measured on ({ROD_RADII}). "
                         f"The gains would be put on the wrong rods.")
    return img, mm_per_px


def group_rods(img, mm_per_px):
    """Assign every rod to one of the six radii by its measured area."""
    lbl, n = ndimage.label(img > 0.5 * img.max())
    if n < len(ROD_RADII):
        raise SystemExit(f"ERROR: found only {n} rods in the phantom.")
    areas = ndimage.sum(np.ones_like(img), lbl, range(1, n + 1))
    r_px = np.sqrt(areas / np.pi)
    r_mm = r_px * mm_per_px
    # Nearest of the six nominal radii, in mm.
    which = np.argmin(np.abs(r_mm[:, None] - np.array(ROD_RADII)[None, :]), axis=1)
    groups = {}
    for g in range(len(ROD_RADII)):
        ids = np.flatnonzero(which == g) + 1
        if ids.size == 0:
            raise SystemExit(f"ERROR: no rods matched radius {ROD_RADII[g]} mm.")
        groups[g] = (np.isin(lbl, ids), ndimage.center_of_mass(
            np.isin(lbl, ids).astype(float)))
    return groups


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="gain_map.png")
    ap.add_argument("--dpi", type=int, default=220)
    a = ap.parse_args()

    img, mm_per_px = load_phantom(PHANTOM)
    groups = group_rods(img, mm_per_px)
    gains = [100 * (o / r - 1) for o, r in zip(OURS_MEAN, REF_MEAN)]

    cmap = LinearSegmentedColormap.from_list(
        "gain", ["#DCE6F2", "#7FA8D4", "#2A78D6", "#14508F"])
    norm = Normalize(vmin=0, vmax=max(gains))

    canvas = np.zeros(img.shape + (4,))
    for g, (mask, _) in groups.items():
        canvas[mask] = cmap(norm(gains[g]))

    fig, ax = plt.subplots(figsize=(5.2, 5.4))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.imshow(canvas, interpolation="nearest")
    for g, (_, (cy, cx)) in groups.items():
        # Push the label well outside its own sector; at 0.30 of the width it
        # landed on the rods it was labelling.
        h, w = img.shape
        v = np.array([cx - w / 2, cy - h / 2], dtype=float)
        v = v / (np.linalg.norm(v) + 1e-9) * (w * 0.52)
        ax.annotate(f"{ROD_RADII[g]:.3f} mm\n+{gains[g]:.0f}%", xy=(cx, cy),
                    xytext=(w / 2 + v[0], h / 2 + v[1]),
                    ha="center", va="center", fontsize=11.5, fontweight="bold",
                    color=INK, linespacing=1.25,
                    bbox=dict(boxstyle="round,pad=0.22", fc="white",
                              ec="#D8D7D2", lw=0.8),
                    arrowprops=dict(arrowstyle="-", color="#B9BDC2", lw=0.9))
    ys, xs = np.nonzero(img > 0.5 * img.max())
    pad = 0.30 * max(np.ptp(xs), np.ptp(ys))
    ax.set_xlim(xs.min() - pad, xs.max() + pad)
    ax.set_ylim(ys.max() + pad, ys.min() - pad)
    ax.axis("off")
    ax.set_title("Gain by rod size", fontsize=14, fontweight="bold",
                 color=INK, pad=8)
    peak = int(np.argmax(gains))
    fig.text(0.5, 0.045,
             f"Gain peaks at {ROD_RADII[peak]:.3f} mm, among the finest rods "
             f"either\ngeometry resolves. More counts alone would lift all six "
             f"alike.",
             ha="center", fontsize=10.5, color=MUTED)
    fig.tight_layout(rect=(0, 0.11, 1, 0.97))
    fig.savefig(a.out, dpi=a.dpi, facecolor="white")
    # The caption claims an interior peak. The smallest rods sit near the
    # resolution limit of BOTH geometries, so they do not gain the most --
    # saying "smallest gain most" would be wrong, and was, in a first draft.
    if peak in (0, len(gains) - 1):
        raise SystemExit(f"ERROR: gain peaks at an end of the range "
                         f"({ROD_RADII[peak]:.3f} mm); the caption claims an "
                         f"interior peak.")
    print(f"wrote {a.out}")
    for g, r in enumerate(ROD_RADII):
        print(f"  {r:.3f} mm  +{gains[g]:.0f}%  "
              f"{int(groups[g][0].sum())} px")


if __name__ == "__main__":
    main()
