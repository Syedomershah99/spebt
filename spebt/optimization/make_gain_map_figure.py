#!/usr/bin/env python3
"""Where in the phantom the CNR gain actually landed.

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

# Math as on the slides (Dr. Yao, 22 Oct; QA, 24 Oct): symbols in italic Times New Roman.
plt.rcParams.update({"mathtext.fontset": "custom", "mathtext.rm": "Times New Roman",
                     "mathtext.it": "Times New Roman:italic",
                     "mathtext.bf": "Times New Roman:bold",
                     "mathtext.bfit": "Times New Roman:bold:italic"})

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


def edge_point(mask, target):
    """Where a leader line from `target` (x, y) should stop: on the rim of
    the nearest rod of this group, not at the group's centre. The nearest rod
    pixel is moved half a pixel towards the target, onto the rod's edge."""
    yy, xx = np.nonzero(mask)
    i = int(np.argmin((xx - target[0]) ** 2 + (yy - target[1]) ** 2))
    p = np.array([xx[i], yy[i]], dtype=float)
    u = np.asarray(target, dtype=float) - p
    return p + 0.5 * u / (np.linalg.norm(u) + 1e-9)


def layout(groups, rods_xy, box_in, ax_in, gap_in):
    """Largest phantom scale (inches per pixel) and one common label radius
    such that every label box clears every rod by gap_in and the rods plus
    labels fit the axes. Labels sit on the ray from the centre of the rod
    field through each group's centroid."""
    x0, y0 = rods_xy.min(0)
    x1, y1 = rods_xy.max(0)
    c = np.array([(x0 + x1) / 2, (y0 + y1) / 2])
    dirs = {}
    for g, (_, (cy, cx)) in groups.items():
        v = np.array([cx, cy]) - c
        dirs[g] = v / np.linalg.norm(v)

    def fits(scale):
        bw, bh = box_in[0] / scale / 2, box_in[1] / scale / 2
        gp = gap_in / scale
        for R in np.arange(0, 2000, 1.0):
            ok = True
            for v in dirs.values():
                px, py = c + R * v
                hit = ((np.abs(rods_xy[:, 0] - px) <= bw + gp) &
                       (np.abs(rods_xy[:, 1] - py) <= bh + gp))
                if hit.any():
                    ok = False
                    break
            if ok:
                break
        pos = {g: c + R * v for g, v in dirs.items()}
        lo = np.minimum([x0, y0], np.min([p - [bw, bh] for p in pos.values()], 0))
        hi = np.maximum([x1, y1], np.max([p + [bw, bh] for p in pos.values()], 0))
        span = (hi - lo) * scale
        return span[0] <= ax_in[0] and span[1] <= ax_in[1], pos, lo, hi

    lo_s, hi_s = 0.002, 0.02
    for _ in range(40):
        mid = (lo_s + hi_s) / 2
        if fits(mid)[0]:
            lo_s = mid
        else:
            hi_s = mid
    _, pos, lo, hi = fits(lo_s)
    return lo_s, pos, lo, hi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="gain_map.png")
    ap.add_argument("--dpi", type=int, default=300)
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

    # Slide 17 box is about 5.6 x 5.3 in. Font sizes leave room for the
    # picture to be fitted slightly smaller (5.4 x 5.1 in, scale 0.96) and
    # still read at 12 pt or more.
    FW, FH = 5.6, 5.3
    fig = plt.figure(figsize=(FW, FH))
    fig.patch.set_facecolor("white")
    AX = [0.0, 0.095, 1.0, 0.805]              # figure fractions
    ax = fig.add_axes(AX)
    ax.set_facecolor("white")
    ax.imshow(canvas, interpolation="nearest")

    # Diameters, as slide 6 introduces the rods (Dr. Yao, R11-rY, 30 Oct 2026).
    labels = {g: f"$\\mathbfit{{d}}$ = {2 * ROD_RADII[g]:.2f} mm\n+{gains[g]:.0f}%" for g in groups}
    style = dict(ha="center", va="center", fontsize=13, fontweight="bold",
                 color=INK, linespacing=1.25,
                 bbox=dict(boxstyle="round,pad=0.25", fc="white",
                           ec="#D8D7D2", lw=0.9))
    # Measure the largest label box once, in inches.
    r = fig.canvas.get_renderer()
    box_in = np.zeros(2)
    for t in labels.values():
        probe = fig.text(0.5, 0.5, t, **style)
        probe.update_bbox_position_size(r)
        bb = probe.get_bbox_patch().get_window_extent(r)
        box_in = np.maximum(box_in, [bb.width / fig.dpi, bb.height / fig.dpi])
        probe.remove()

    rods = img > 0.5 * img.max()
    yy, xx = np.nonzero(rods)
    rods_xy = np.c_[xx, yy].astype(float)
    margin = 0.10                               # inches, all round
    ax_in = (AX[2] * FW - 2 * margin, AX[3] * FH - 2 * margin)
    scale, pos, lo, hi = layout(groups, rods_xy, box_in, ax_in, gap_in=0.10)

    for g, (mask, _) in groups.items():
        ax.annotate(labels[g], xy=edge_point(mask, pos[g]), xytext=pos[g],
                    arrowprops=dict(arrowstyle="-", color="#B9BDC2", lw=1.1,
                                    shrinkA=0, shrinkB=0), **style)
    # Centre the rods-plus-labels block in the axes at the chosen scale.
    cx, cy = (lo + hi) / 2
    half_w = AX[2] * FW / scale / 2
    half_h = AX[3] * FH / scale / 2
    ax.set_xlim(cx - half_w, cx + half_w)
    ax.set_ylim(cy + half_h, cy - half_h)
    ax.axis("off")

    fig.text(0.5, 0.975, "CNR gain by rod diameter", ha="center", va="top",
             fontsize=16, fontweight="bold", color=INK)
    # Key: a light-to-dark swatch in the same colour scale as the rods.
    key = "darker rods = larger gain"
    kt = fig.text(0, 0, key, fontsize=12.5, color=MUTED, va="center")
    kw = kt.get_window_extent(r).width / fig.dpi
    kt.remove()
    sw, sh, sp = 0.55, 0.13, 0.10               # swatch w, h, spacing (in)
    x0 = (FW - (sw + sp + kw)) / 2
    yk = 0.045
    kax = fig.add_axes([x0 / FW, yk - sh / FH / 2, sw / FW, sh / FH])
    kax.imshow(np.linspace(0, max(gains), 256)[None, :], cmap=cmap, norm=norm,
               aspect="auto", interpolation="bilinear")
    kax.set_xticks([]); kax.set_yticks([])
    for spn in kax.spines.values():
        spn.set_color("#D8D7D2")
    fig.text((x0 + sw + sp) / FW, yk, key, fontsize=12.5, color=MUTED,
             va="center", ha="left")

    peak = int(np.argmax(gains))
    fig.savefig(a.out, dpi=a.dpi, facecolor="white")
    # The caption claims an interior peak. The smallest rods sit near the
    # resolution limit of BOTH geometries, so they do not gain the most --
    # saying "smallest gain most" would be wrong, and was, in a first draft.
    if peak in (0, len(gains) - 1):
        raise SystemExit(f"ERROR: gain peaks at an end of the range "
                         f"({ROD_RADII[peak]:.3f} mm); the caption claims an "
                         f"interior peak.")
    print(f"wrote {a.out}")
    print(f"  phantom scale {scale * 100:.3f} in per 100 px, label box "
          f"{box_in[0]:.2f} x {box_in[1]:.2f} in")
    for g, rr in enumerate(ROD_RADII):
        print(f"  {rr:.3f} mm  +{gains[g]:.0f}%  "
              f"{int(groups[g][0].sum())} px")


if __name__ == "__main__":
    main()
