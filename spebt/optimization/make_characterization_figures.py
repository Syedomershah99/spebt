#!/usr/bin/env python3
"""Slides 13-15: the benchmark beside the MOBO-optimized design (Dr. Yao, R10-rY, 29 Oct 2026).

Reads the two npz files of compute_design_maps.py and writes, for each slide, the two
click-build layers the builder places in the 5.82 x 3.55 in box to the right of the
table (so the figure is shown at scale 1 and these font sizes are the sizes seen):
the benchmark's column, then the MOBO-optimized column with the shared colour bars.
Laid over white in order they give the full figure (write_layers checks it).

  char_hist        strip FWHM histogram (strips above the 1% sensitivity floor; the
                   FWHM filter at 0.45 mm marked) and strips per detector with the filter
  char_asci        per-pixel ASCI with the filter, 1 degree bins (asci_pct_fwhm0p45)
  char_sens_ppds   sensitivity, sum_i PPDF_ij, and PPDS (--ppds eq6 or ry), all rings

Both designs share each row's axis or colour scale, so the columns compare directly.

  python make_characterization_figures.py --bench bench.npz --mobo mobo.npz \\
      --outdir ../../deck_figures [--ppds ry]
"""
import argparse
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from make_mpxi_optimum_figure import write_layers, INK, INK_MUTED, GRID

FIG_W, FIG_H, DPI = 5.82, 3.55, 300
FS_HEAD, FS_LABEL, FS_TICK, FS_NOTE = 11.5, 11, 10, 10
C_BAR, C_FILTER = "#2a78d6", "#eb6834"
CMAP = "viridis"
PPDS_SCALE = {"ry_mask": 1e-3, "eq6": 1e2}        # shown as x10^3 and x10^-2 per mm^2
PPDS_BAR = {"ry_mask": "×10³ mm⁻²", "eq6": "×10⁻² mm⁻²"}
N = 200                     # 0.05 mm pixels; pixel p is (x, y) = (p // 200, p % 200)
EXTENT = (-5, 5, -5, 5)


def load(path):
    z = np.load(path)
    return {k: z[k] for k in z.files}


def style(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_MUTED, labelsize=FS_TICK, length=3)


def column_axes(fig, rows, top=0.97, bottom=0.12, gap=0.13, left=(0.12, 0.60), width=0.36):
    """Two columns of `rows` axes in figure fractions; the right column ends at 0.96."""
    h = (top - bottom - gap * (rows - 1)) / rows
    return [[fig.add_axes([x, top - (r + 1) * h - r * gap, width, h]) for r in range(rows)]
            for x in left]


def hist_figure(d, out):
    fig = plt.figure(figsize=(FIG_W, FIG_H))
    cols = column_axes(fig, 2, top=0.93, bottom=0.12, gap=0.22, left=(0.11, 0.60), width=0.36)
    edges = np.arange(0, 2.0001, 0.1)
    ymax_f = ymax_m = 0
    stats = []
    for c, key in enumerate(("bench", "mobo")):
        s, k = d[key]["strips"], d[key]["dets"]
        fw = s[(s[:, 5] > 0), 2]                       # floor & finite FWHM > 0
        h, _ = np.histogram(np.clip(fw, 0, 1.9999), edges)
        share = h / fw.size * 100
        ax = cols[c][0]
        ax.bar(edges[:-1], share, width=0.1, align="edge", color=C_BAR, lw=0)
        ax.axvline(0.45, color=C_FILTER, lw=1.6, ls="--")
        ymax_f = max(ymax_f, share.max())
        win = (fw <= 0.45).mean() * 100
        kw = k[:, 4].astype(int)
        act = kw[kw > 0]
        bins = np.arange(0, 9)
        hk = np.bincount(np.clip(kw, 0, 8), minlength=9)[:9] / kw.size * 100
        ax2 = cols[c][1]
        ax2.bar(bins, hk, width=0.8, color=[GRID] + [C_BAR] * 8, lw=0)
        ymax_m = max(ymax_m, hk.max())
        stats.append((win, act.mean(), (kw == 0).mean() * 100))
        ax.text(0.98, 0.95, f"{win:.0f}% within 0.45 mm", transform=ax.transAxes, ha="right",
                va="top", fontsize=FS_NOTE, color=C_FILTER)
        ax2.text(0.98, 0.95, f"mean {act.mean():.2f}", transform=ax2.transAxes, ha="right",
                 va="top", fontsize=FS_NOTE, color=INK)
    for c in range(2):
        ax, ax2 = cols[c]
        ax.set_xlim(0, 2); ax.set_ylim(0, ymax_f * 1.15)
        ax.set_xticks([0, 0.45, 1, 1.5, 2]); ax.set_xticklabels(["0", "0.45", "1", "1.5", "≥2"])
        ax.set_xlabel("strip FWHM (mm)", fontsize=FS_LABEL, color=INK, labelpad=2)
        ax2.set_xlim(-0.6, 8.6); ax2.set_ylim(0, ymax_m * 1.15)
        ax2.set_xticks(range(0, 9)); ax2.set_xticklabels([str(i) for i in range(8)] + ["≥8"])
        ax2.set_xlabel("MPXI: strips per detector", fontsize=FS_LABEL, color=INK, labelpad=2)
        if c == 0:
            ax.set_ylabel("% of strips", fontsize=FS_LABEL, color=INK)
            ax2.set_ylabel("% of detectors", fontsize=FS_LABEL, color=INK)
        for a in (ax, ax2):
            style(a); a.yaxis.grid(True, color=GRID, lw=0.6); a.set_axisbelow(True)
    layers = {"benchmark": cols[0], "mobo": cols[1]}
    write_layers(fig, cols[0][0], [a for col in cols for a in col], layers, out, DPI)
    return stats


# Centres of the slide's "Benchmark" and "MOBO-optimized" labels within the figure box.
CX = (1.43, 4.40)


def show_map(ax, vec, vmin, vmax):
    im = ax.imshow(vec.reshape(N, N).T, origin="lower", extent=EXTENT, cmap=CMAP,
                   vmin=vmin, vmax=vmax, interpolation="nearest")
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_color(GRID)
    return im


def maps_figure(d, rows, out):
    """rows: list of (label, key in npz, unit text, scale). One row: large maps and a
    horizontal colour bar under them; two rows: smaller maps, a colour bar per row."""
    fig = plt.figure(figsize=(FIG_W, FIG_H))
    if len(rows) == 1:
        side, ys = 2.35, [FIG_H - 0.25 - 2.35]
    else:
        side = 1.45
        ys = [FIG_H - 0.22 - side, FIG_H - 0.22 - side - 0.24 - side]
    axes = [[fig.add_axes([(cx - side / 2) / FIG_W, y / FIG_H, side / FIG_W, side / FIG_H])
             for y in ys] for cx in CX]
    if len(rows) == 1:
        x0, x1 = CX[0] - side / 2, CX[1] + side / 2
        cbars = [fig.add_axes([x0 / FIG_W, (ys[0] - 0.42) / FIG_H, (x1 - x0) / FIG_W, 0.09 / FIG_H])]
    else:
        cbars = [fig.add_axes([(CX[1] + side / 2 + 0.08) / FIG_W, y / FIG_H, 0.08 / FIG_W, side / FIG_H])
                 for y in ys]
    means = []
    for r, (label, key, unit, scale, bar, fmt) in enumerate(rows):
        vals = [d[k][key] * scale for k in ("bench", "mobo")]
        vmin, vmax = 0, max(np.percentile(v, 99.5) for v in vals)
        for c, v in enumerate(vals):
            im = show_map(axes[c][r], v, vmin, vmax)
            axes[c][r].set_title(f"{label}, mean {np.nanmean(v):{fmt}}{unit}", fontsize=FS_NOTE,
                                 color=INK, pad=2)
            means.append(v.mean())
        horiz = len(rows) == 1
        cb = fig.colorbar(im, cax=cbars[r], orientation="horizontal" if horiz else "vertical")
        cb.ax.tick_params(labelsize=FS_TICK - 1, colors=INK_MUTED, length=2, pad=1)
        cb.outline.set_edgecolor(GRID)
        if horiz:
            cbars[r].set_xlabel(bar, fontsize=FS_LABEL - 0.5, color=INK, labelpad=2)

    layers = {"benchmark": axes[0], "mobo": axes[1] + cbars}
    write_layers(fig, axes[0][0], [a for col in axes for a in col] + cbars, layers, out, DPI)
    return means


def fwhm_mpxi_figure(d, out):
    """Slide 13 ("FWHMw Map and MPXI Histograms"): Dr. Yao's wFWHM_j map on top, strips
    per detector with the FWHM filter below; a shared colour bar and a shared y axis."""
    fig = plt.figure(figsize=(FIG_W, FIG_H))
    side = 1.55
    y_map = FIG_H - 0.2 - side
    maps = [fig.add_axes([(cx - side / 2) / FIG_W, y_map / FIG_H, side / FIG_W, side / FIG_H]) for cx in CX]
    cbar = fig.add_axes([(CX[1] + side / 2 + 0.08) / FIG_W, y_map / FIG_H, 0.08 / FIG_W, side / FIG_H])
    hw, hh, y_h = 2.1, 1.05, 0.42
    hists = [fig.add_axes([(cx - hw / 2 + 0.2) / FIG_W, y_h / FIG_H, hw / FIG_W, hh / FIG_H]) for cx in CX]
    vals = [d[k]["wfwhm_total"] for k in ("bench", "mobo")]
    vmax = max(np.nanpercentile(v, 99.5) for v in vals)
    vmin = min(np.nanpercentile(v, 0.5) for v in vals)
    stats = []
    for c, (key, v) in enumerate(zip(("bench", "mobo"), vals)):
        im = show_map(maps[c], np.nan_to_num(v, nan=vmin), vmin, vmax)
        maps[c].set_title(f"FWHM$_{{w}}$, mean {np.nanmean(v):.3f} mm", fontsize=FS_NOTE, color=INK, pad=2)
        kw = d[key]["dets"][:, 4].astype(int)
        act = kw[kw > 0]
        hk = np.bincount(np.clip(kw, 0, 8), minlength=9)[:9] / kw.size * 100
        hists[c].bar(np.arange(9), hk, width=0.8, color=[GRID] + [C_BAR] * 8, lw=0)
        hists[c].text(0.98, 0.95, f"mean {act.mean():.2f}", transform=hists[c].transAxes,
                      ha="right", va="top", fontsize=FS_NOTE, color=INK)
        stats.append((np.nanmean(v), act.mean(), (kw == 0).mean() * 100))
    cb = fig.colorbar(im, cax=cbar)
    cb.ax.tick_params(labelsize=FS_TICK - 1, colors=INK_MUTED, length=2, pad=1)
    cb.outline.set_edgecolor(GRID)

    ymax = max(h.get_ylim()[1] for h in hists)
    for c, h in enumerate(hists):
        h.set_xlim(-0.6, 8.6); h.set_ylim(0, ymax * 1.1)
        h.set_xticks(range(9)); h.set_xticklabels([str(i) for i in range(8)] + ["≥8"])
        # "MPXI" alone, as Dr. Yao had slide 10's axis titled (R9-rY)
        h.set_xlabel("MPXI", fontsize=FS_LABEL - 1, color=INK, labelpad=1)
        if c == 0:
            h.set_ylabel("% of detectors", fontsize=FS_LABEL - 1, color=INK)
        style(h); h.yaxis.grid(True, color=GRID, lw=0.6); h.set_axisbelow(True)
    layers = {"benchmark": [maps[0], hists[0]], "mobo": [maps[1], hists[1], cbar]}
    write_layers(fig, hists[0], maps + hists + [cbar], layers, out, DPI)
    return stats


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--bench", required=True)
    ap.add_argument("--mobo", required=True)
    ap.add_argument("--outdir", default=os.path.join(os.path.dirname(__file__), "..", "..", "deck_figures"))
    ap.add_argument("--ppds", choices=("eq6", "ry_mask"), default="ry_mask",
                    help="ry_mask: Dr. Yao's R10-rY slide 6 form, inside the strips; eq6: the "
                         "strategy document's, as the searches computed it (radial width corrected)")
    a = ap.parse_args()
    d = {"bench": load(a.bench), "mobo": load(a.mobo)}
    st = fwhm_mpxi_figure(d, os.path.join(a.outdir, "char_hist.png"))
    print("wFWHM map mean (mm), MPXI active mean, % detectors without a narrow strip:", st)
    m = maps_figure(d, [("ASCI", "asci_win", "%", 1.0, "ASCI (%)", ".1f")],
                    os.path.join(a.outdir, "char_asci.png"))
    print("asci means:", m)
    m = maps_figure(d, [("Sensitivity", "sens_all", "", 1.0, "", ".2f"),
                        (f"PPDS ({PPDS_BAR[a.ppds]})", f"{a.ppds}_all", "", PPDS_SCALE[a.ppds], "", ".2f")],
                    os.path.join(a.outdir, "char_sens_ppds.png"))
    print("sens, ppds means:", m)


if __name__ == "__main__":
    main()
