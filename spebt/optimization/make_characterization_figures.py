#!/usr/bin/env python3
"""Slides 13-15: the benchmark beside the MOBO-optimized design (Dr. Yao, R10-rY, 29 Oct 2026).

Reads the two npz files of compute_design_maps.py and writes, for each slide, the two
click-build layers the builder places in the 5.82 x 3.55 in box to the right of the
table (so the figure is shown at scale 1 and these font sizes are the sizes seen):
the benchmark's column with the shared colour bars (QA of R11: the first map shown needs
its scale), then the MOBO-optimized column.
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
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from make_mpxi_optimum_figure import write_layers, INK, INK_MUTED, GRID

FIG_W, FIG_H, DPI = 5.82, 3.55, 300         # hist_figure (the strip-FWHM histograms, kept on offer)
FS_HEAD, FS_LABEL, FS_TICK, FS_NOTE = 11.5, 11, 10, 10
C_BAR, C_FILTER = "#2a78d6", "#eb6834"
CMAP = "viridis"
# The maps sum the PPDFs of T8's 8 bed positions; a T8 scan shares its time among them, so
# the system's sensitivity is their mean (Dr. Yao, 30 Oct 2026: "check the maximum value of
# the sensitivity color palette"). Both maps are shown per bed position.
PER_POSITION = 1 / 8
# Dr. Yao, 31 Oct: "report PPDS_j the unit of per pixel ... scaled by 0.05^2" (A_i in pixels).
PIXEL_MM = 0.05
PPDS_SCALE = {"ry_mask": PER_POSITION * PIXEL_MM ** 2, "eq6": 1e2 * PER_POSITION}
PPDS_UNIT = {"ry_mask": "", "eq6": " ×10⁻² mm⁻²"}   # per pixel: on the colour bar, as sensitivity
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


# Slides 13-15 since R12 (Dr. Yao's R11-rY, 30 Oct 2026: the tables gone, "the images and
# graphs may be enlarged to use the space on the slide", the colour bars vertical): each
# figure fills up to AREA_W x AREA_H in to the right of the slide's formula column, the
# arrays drawn as stored (x on the first index, down the page) as the phantom on slides 6,
# 12 and 20. Column centres and slide 13's divider go to char_layout.json for the builder.
AREA_W, AREA_H = 7.86, 4.86
FS_T, FS_TK, FS_L = 12.5, 11, 12             # titles, ticks, axis and bar labels
LAYOUT = {}


def show_map(ax, vec, vmin, vmax):
    im = ax.imshow(vec.reshape(N, N), cmap=CMAP, vmin=vmin, vmax=vmax, interpolation="nearest")
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_color(GRID)
    return im


def add_axes_in(fig, x, y, w, h, fw, fh):
    return fig.add_axes([x / fw, y / fh, w / fw, h / fh])


def vbar(fig, im, x, y, h, fw, fh, label=None):
    cax = add_axes_in(fig, x, y, 0.1, h, fw, fh)
    cb = fig.colorbar(im, cax=cax)
    cb.ax.tick_params(labelsize=FS_TK - 1, colors=INK_MUTED, length=2, pad=2)
    cb.outline.set_edgecolor(GRID)
    if label:
        cb.set_label(label, fontsize=FS_L - 1, color=INK, labelpad=4)
    return cax


def maps_figure(d, rows, out, side, name, gap=0.22, left=0.08):
    """rows: (title, npz key, unit, scale, bar label, number format). Benchmark left,
    MOBO-optimized right, one vertical colour bar per row at the right; layer 1 holds the
    benchmark's maps and the bars (the first map shown needs its scale), layer 2 the rest."""
    top, title_h, row_gap = 0.05, 0.32, 0.3
    cx = (left + side / 2, left + side * 1.5 + gap)
    fw = cx[1] + side / 2 + 1.05
    fh = top + len(rows) * (title_h + side) + (len(rows) - 1) * row_gap + 0.05
    fig = plt.figure(figsize=(fw, fh))
    bench, mobo, bars, means, ys = [], [], [], [], []
    for r, (label, key, unit, scale, bar, fmt) in enumerate(rows):
        y = fh - top - title_h - side - r * (title_h + side + row_gap)
        ys.append(y)
        vals = [d[k][key] * scale for k in ("bench", "mobo")]
        vmax = max(np.percentile(v, 99.5) for v in vals)
        for c, v in enumerate(vals):
            ax = add_axes_in(fig, cx[c] - side / 2, y, side, side, fw, fh)
            im = show_map(ax, v, 0, vmax)
            ax.set_title(f"{label}, mean {np.nanmean(v):{fmt}}{unit}", fontsize=FS_T, color=INK, pad=4)
            (bench if c == 0 else mobo).append(ax)
            means.append(float(np.nanmean(v)))
        bars.append(vbar(fig, im, cx[1] + side / 2 + 0.14, y, side, fw, fh, bar))
    LAYOUT[name] = {"w": round(fw, 4), "h": round(fh, 4), "cx": [round(v, 4) for v in cx]}
    if len(rows) == 2:  # mid-gap between the rows, from the bottom, for his slide 15 divider
        LAYOUT[name]["divider"] = round((ys[0] + ys[1] + side + title_h) / 2, 4)
    write_layers(fig, bench[0], bench + mobo + bars, {"benchmark": bench + bars, "mobo": mobo}, out, DPI)
    return means


def fwhm_mpxi_figure(d, out):
    """Slide 13 ("wFWHM Map and MPXI Histograms"): his wFWHM_j map above, strips per
    detector with the FWHM filter below, a line on the slide between the two. The
    histograms cover the detectors with one or more such strips, the set MPXI's mean
    is taken over (R13: a grey bar for the rest confused Dr. Yao)."""
    fw, fh = AREA_W, AREA_H
    side, hw, hh, y_h = 1.9, 3.0, 1.62, 0.55
    cx = (2.1, 5.5)
    y_map = fh - 0.05 - 0.32 - side
    vals = [d[k]["wfwhm_total"] for k in ("bench", "mobo")]
    # A fixed scale (QA of R11): the maps span only 0.477-0.500 mm, and a scale fitted to
    # them made a 0.02 mm spread look like the whole colour range.
    vmin, vmax = 0.40, 0.60
    fig = plt.figure(figsize=(fw, fh))
    maps = [add_axes_in(fig, c - side / 2, y_map, side, side, fw, fh) for c in cx]
    hists = [add_axes_in(fig, c - hw / 2, y_h, hw, hh, fw, fh) for c in cx]
    stats = []
    for c, (key, v) in enumerate(zip(("bench", "mobo"), vals)):
        im = show_map(maps[c], np.nan_to_num(v, nan=vmin), vmin, vmax)
        maps[c].set_title(f"wFWHM, mean {np.nanmean(v):.3f} mm", fontsize=FS_T, color=INK, pad=4)
        kw = d[key]["dets"][:, 4].astype(int)
        act = kw[kw > 0]
        # 5 and up in one bar (his 2 Nov meeting): alone, 6 to 8 were too small to see
        hk = np.bincount(np.clip(act, 0, 5), minlength=6)[1:6] / act.size * 100
        hists[c].bar(np.arange(1, 6), hk, width=0.8, color=C_BAR, lw=0)
        hists[c].text(0.98, 0.95, f"mean {act.mean():.2f}", transform=hists[c].transAxes,
                      ha="right", va="top", fontsize=FS_T, color=INK)
        stats.append((np.nanmean(v), act.mean(), (kw == 0).mean() * 100))
    cbar = vbar(fig, im, cx[1] + side / 2 + 0.14, y_map, side, fw, fh, "wFWHM (mm)")
    ymax = max(h.get_ylim()[1] for h in hists)
    for c, h in enumerate(hists):
        h.set_xlim(0.4, 5.6); h.set_ylim(0, ymax * 1.1)
        h.set_xticks(range(1, 6)); h.set_xticklabels([str(i) for i in range(1, 5)] + ["≥5"])
        h.set_xlabel("MPXI", fontsize=FS_L, color=INK, labelpad=2)
        if c == 0:
            h.set_ylabel("% of detectors", fontsize=FS_L, color=INK)
        style(h); h.tick_params(labelsize=FS_TK)
        h.yaxis.grid(True, color=GRID, lw=0.6); h.set_axisbelow(True)
    LAYOUT["hist"] = {"w": fw, "h": fh, "cx": list(cx), "divider": round((y_map + y_h + hh) / 2, 4)}
    write_layers(fig, hists[0], maps + hists + [cbar],
                 {"benchmark": [maps[0], hists[0], cbar], "mobo": [maps[1], hists[1]]}, out, DPI)
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
                    os.path.join(a.outdir, "char_asci.png"), 3.25, "asci")
    print("asci means:", m)
    m = maps_figure(d, [("Sensitivity", "sens_all", "", PER_POSITION, "probability (2-D)", ".3f"),
                        ("PPDS", f"{a.ppds}_all", PPDS_UNIT[a.ppds], PPDS_SCALE[a.ppds], "per pixel (2-D)", ".3f")],
                    os.path.join(a.outdir, "char_sens_ppds.png"), 1.91, "sens_ppds", gap=0.45, left=0.3)
    print("sens, ppds means (per bed position):", m)
    with open(os.path.join(a.outdir, "char_layout.json"), "w") as f:
        json.dump(LAYOUT, f, indent=1)
    print("layout:", LAYOUT)


if __name__ == "__main__":
    main()
