#!/usr/bin/env python3
"""
MIC figure: where CNR peaks against multiplexing.

The history this figure exists to settle. MPXI was originally MINIMIZED, on the
assumption that multiplexing costs image quality. Measured in physical units it
correlates +0.55 with CNR, so we were optimizing away from the thing we wanted.
RY predicted that from the splitting physics before it was measured, and the
sign was reversed in Aug 2026.

But "maximize" then overshot. A Spearman coefficient measures MONOTONE
association, and this relationship turns over: CNR climbs with MPXI to about
2.0 and falls away above it. A healthy positive rho was fully consistent with
the wrong instruction, and binning is what exposed the shape.

The figure shows both: every design as a point, and the binned mean as the
trend, so the turnover is visible rather than asserted. The bars are the
standard deviation of CNR within each band, not standard errors.

The figure is sized to its box on the slide (8.45 x 5.1 in) so it is shown at
scale 1 and the font sizes below are the sizes the audience sees.

Usage (from optimization/):
  python make_mpxi_optimum_figure.py \
      --data ../archive_backup/results_summary_mobo.csv \
      --out ../../deck_figures/mpxi_optimum.png
"""
import argparse

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

# Validated with the dataviz palette validator (light surface, categorical):
# all six checks pass, CVD dE 24.7 protan.
C_POINTS, C_TREND = "#2a78d6", "#eb6834"
INK, INK_MUTED, GRID = "#0b0b0b", "#52514e", "#d8d7d2"

MPXI = "mpxi_windowed_active_mean"
CNR = "cnr_sector_mean"

# Slide 12 box, in inches. At this size the figure is placed at scale 1.
FIG_W, FIG_H = 8.45, 5.1
FS_TITLE, FS_LABEL, FS_TICK, FS_TEXT, FS_CALLOUT = 15, 13, 12, 12, 13


def main():
    ap = argparse.ArgumentParser(description="Build the MPXI-optimum figure")
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="mpxi_optimum.png")
    ap.add_argument("--bins", type=int, default=8)
    ap.add_argument("--dpi", type=int, default=300)
    args = ap.parse_args()

    df = pd.read_csv(args.data).dropna(subset=[MPXI, CNR])
    # Quantile bins keep the counts comparable. Equal-width bins would put most
    # designs in the first one, since MPXI is right-skewed.
    df["bin"] = pd.qcut(df[MPXI], q=args.bins, duplicates="drop")
    # sd is the spread of CNR across the designs in a band (ddof=1), which is
    # what the bars show.
    g = df.groupby("bin", observed=True).agg(
        x=(MPXI, "mean"), y=(CNR, "mean"), sd=(CNR, "std"), n=(CNR, "size"))
    peak = g["y"].idxmax()
    peak_x, peak_y = g.loc[peak, "x"], g.loc[peak, "y"]
    # The top two bins are close, so the honest claim is a REGION rather than a
    # point. Quoting a single peak the trend does not clearly resolve would
    # overstate what 8 bins can say.
    top2 = g.nlargest(2, "y")["x"].sort_values()
    lo_x, hi_x = float(top2.iloc[0]), float(top2.iloc[-1])

    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    # Dr. Yao, 16 Oct: beginner-friendly. 285 raw points drown the one line
    # that carries the claim, so they sit in the background and the trend is
    # what the eye lands on. Alpha 0.25 rather than 0.13 so they still show
    # on a projector.
    ax.scatter(df[MPXI], df[CNR], s=16, color=C_POINTS, alpha=0.25,
               edgecolors="none", zorder=2, label=f"one dot = one design "
               f"(n={len(df)})")
    ax.errorbar(g["x"], g["y"], yerr=g["sd"], color=C_TREND, lw=2.4,
                marker="o", ms=8, capsize=4, zorder=4,
                markeredgecolor="white", markeredgewidth=1.4,
                label="average CNR in each bin, ±1 SD")

    # One annotation carrying the claim, rather than a label on every bin.
    # It sits above every dot in the upper left (the highest there are at
    # MPXI 1.33 and 1.38, CNR 4.59 and 4.58) and may run past the top of the axes,
    # where there is no grid line, so no dot or line runs through the text.
    ax.annotate(f"best here,\nbetween {lo_x:.1f} and {hi_x:.1f}",
                xy=(peak_x, peak_y), xytext=(peak_x - 0.52, 4.64),
                fontsize=FS_CALLOUT, fontweight="bold", color=INK, ha="center",
                va="bottom", annotation_clip=False,
                arrowprops=dict(arrowstyle="->", color=INK_MUTED, lw=1.4,
                                shrinkA=2, shrinkB=4))
    # Plain-language captions in a clear band under the data and under the
    # lowest grid line, in axes fractions so they cannot drift into the
    # error bars.
    ax.text(0.175, 0.025, "more strips per detector,\nmore signal",
            transform=ax.transAxes, fontsize=FS_TEXT, color=INK_MUTED,
            ha="center", va="bottom")
    ax.text(0.69, 0.025, "too many strips, less certain\nwhere they came from",
            transform=ax.transAxes, fontsize=FS_TEXT, color=INK_MUTED,
            ha="center", va="bottom")

    ax.set_xlabel("MPXI, strips under 0.45 mm: average per detector that sees one",
                  fontsize=FS_LABEL, color=INK)
    ax.set_ylabel("CNR", fontsize=FS_LABEL, color=INK)
    ax.set_title("More sharing helps, up to a point, then it hurts",
                 fontsize=FS_TITLE, color=INK, pad=30)

    ax.yaxis.grid(True, color=GRID, lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_MUTED, labelsize=FS_TICK)

    # Whole-number grid lines leave clear bands for the text: the captions
    # below 1, the legend and the annotation above 4. The top stops just
    # short of 5 so no grid line is drawn through the annotation.
    ax.yaxis.set_major_locator(MultipleLocator(1.0))
    if float(df[CNR].max()) >= 4.98:
        raise SystemExit("ERROR: a design is above the axis top; rethink ylim")
    ax.set_ylim(0.30, 4.98)

    # Upper right is empty of data above CNR 4 (no design past MPXI 2.3
    # reaches 3.8), so the legend sits there, under the axis top.
    leg = ax.legend(frameon=False, fontsize=FS_TEXT, loc="upper right",
                    borderaxespad=0.2)
    for t in leg.get_texts():
        t.set_color(INK)

    fig.tight_layout()
    fig.savefig(args.out, dpi=args.dpi, facecolor="white")
    print(f"wrote {args.out}")
    print(f"  {len(df)} designs, {len(g)} bins, peak bin mean CNR "
          f"{peak_y:.2f} at MPXI {peak_x:.2f}; band {lo_x:.1f} to {hi_x:.1f}")
    print(f"  highest-MPXI bin is {peak_y - g['y'].iloc[-1]:.2f} below the peak")


if __name__ == "__main__":
    main()
