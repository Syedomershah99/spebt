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

Add --layers to write slide 10's two click-build layers beside --out instead of
the figure itself (mpxi_optimum_l1_points.png, mpxi_optimum_l2_trend.png):
transparent PNGs of the same frame that, laid over white in that order, give
the full figure, which the script checks before it exits. --out is not written.
"""
import argparse
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.offsetbox import DrawingArea
from matplotlib.text import Text
from matplotlib.ticker import MultipleLocator
from PIL import Image

# Validated with the dataviz palette validator (light surface, categorical):
# all six checks pass, CVD dE 24.7 protan.
C_POINTS, C_TREND = "#2a78d6", "#eb6834"
INK, INK_MUTED, GRID = "#0b0b0b", "#52514e", "#d8d7d2"

MPXI = "mpxi_windowed_active_mean"
CNR = "cnr_sector_mean"

# Slide 12 box, in inches. At this size the figure is placed at scale 1.
FIG_W, FIG_H = 8.45, 5.1
FS_TITLE, FS_LABEL, FS_TICK, FS_TEXT, FS_CALLOUT = 15, 13, 12, 12, 13


def write_layers(fig, ax, drawn, layers, out, dpi):
    """Write the click-build layers beside `out` (not `out` itself) and check
    that, laid over white in order, they give the full figure.

    Each layer is the full figure's frame with only its own artists shown, as
    straight-alpha RGBA. Its alpha is Agg's, drawn on transparency. Its colour
    is solved so that, laid over the layers before it, it gives what Agg draws
    on white with those layers shown: blending the overlapping translucent dots
    onto transparency rounds differently, by up to 4 levels. tight_layout has
    run once with everything shown and savefig does not rerun it; the x limits
    are frozen here too, so no layer can be laid out differently.
    """
    ax.set_xlim(ax.get_xlim())
    placed = [id(a) for arts in layers.values() for a in arts]
    if len(set(placed)) != len(placed):
        raise SystemExit("ERROR: an artist is in two layers")
    shown = [a.get_visible() for a in drawn]

    def render(arts, **kw):
        mine = {id(a) for a in arts}
        for a, vis in zip(drawn, shown):
            a.set_visible(vis and id(a) in mine)
        buf = BytesIO()
        fig.savefig(buf, format="png", dpi=dpi, **kw)
        return np.asarray(Image.open(buf), dtype=float)

    full = render(drawn, facecolor="white")[..., :3]
    under = np.full(full.shape, 255.0)
    stack, so_far = under.copy(), []
    stem = Path(out).with_suffix("")
    for k, (name, arts) in enumerate(layers.items(), 1):
        so_far += arts
        alpha = render(arts, transparent=True)[..., 3:] / 255
        over = render(so_far, facecolor="white")[..., :3]
        rgb = np.divide(over - (1 - alpha) * under, alpha,
                        out=np.zeros_like(over), where=alpha > 0)
        rgba = np.round(np.dstack([np.clip(rgb, 0, 255), alpha * 255]))
        path = f"{stem}_l{k}_{name}.png"
        Image.fromarray(rgba.astype(np.uint8)).save(path, dpi=(dpi, dpi))
        print(f"wrote {path}")
        rgba = np.asarray(Image.open(path), dtype=float)
        a = rgba[..., 3:] / 255
        stack = rgba[..., :3] * a + stack * (1 - a)
        under = over
    for a, vis in zip(drawn, shown):
        a.set_visible(vis)
    diff = np.abs(np.round(stack) - full)
    off = int((diff.max(axis=-1) > 8).sum())
    print(f"  layers over white vs the full figure: max {diff.max():.0f}, mean "
          f"{diff.mean():.4f} levels, {off} px off by more than 8")
    if off:
        raise SystemExit("ERROR: the layers do not stack to the full figure")


def main():
    ap = argparse.ArgumentParser(description="Build the MPXI-optimum figure")
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="mpxi_optimum.png")
    ap.add_argument("--bins", type=int, default=8)
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument("--layers", action="store_true",
                    help="write the click-build layers beside --out instead of it")
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
    dots = ax.scatter(df[MPXI], df[CNR], s=16, color=C_POINTS, alpha=0.25,
                      edgecolors="none", zorder=2, label=f"one dot = one design "
                      f"(n={len(df)})")
    trend = ax.errorbar(g["x"], g["y"], yerr=g["sd"], color=C_TREND, lw=2.4,
                        marker="o", ms=8, capsize=4, zorder=4,
                        markeredgecolor="white", markeredgewidth=1.4,
                        label="average CNR in each bin, ±1 SD")

    # One annotation carrying the claim, rather than a label on every bin.
    # It sits above every dot in the upper left (the highest there are at
    # MPXI 1.33 and 1.38, CNR 4.59 and 4.58) and may run past the top of the axes,
    # where there is no grid line, so no dot or line runs through the text.
    best = ax.annotate(f"best here,\nbetween {lo_x:.1f} and {hi_x:.1f}",
                       xy=(peak_x, peak_y), xytext=(peak_x - 0.52, 4.64),
                       fontsize=FS_CALLOUT, fontweight="bold", color=INK,
                       ha="center", va="bottom", annotation_clip=False,
                       arrowprops=dict(arrowstyle="->", color=INK_MUTED, lw=1.4,
                                       shrinkA=2, shrinkB=4))
    # Plain-language captions in a clear band under the data and under the
    # lowest grid line, in axes fractions so they cannot drift into the
    # error bars.
    note_l = ax.text(0.175, 0.025, "more strips per detector,\nmore signal",
                     transform=ax.transAxes, fontsize=FS_TEXT, color=INK_MUTED,
                     ha="center", va="bottom")
    note_r = ax.text(0.69, 0.025,
                     "too many strips, less certain\nwhere they came from",
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

    # Slide 10 builds the figure with its bullets (Dr. Yao, 22 Oct): the designs
    # from the slide's entry, then the binned trend and everything that reads it
    # with bullet 4. The legend is split by entry, each where it is in the full
    # figure.
    key_dots, key_trend = ([*box.get_children(), text] for box, text in
                           zip(leg.findobj(DrawingArea), leg.get_texts()))
    layers = {"points": [ax.xaxis, ax.yaxis, ax.spines["left"],
                         ax.spines["bottom"], dots, *key_dots],
              "trend": [*trend.get_children(), best, note_l, note_r, ax.title,
                        *key_trend]}

    fig.tight_layout()
    if args.layers:
        # Everything the axes draws, the legend by entry. Each layer shows only
        # its own, so an artist left out of both is missing from the stack and
        # fails the check. The empty left and right titles draw nothing and
        # stay as they are: hidden, they would lift the title off the figure.
        drawn = [a for a in (*ax.get_children(), *key_dots, *key_trend)
                 if a is not ax.patch and a is not leg
                 and not (isinstance(a, Text) and not a.get_text())]
        write_layers(fig, ax, drawn, layers, args.out, args.dpi)
    else:
        fig.savefig(args.out, dpi=args.dpi, facecolor="white")
        print(f"wrote {args.out}")
    print(f"  {len(df)} designs, {len(g)} bins, peak bin mean CNR "
          f"{peak_y:.2f} at MPXI {peak_x:.2f}; band {lo_x:.1f} to {hi_x:.1f}")
    print(f"  highest-MPXI bin is {peak_y - g['y'].iloc[-1]:.2f} below the peak")


if __name__ == "__main__":
    main()
