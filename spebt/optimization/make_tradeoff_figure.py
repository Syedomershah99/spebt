#!/usr/bin/env python3
"""The ASCI vs CNR trade-off along the ring-II diameter.

This is the figure that answers "why not just optimize CNR." Moving ring II
outward raises ASCI monotonically while CNR peaks and then falls, so a
single-objective run would walk down this curve without ever reporting what it
gave up.

Data: the 8-design ring2_* sweep (ring2_000..ring2_007, merged into the
archive 2026-09-30). Ring-II inner diameter runs 393.2 to 540.0 mm and its
crystal count 964 to 1320 at constant pair spacing, because the packing limit
scales with diameter. The companion fine sweep at fixed d2 (nd2fine_*) shows
crystal count alone produces no trend, which is what lets us attribute this
axis to the diameter.

Ring III does NOT stay put along the whole sweep. For the first four designs
it stays at 492.2 mm. For the last four (ring II at 477.1, 498.0, 519.0 and
540.0 mm) ring III moved out to 509.1, 530.0, 551.0 and 572.0 mm to keep the
10 mm clearance between rings (MIN_RING_GAP_MM in mobo_agent.py, so
d3 >= d2 + 32 mm; see make_ring2_seeds.py). Those four points change two
things at once, so both panels draw them as hollow markers and the CNR panel
carries the note "ring III also moved".

ASCI is the search's ASCI: beams narrower than 0.45 mm, 1 degree bins
(asci_pct_fwhm0p45 in results_summary_mobo.csv), 56.59 to 64.52%.

Every CNR point here is a 5-SEED MEAN with its own standard error, measured
2026-10-05/06. The single-draw version of this figure was misleading twice
over: it put the breakpoint at ~500 mm when the measured one is nearer 455,
and it showed a cliff between 1218 and 1268 that is not there. Three separate
times on this dataset a shape read off single draws failed to survive proper
measurement, so nothing goes on a slide at n=1.

The figure is sized to its box on the slide (8.45 x 5.05 in) so it is shown
at scale 1 and the font sizes below are the sizes the audience sees.

Click-build layers (Dr. Yao, 22 Oct: the figure builds with the words). With
--layers the script writes, next to --out and instead of the figure itself,
  <stem>_l1_frame.png  axes, ticks, tick labels, grid, x label, the dashed
                       marker, the "ring III also moved" note and the
                       footnote (on the slide from entry)
  <stem>_l2_asci.png   ASCI line and markers, its y label, the top title
                       (bullet 2)
  <stem>_l3_cnr.png    CNR line, markers and error bars, its y label, the
                       bottom title (bullet 3)
All three come from the same figure object as the full render. Each layer
paints only its own elements onto a transparent canvas while the rest are
still laid out, so the axes, limits, tick labels, aligned y labels and titles
sit exactly where they do in the full figure. Nothing is written unless the
three laid over white in order reproduce the full figure to within 2 levels,
and the still itself is never touched by --layers.

Usage:
  python make_tradeoff_figure.py --out ../../deck_figures/asci_cnr_tradeoff.png
  python make_tradeoff_figure.py --out ../../deck_figures/asci_cnr_tradeoff.png --layers
"""
import argparse
import contextlib
import io
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.axis import Axis, Tick
from matplotlib.text import Text
from matplotlib.ticker import MultipleLocator
import numpy as np
from PIL import Image

# results/ring2_seed_out, merged into results_summary_mobo.csv on 2026-09-30.
# ASCI is deterministic, one value per design. D3 is the ring-III inner
# diameter of each design, which is what marks the last four as hollow.
ND2   = [964, 1016, 1066, 1116, 1168, 1218, 1268, 1320]
D2    = [393.2, 414.1, 435.1, 456.1, 477.1, 498.0, 519.0, 540.0]
D3    = [492.2, 492.2, 492.2, 492.2, 509.1, 530.0, 551.0, 572.0]
ASCI  = [56.59, 58.55, 58.76, 59.38, 61.08, 61.98, 63.41, 64.52]
# 5-seed means and standard deviations; the bars are sd / sqrt(5).
CNR   = [4.5092, 4.5930, 4.6463, 4.5908, 4.5174, 4.3718, 4.3183, 4.3272]
CNR_SD = [0.1239, 0.1124, 0.0583, 0.0952, 0.0635, 0.1410, 0.1043, 0.0184]

# Same validated palette as the other deck figures (CVD dE 24.7 protan).
C_ASCI, C_CNR = "#eb6834", "#2a78d6"

N_SEEDS = 5
INK, INK_MUTED, GRID = "#0b0b0b", "#52514e", "#d8d7d2"

# Slide 11 box, in inches. At this size the figure is placed at scale 1.
FIG_W, FIG_H = 8.45, 5.05
FS_TITLE, FS_LABEL, FS_TICK, FS_NOTE = 14.5, 13, 12, 12

# Slide 11's clicks: the frame from entry, ASCI with bullet 2, CNR with 3.
LAYERS = ("frame", "asci", "cnr")
# The layers laid over white must give the full figure to within this many
# levels, with no pixel off by more than LAYER_OFF.
LAYER_MAX, LAYER_OFF = 2, 8


def _paint_nothing(*args, **kwargs):
    """Stands in for an artist's draw(): laid out as usual, paints nothing."""


@contextlib.contextmanager
def _only(layer, keep):
    """Let only the elements of layer `keep` paint.

    The others stay visible as far as matplotlib knows, so tick-label
    extents, the aligned y labels and the title positions are computed
    exactly as for the full figure; only their draw() is swapped out. Ticks
    (marks, labels and grid lines) are created at draw time, so they are
    muted through their class; they all belong to the frame.
    """
    muted = [a for name, arts in layer.items() if name != keep for a in arts]
    tick_draw = Tick.draw
    for a in muted:
        a.draw = _paint_nothing
    if keep != "frame":
        Tick.draw = _paint_nothing
    try:
        yield
    finally:
        Tick.draw = tick_draw
        for a in muted:
            del a.draw


def _check_partition(fig, layer):
    """Every element that paints is in exactly one layer.

    Ticks are the frame's by construction. The white figure and axes
    backgrounds are not in any layer: they are the white the layers are
    laid over.
    """
    owner = {}
    for name, arts in layer.items():
        for a in arts:
            if id(a) in owner:
                raise SystemExit(f"ERROR: {a!r} is in both the {owner[id(a)]} "
                                 f"and the {name} layer")
            owner[id(a)] = name
    elements = [a for a in fig.get_children()
                if a is not fig.patch and not isinstance(a, Axes)]
    for ax in fig.axes:
        for c in ax.get_children():
            if c is ax.patch:
                continue
            elements += [c.label, c.offsetText] if isinstance(c, Axis) else [c]
    loose = [a for a in elements
             if a.get_visible() and id(a) not in owner
             and not (isinstance(a, Text) and not a.get_text())]
    if loose:
        raise SystemExit(f"ERROR: drawn but in no layer: {loose}")


def write_layers(fig, layer, out, dpi):
    """Write the click-build layers of the full figure next to `out`."""
    _check_partition(fig, layer)

    def render(**kw):
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=dpi, **kw)
        return buf.getvalue()

    def pixels(png):
        return np.asarray(Image.open(io.BytesIO(png)))

    def geometry():
        return [(ax.get_position().bounds, ax.get_xlim(), ax.get_ylim())
                for ax in fig.axes]

    # The full figure exactly as the default run writes it.
    full = pixels(render(facecolor="white"))
    if os.path.exists(out) and not np.array_equal(
            np.asarray(Image.open(out).convert("RGBA")), full):
        raise SystemExit(f"ERROR: {out} is not the figure this script draws "
                         f"now; rerun without --layers first")
    # Fix the limits once (the values autoscale just gave), so no layer can
    # autoscale differently. Positions and size are fixed by subplots_adjust
    # and figsize, and no layout engine runs.
    for ax in fig.axes:
        ax.set_xlim(ax.get_xlim())
        ax.set_ylim(ax.get_ylim())
    fixed = geometry()
    pngs = []
    for name in LAYERS:
        with _only(layer, name):
            pngs.append(render(transparent=True))
        if geometry() != fixed:
            raise SystemExit(f"ERROR: rendering the {name} layer moved the axes")
    if not np.array_equal(pixels(render(facecolor="white")), full):
        raise SystemExit("ERROR: the full figure changed while the layers "
                         "were rendered")

    # Straight alpha, laid over white in click order.
    stack = np.full(full.shape[:2] + (3,), 255.0)
    for name, png in zip(LAYERS, pngs):
        px = pixels(png)
        if px.shape != full.shape:
            raise SystemExit(f"ERROR: the {name} layer is {px.shape}, "
                             f"the figure {full.shape}")
        a = px[..., 3:] / 255.0
        stack = px[..., :3] * a + stack * (1.0 - a)
    diff = np.abs(np.rint(stack) - full[..., :3])
    worst, off = int(diff.max()), int((diff.max(axis=-1) > LAYER_OFF).sum())
    print(f"  layers over white vs the full figure: max {worst}, "
          f"mean {diff.mean():.5f} levels, {off} pixels off by more than "
          f"{LAYER_OFF}")
    if worst > LAYER_MAX or off:
        raise SystemExit("ERROR: the layers do not reproduce the figure; "
                         "nothing written")
    stem = os.path.splitext(out)[0]
    for k, (name, png) in enumerate(zip(LAYERS, pngs), 1):
        path = f"{stem}_l{k}_{name}.png"
        with open(path, "wb") as f:
            f.write(png)
        print(f"wrote {path}")


def main():
    """Two stacked panels, one scale each.

    Dr. Yao, 16 Oct: the dual-axis version was not readable. Two y-axes on one
    frame force the reader to work out which curve belongs to which scale
    before they can see the point. Stacked panels sharing the x-axis say the
    same thing with no decoding: this rises all the way, that turns over.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="asci_cnr_tradeoff.png")
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument("--layers", action="store_true",
                    help="write the slide-11 click-build layers "
                         "<stem>_l1_frame.png, <stem>_l2_asci.png and "
                         "<stem>_l3_cnr.png next to --out instead of the "
                         "figure (see the module docstring)")
    args = ap.parse_args()

    x = np.asarray(D2)
    asci = np.asarray(ASCI)
    cnr = np.asarray(CNR)
    sem = np.asarray(CNR_SD) / np.sqrt(N_SEEDS)
    moved = np.asarray(D3) != D3[0]
    if list(np.flatnonzero(moved)) != [4, 5, 6, 7]:
        raise SystemExit("ERROR: expected ring III to move for the last four "
                         "designs only; the hollow markers and note would be wrong")

    # The argmax alone is over-precise: 414, 435 and 456 mm are all within
    # two standard errors of each other. The honest statement is where
    # CNR FIRST drops clearly below the peak.
    k = int(np.argmax(cnr))
    if k in (0, len(cnr) - 1):
        raise SystemExit("ERROR: CNR does not turn over inside the range; "
                         "the panel titles would be wrong.")
    below = [i for i in range(k + 1, len(cnr))
             if (cnr[k] - cnr[i]) / np.hypot(sem[i], sem[k]) > 2]
    if not below:
        raise SystemExit("ERROR: CNR never drops clearly below the peak")
    turn = D2[below[0] - 1]
    # The bottom title quotes both numbers, so pin them.
    assert D2[k] == 435.1, f"CNR argmax moved to {D2[k]} mm"
    assert f"{turn:.0f}" == "456", f"2-SE turn moved to {turn} mm"

    fig, (ax_a, ax_c) = plt.subplots(2, 1, figsize=(FIG_W, FIG_H), sharex=True)
    fig.patch.set_facecolor("white")
    # Each drawn element under the slide-11 click that shows it (--layers).
    # Ticks, tick labels and grid lines are the frame's, see _only().
    layer = {name: [] for name in LAYERS}

    layer["asci"] += ax_a.plot(x, asci, "-", color=C_ASCI, lw=2.4, zorder=3)
    layer["asci"] += ax_a.plot(x[~moved], asci[~moved], "o", color=C_ASCI,
                               ms=7.5, zorder=4)
    layer["asci"] += ax_a.plot(x[moved], asci[moved], "o", color=C_ASCI, ms=7.5,
                               mfc="white", mec=C_ASCI, mew=2.0, zorder=4)
    layer["asci"].append(ax_a.set_ylabel("ASCI, strips\nunder 0.45 mm (%)",
                                         fontsize=FS_LABEL, color=C_ASCI))
    layer["asci"].append(ax_a.set_title(
        "Move ring II outward and ASCI keeps rising",
        fontsize=FS_TITLE, color=INK, pad=8, loc="left"))

    layer["cnr"] += ax_c.errorbar(x, cnr, yerr=sem, fmt="none", ecolor=C_CNR,
                                  elinewidth=2.0, capsize=4, capthick=2.0,
                                  zorder=2).get_children()
    layer["cnr"] += ax_c.plot(x, cnr, "-", color=C_CNR, lw=2.4, zorder=3)
    layer["cnr"] += ax_c.plot(x[~moved], cnr[~moved], "s", color=C_CNR, ms=7.5,
                              zorder=4)
    layer["cnr"] += ax_c.plot(x[moved], cnr[moved], "s", color=C_CNR, ms=7.5,
                              mfc="white", mec=C_CNR, mew=2.0, zorder=4)
    layer["cnr"].append(ax_c.set_ylabel("CNR", fontsize=FS_LABEL, color=C_CNR))
    layer["frame"].append(ax_c.set_xlabel("Ring-II inner diameter (mm)",
                                          fontsize=FS_LABEL, color=INK))
    layer["cnr"].append(ax_c.set_title(
        f"but CNR peaks near {D2[k]:.0f} mm and is clearly lower "
        f"beyond {turn:.0f} mm",
        fontsize=FS_TITLE, color=INK, pad=8, loc="left"))

    # The hollow points also moved ring III, so they change two things at
    # once. The note sits centred over those four, in the clear band between
    # the 4.5 and 4.6 grid lines, right of the 477 mm error bar.
    layer["frame"].append(ax_c.text(
        (x[moved][0] + x[moved][-1]) / 2, 4.55, "ring III also moved",
        fontsize=FS_NOTE, color=INK_MUTED, ha="center", va="center", zorder=5))

    # One vertical marker on both panels: the place the two disagree.
    for ax in (ax_a, ax_c):
        layer["frame"].append(ax.axvline(turn, color=INK_MUTED, lw=1.2, ls="--",
                                         zorder=1))
        ax.set_facecolor("white")
        ax.yaxis.grid(True, color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(GRID)
        ax.tick_params(colors=INK_MUTED, labelsize=FS_TICK)
        layer["frame"] += [*ax.spines.values(), ax.xaxis.offsetText,
                           ax.yaxis.offsetText]
    # Fixed steps: at this height the default picks 57.5, 62.5 for ASCI and
    # only two ticks for CNR.
    ax_a.yaxis.set_major_locator(MultipleLocator(2))
    ax_c.yaxis.set_major_locator(MultipleLocator(0.1))
    # No cross-panel annotation: it ran through the lower title. The
    # dashed line and the panel titles carry it.

    # Margins are set by hand, in inches, so the caption sits in its own band
    # below the x label instead of touching it.
    fig.subplots_adjust(left=1.12 / FIG_W, right=1 - 0.22 / FIG_W,
                        top=1 - 0.42 / FIG_H, bottom=0.86 / FIG_H, hspace=0.30)
    fig.align_ylabels([ax_a, ax_c])
    layer["frame"].append(fig.text(
        0.5, 0.07 / FIG_H,
        f"Each CNR point is a mean over {N_SEEDS} noise realizations; "
        f"bars are standard errors.",
        ha="center", va="bottom", fontsize=FS_NOTE, color=INK_MUTED,
        style="italic"))
    if args.layers:
        write_layers(fig, layer, args.out, args.dpi)
        return
    fig.savefig(args.out, dpi=args.dpi, facecolor="white")
    print(f"wrote {args.out}")
    print(f"  CNR peaks at d2 = {D2[k]:.1f} mm, clearly lower beyond "
          f"{turn:.0f} mm; ASCI rises {ASCI[0]:.2f} to {ASCI[-1]:.2f}%")


if __name__ == "__main__":
    main()
