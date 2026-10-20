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

Usage:
  python make_tradeoff_figure.py --out ../../deck_figures/asci_cnr_tradeoff.png
"""
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import numpy as np

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

    ax_a.plot(x, asci, "-", color=C_ASCI, lw=2.4, zorder=3)
    ax_a.plot(x[~moved], asci[~moved], "o", color=C_ASCI, ms=7.5, zorder=4)
    ax_a.plot(x[moved], asci[moved], "o", color=C_ASCI, ms=7.5, mfc="white",
              mec=C_ASCI, mew=2.0, zorder=4)
    ax_a.set_ylabel("ASCI, strips\nunder 0.45 mm (%)", fontsize=FS_LABEL,
                    color=C_ASCI)
    ax_a.set_title("Move ring II outward and ASCI keeps rising",
                   fontsize=FS_TITLE, color=INK, pad=8, loc="left")

    ax_c.errorbar(x, cnr, yerr=sem, fmt="none", ecolor=C_CNR, elinewidth=2.0,
                  capsize=4, capthick=2.0, zorder=2)
    ax_c.plot(x, cnr, "-", color=C_CNR, lw=2.4, zorder=3)
    ax_c.plot(x[~moved], cnr[~moved], "s", color=C_CNR, ms=7.5, zorder=4)
    ax_c.plot(x[moved], cnr[moved], "s", color=C_CNR, ms=7.5, mfc="white",
              mec=C_CNR, mew=2.0, zorder=4)
    ax_c.set_ylabel("CNR", fontsize=FS_LABEL, color=C_CNR)
    ax_c.set_xlabel("Ring-II inner diameter (mm)", fontsize=FS_LABEL, color=INK)
    ax_c.set_title(f"but CNR peaks near {D2[k]:.0f} mm and is clearly lower "
                   f"beyond {turn:.0f} mm",
                   fontsize=FS_TITLE, color=INK, pad=8, loc="left")

    # The hollow points also moved ring III, so they change two things at
    # once. The note sits centred over those four, in the clear band between
    # the 4.5 and 4.6 grid lines, right of the 477 mm error bar.
    ax_c.text((x[moved][0] + x[moved][-1]) / 2, 4.55, "ring III also moved",
              fontsize=FS_NOTE, color=INK_MUTED, ha="center", va="center",
              zorder=5)

    # One vertical marker on both panels: the place the two disagree.
    for ax in (ax_a, ax_c):
        ax.axvline(turn, color=INK_MUTED, lw=1.2, ls="--", zorder=1)
        ax.set_facecolor("white")
        ax.yaxis.grid(True, color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(GRID)
        ax.tick_params(colors=INK_MUTED, labelsize=FS_TICK)
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
    fig.text(0.5, 0.07 / FIG_H,
             f"Each CNR point is a mean over {N_SEEDS} noise realizations; "
             f"bars are standard errors.",
             ha="center", va="bottom", fontsize=FS_NOTE, color=INK_MUTED,
             style="italic")
    fig.savefig(args.out, dpi=args.dpi, facecolor="white")
    print(f"wrote {args.out}")
    print(f"  CNR peaks at d2 = {D2[k]:.1f} mm, clearly lower beyond "
          f"{turn:.0f} mm; ASCI rises {ASCI[0]:.2f} to {ASCI[-1]:.2f}%")


if __name__ == "__main__":
    main()
