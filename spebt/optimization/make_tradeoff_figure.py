#!/usr/bin/env python3
"""The ASCI vs CNR trade-off along ring-2 radius.

This is the figure that answers "why not just optimize CNR." Moving ring 2
outward raises angular sampling coverage monotonically and costs contrast, so a
single-objective run would walk down this curve without ever reporting what it
gave up.

Data: the 8-design ring2_* sweep (merged into the archive 2026-09-30). Crystal
count and radius rise together along it, because the packing limit scales with
radius. The companion fine sweep at fixed d2 (nd2fine_*) shows crystal count
alone produces no trend, which is what lets us attribute this axis to radius.

Every CNR point here is a 5-SEED MEAN with its own standard error, measured
2026-10-05/06. The single-draw version of this figure was misleading twice
over: it put the breakpoint at ~500 mm when the measured one is nearer 455,
and it showed a cliff between 1218 and 1268 that is not there. Three separate
times on this dataset a shape read off single draws failed to survive proper
measurement, so nothing goes on a slide at n=1.

Usage:
  python make_tradeoff_figure.py --out ../deck_figures/asci_cnr_tradeoff.png
"""
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# results/ring2_seed_out, merged into results_summary_mobo.csv on 2026-09-30.
# Single in-loop draws; the sweep's shape is what matters here, and the two
# endpoints were each confirmed at 5 seeds.
ND2   = [964, 1016, 1066, 1116, 1168, 1218, 1268, 1320]
D2    = [393.2, 414.1, 435.1, 456.1, 477.1, 498.0, 519.0, 540.0]
ASCI  = [56.59, 58.55, 58.76, 59.38, 61.08, 61.98, 63.41, 64.52]
# 5-seed means and standard errors (sd / sqrt(5)).
CNR   = [4.5092, 4.5930, 4.6463, 4.5908, 4.5174, 4.3718, 4.3183, 4.3272]
CNR_SD = [0.1239, 0.1124, 0.0583, 0.0952, 0.0635, 0.1410, 0.1043, 0.0184]

# Same validated palette as the other deck figures (CVD dE 24.7 protan).
C_ASCI, C_CNR = "#eb6834", "#2a78d6"

N_SEEDS = 5
INK, INK_MUTED, GRID = "#0b0b0b", "#52514e", "#d8d7d2"


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

    sem = [sd / np.sqrt(N_SEEDS) for sd in CNR_SD]
    # The argmax alone is over-precise: 414, 435 and 456 mm are all within
    # two standard errors of each other. The honest statement is where
    # contrast FIRST drops clearly below the peak.
    k = int(np.argmax(CNR))
    sem_all = [sd / np.sqrt(N_SEEDS) for sd in CNR_SD]
    below = [i for i in range(k + 1, len(CNR))
             if (CNR[k] - CNR[i]) / np.sqrt(sem_all[i] ** 2 + sem_all[k] ** 2) > 2]
    if not below:
        raise SystemExit("ERROR: contrast never drops clearly below the peak")
    turn = D2[below[0] - 1]

    fig, (ax_a, ax_c) = plt.subplots(
        2, 1, figsize=(9.2, 5.6), sharex=True,
        gridspec_kw=dict(hspace=0.42, height_ratios=[1, 1]))
    fig.patch.set_facecolor("white")

    ax_a.plot(D2, ASCI, "-o", color=C_ASCI, lw=2.4, ms=7, zorder=3)
    ax_a.set_ylabel("angular coverage\nASCI (%)", fontsize=12, color=C_ASCI)
    ax_a.set_title("Move ring 2 outward and angular coverage keeps rising",
                   fontsize=12.5, color=INK, pad=7, loc="left")

    ax_c.errorbar(D2, CNR, yerr=sem, fmt="-s", color=C_CNR, lw=2.4, ms=7,
                  capsize=4, zorder=3)
    ax_c.set_ylabel("contrast\nCNR", fontsize=12, color=C_CNR)
    ax_c.set_xlabel("ring-2 inner diameter (mm)", fontsize=12, color=INK)
    ax_c.set_title(f"but contrast is flat to about {turn:.0f} mm, then falls away",
                   fontsize=12.5, color=INK, pad=7, loc="left")

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
        ax.tick_params(colors=INK_MUTED, labelsize=10.5)
    # No cross-panel annotation: it ran through the lower title. The
    # dashed line and the panel titles carry it.

    fig.text(0.5, 0.012,
             f"Every contrast point is a mean over {N_SEEDS} noise "
             f"realizations; bars are standard errors.",
             ha="center", fontsize=10, color=INK_MUTED, style="italic")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(args.out, dpi=args.dpi, facecolor="white")
    print(f"wrote {args.out}")
    print(f"  contrast flat to d2 = {turn:.0f} mm; coverage rises "
          f"{ASCI[0]:.1f} to {ASCI[-1]:.1f}%")
    if k in (0, len(CNR) - 1):
        raise SystemExit("ERROR: contrast does not turn over inside the range; "
                         "the panel titles would be wrong.")


if __name__ == "__main__":
    main()
