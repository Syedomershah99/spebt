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
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="asci_cnr_tradeoff.png")
    ap.add_argument("--dpi", type=int, default=300)
    args = ap.parse_args()

    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    ax.plot(D2, ASCI, "-o", color=C_ASCI, lw=2.2, ms=7, zorder=3,
            label="ASCI @ 0.45 mm (angular coverage)")
    ax.set_xlabel("Ring-2 inner diameter (mm)", color=INK, fontsize=11)
    ax.set_ylabel("ASCI @ 0.45 mm  (%)", color=C_ASCI, fontsize=11)
    ax.tick_params(axis="y", colors=C_ASCI)
    ax.tick_params(axis="x", colors=INK_MUTED)
    ax.set_ylim(55, 66)

    ax2 = ax.twinx()
    sem = np.array(CNR_SD) / np.sqrt(N_SEEDS)
    ax2.errorbar(D2, CNR, yerr=sem, fmt="-s", color=C_CNR, lw=2.2, ms=7,
                 capsize=3, elinewidth=1.4, zorder=3,
                 label="CNR sector-mean (contrast), 5 seeds")
    ax2.set_ylabel("CNR sector-mean", color=C_CNR, fontsize=11)
    ax2.tick_params(axis="y", colors=C_CNR)
    ax2.set_ylim(4.20, 4.78)

    ax.grid(True, axis="y", color=GRID, lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    for sp in ("top",):
        ax.spines[sp].set_visible(False)
        ax2.spines[sp].set_visible(False)
    for a in (ax, ax2):
        for sp in ("left", "right", "bottom"):
            a.spines[sp].set_color(GRID)

    # Crystal count on a secondary x-axis: it rises with radius along this
    # sweep, and leaving it off invites the reading that count is the driver.
    axt = ax.secondary_xaxis("top")
    axt.set_xticks(D2[::2])
    axt.set_xticklabels([str(n) for n in ND2[::2]])
    axt.set_xlabel("Ring-2 crystal count (rises with radius here)",
                   color=INK_MUTED, fontsize=9, labelpad=3)
    axt.tick_params(colors=INK_MUTED, labelsize=9)
    axt.spines["top"].set_color(GRID)

    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    leg = ax.legend(h1 + h2, l1 + l2, loc="lower left", frameon=False,
                    fontsize=10)
    for t in leg.get_texts():
        t.set_color(INK)

    # Title and subtitle live at figure level with an explicit top margin, so
    # they cannot collide with the secondary axis label the way an axes-level
    # title did.
    fig.text(0.075, 0.955,
             "Angular coverage rises throughout; contrast holds, then falls "
             "past ~455 mm",
             color=INK, fontsize=13.5, ha="left", va="top", weight="semibold")
    fig.text(0.075, 0.900,
             "Every CNR point is a 5-seed mean; bars are standard errors. "
             "The last three sit clearly below the first four.",
             color=INK_MUTED, fontsize=9.5, ha="left", va="top")

    fig.tight_layout(rect=[0, 0, 1, 0.83])
    fig.savefig(args.out, dpi=args.dpi, facecolor="white")
    print(f"wrote {args.out}")

    print("\nsanity check against the source numbers:")
    from scipy.stats import spearmanr
    print("  Spearman(d2, ASCI) = %+.2f" % spearmanr(D2, ASCI).correlation)
    print("  Spearman(d2, CNR)  = %+.2f" % spearmanr(D2, CNR).correlation)
    print("  ASCI %.2f to %.2f  (+%.0f%%)" % (min(ASCI), max(ASCI),
                                              100*(max(ASCI)-min(ASCI))/min(ASCI)))
    print("  CNR  %.3f to %.3f  (-%.0f%%)" % (min(CNR), max(CNR),
                                              100*(max(CNR)-min(CNR))/max(CNR)))


if __name__ == "__main__":
    main()
