#!/usr/bin/env python3
"""
The MIC headline figure: our optimized design against the published TMI reference.

Both sides were measured identically: 5 independent Poisson realisations of the
same 150-iteration ML-EM reconstruction, sector-mean CNR on the hot-rod phantom.
The reference configuration (0.4 mm apertures, 180 apertures, 480/720 detectors,
d2 390 mm, d3 520 mm) had never been run through our pipeline before Aug 2026,
which is why this comparison did not exist until now.

Values are recorded here rather than recomputed because they come from a
5-seed repeat study whose raw outputs live on the cluster. Provenance for each
is in the comment beside it; regenerate with analyze_cnr_repeats.py.

Usage:
  python make_reference_figure.py --out reference_comparison.png
"""
import argparse

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Rod radii of the six phantom sections, smallest to largest (mm).
ROD_RADII = [0.100, 0.125, 0.150, 0.175, 0.200, 0.225]

# analyze_cnr_repeats.py, tmi_reference_000, n=5 seeds
REF_MEAN = [2.057, 2.247, 2.592, 4.309, 4.769, 5.559]
REF_STD  = [0.114, 0.079, 0.108, 0.181, 0.130, 0.161]
REF_OVERALL = (3.589, 0.055)

# analyze_cnr_repeats.py, mobo_0296, n=5 seeds
OURS_MEAN = [2.498, 3.615, 4.028, 5.717, 6.362, 6.113]
OURS_STD  = [0.248, 0.267, 0.239, 0.134, 0.104, 0.097]
OURS_OVERALL = (4.722, 0.069)

# Validated with the dataviz palette validator (light surface, categorical):
# CVD separation dE 24.7 protan / 32.7 tritan, normal-vision 33.6, all checks pass.
C_REF, C_OURS = "#eb6834", "#2a78d6"
INK, INK_MUTED, GRID = "#0b0b0b", "#52514e", "#d8d7d2"


def main():
    ap = argparse.ArgumentParser(description="Build the reference-comparison figure")
    ap.add_argument("--out", default="reference_comparison.png")
    ap.add_argument("--dpi", type=int, default=300)
    args = ap.parse_args()

    gain = [100 * (o / r - 1) for o, r in zip(OURS_MEAN, REF_MEAN)]
    overall_gain = 100 * (OURS_OVERALL[0] / REF_OVERALL[0] - 1)
    # The subtitle names the two rod radii with the largest gain. Check it
    # against the numbers instead of trusting the text.
    top2 = sorted(np.argsort(gain)[-2:].tolist())
    if [ROD_RADII[i] for i in top2] != [0.125, 0.150]:
        raise SystemExit(f"ERROR: largest gains are at {[ROD_RADII[i] for i in top2]} "
                         f"mm, not 0.125 and 0.150 mm; the subtitle would be wrong.")

    # Slide 16 box is 11.89 x 4.9 in and the picture is fitted by height, so
    # this size is shown at scale 1 and the font sizes below are slide points.
    fig, ax = plt.subplots(figsize=(11.0, 4.9))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    x = np.arange(len(ROD_RADII))
    # A 2px-equivalent gap between adjacent bars: width 0.38 inside a 0.5 slot.
    w = 0.38
    ax.bar(x - w / 2, REF_MEAN, w, yerr=REF_STD, capsize=3.5,
           color=C_REF, label="Published benchmark", zorder=3,
           error_kw=dict(ecolor=INK_MUTED, lw=1.3, capthick=1.3))
    ax.bar(x + w / 2, OURS_MEAN, w, yerr=OURS_STD, capsize=3.5,
           color=C_OURS, label="MOBO-optimized", zorder=3,
           error_kw=dict(ecolor=INK_MUTED, lw=1.3, capthick=1.3))

    # Direct labels carry the story; a number on every bar would not.
    for xi, (o, s, g) in enumerate(zip(OURS_MEAN, OURS_STD, gain)):
        ax.text(xi, o + s + 0.22, f"+{g:.0f}%", ha="center", va="bottom",
                fontsize=13, fontweight="bold", color=INK, zorder=4)

    ax.set_xticks(x)
    # Diameters, as slide 6 introduces the rods (Dr. Yao, R11-rY, 30 Oct 2026).
    ax.set_xticklabels([f"{2 * r:.2f}" for r in ROD_RADII])
    ax.set_xlabel("Hot-rod diameter (mm)", fontsize=13, color=INK, labelpad=6)
    # Each bar is ONE rod section, not the sector mean. The sector mean appears
    # only in the title, where it summarises all six.
    ax.set_ylabel("CNR (mean of 5 noise realizations)", fontsize=13, color=INK,
                  labelpad=8)
    # Dr. Yao, 17 Oct: scientific phrasing rather than "beats". CNR, not
    # "contrast": the bars are contrast-to-noise ratios.
    # One decimal, not zero. The slide that carries this figure also
    # states +31.6% beside it, and "32%" next to "+31.6%" on one slide
    # reads as two different numbers.
    # Dr. Yao, 22 Oct: "benchmark" for the published design, "MOBO-optimized"
    # for ours, as on the slides.
    line1 = (f"MOBO-optimized design: {overall_gain:.1f}% higher CNR than the "
             f"published benchmark")
    line2 = (f"sector mean {OURS_OVERALL[0]:.2f} ± {OURS_OVERALL[1]:.2f} against "
             f"{REF_OVERALL[0]:.2f} ± {REF_OVERALL[1]:.2f} (mean ± SD), largest "
             f"gains at rod diameters 0.25 to 0.30 mm")

    # Recessive grid and axes; no top/right spines.
    ax.yaxis.grid(True, color=GRID, lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_MUTED, labelsize=12.5)
    ax.set_xlim(-0.6, len(ROD_RADII) - 0.4)
    ax.set_ylim(0, max(OURS_MEAN) + 1.15)

    # White box behind the legend so the y = 7 gridline does not run
    # through the entries.
    leg = ax.legend(frameon=True, facecolor="white", edgecolor="none",
                    framealpha=1.0, fontsize=13, loc="upper left",
                    bbox_to_anchor=(0.0, 1.02), borderaxespad=0.3)
    leg.set_zorder(5)
    for t in leg.get_texts():
        t.set_color(INK)

    fig.subplots_adjust(left=0.072, right=0.992, top=0.835, bottom=0.135)
    xc = (fig.subplotpars.left + fig.subplotpars.right) / 2
    fig.text(xc, 0.975, line1, ha="center", va="top", fontsize=15.5, color=INK)
    fig.text(xc, 0.912, line2, ha="center", va="top", fontsize=13, color=INK_MUTED)
    fig.savefig(args.out, dpi=args.dpi, facecolor="white")
    print(f"wrote {args.out}")
    print(f"  overall gain {overall_gain:.1f}%")
    print("  per-rod gain " + ", ".join(f"{g:.0f}%" for g in gain))


if __name__ == "__main__":
    main()
