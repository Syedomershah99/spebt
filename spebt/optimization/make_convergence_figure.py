#!/usr/bin/env python3
"""Where eleven searches ended up, per design parameter.

Replaces a table of standard deviations with the thing the table is trying to
say: every search lands on the same aperture, and nothing else is pinned.
Each parameter is drawn on its own searched range normalised to 0..1, so the
six are comparable on one axis.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import numpy as np
import analyze_search_convergence as asc
import mobo_agent as ma

UB_BLUE, ACCENT = "#005BBB", "#007681"
INK, MUTED, GRID = "#1a1a1a", "#5a5a5a", "#d8d7d2"

labels, X = asc.collect("results/results_summary_mobo.csv")
print(f"{len(labels)} searches")
if len(labels) != 11:
    raise SystemExit(f"ERROR: found {len(labels)} searches; the slide says eleven.")

# Roman numerals, as the deck names the rings everywhere else.
names = ["Aperture diameter", "Number of apertures", "Ring-I crystals",
         "Ring-II crystals", "Ring-II inner diameter", "Ring-III inner diameter"]
# Slide 18 box is 11.89 x 4.15 in and the picture is fitted by height, so
# this size is shown at scale 1 and the font sizes below are slide points.
fig, ax = plt.subplots(figsize=(11.5, 4.15))
fig.patch.set_facecolor("white"); ax.set_facecolor("white")
fig.subplots_adjust(left=0.2, right=0.855, top=0.735, bottom=0.115)
# SD column: x in figure fraction, y in data, so it sits clear of the axes.
col = mtransforms.blended_transform_factory(fig.transFigure, ax.transData)
X_COL = 0.918

rng = np.random.default_rng(0)
sds = []
for j in range(6):
    lo, hi = ma.BOUNDS_MIN[j], ma.BOUNDS_MAX[j]
    v = (X[:, j] - lo) / (hi - lo)
    y = 5 - j
    ax.plot([0, 1], [y, y], color=GRID, lw=1.3, zorder=1, solid_capstyle="butt")
    hot = (j == 0)
    ax.scatter(v, y + rng.uniform(-0.085, 0.085, len(v)),
               s=78 if hot else 52, zorder=3,
               color=UB_BLUE if hot else ACCENT,
               alpha=0.95 if hot else 0.62,
               edgecolor="white", linewidth=1.1)
    sd_pct = 100 * v.std(ddof=1)
    sds.append(sd_pct)
    ax.text(X_COL, y, f"{sd_pct:.1f}%", transform=col, ha="center",
            va="center", fontsize=13.5,
            color=UB_BLUE if hot else MUTED, weight="bold" if hot else "normal")

# The subtitle and the bullets on slide 18 quote these. Stop if they drift.
expected = [1.9, 11.8, 12.4, 14.2, 13.7, 27.0]
if [round(s, 1) for s in sds] != expected:
    raise SystemExit(f"ERROR: SDs read {[round(s, 1) for s in sds]}, "
                     f"expected {expected}.")
ratios = [s / sds[0] for s in sds[1:]]
if not (6 <= min(ratios) < 7 and 14 <= max(ratios) < 15):
    raise SystemExit(f"ERROR: other parameters scatter {min(ratios):.1f} to "
                     f"{max(ratios):.1f} times more widely, not six to fourteen.")
print("  SD, % of range: " + ", ".join(f"{s:.1f}" for s in sds))
print(f"  aperture end points {X[:, 0].min():.3f} to {X[:, 0].max():.3f} mm, "
      f"SD {X[:, 0].std(ddof=1):.3f} mm")
print(f"  ratios to aperture: {min(ratios):.1f} to {max(ratios):.1f}")

ax.set_yticks(range(6)); ax.set_yticklabels(names[::-1], fontsize=13, color=INK)
ax.set_xlim(-0.03, 1.03); ax.set_ylim(-0.5, 5.5)
ax.set_xticks([0, 0.5, 1.0])
ax.set_xticklabels(["range minimum", "midpoint", "range maximum"],
                   fontsize=12.5, color=MUTED)
ax.text(X_COL, 5.82, "SD, % of range", transform=col, ha="center",
        va="center", fontsize=12.5, color=MUTED, weight="bold")
for s in ("top", "right", "left"):
    ax.spines[s].set_visible(False)
ax.spines["bottom"].set_color(GRID)
ax.tick_params(axis="y", length=0, pad=8)
ax.tick_params(axis="x", colors=MUTED)

fig.text(0.012, 0.965, "Each dot is the best design of one search",
         fontsize=15, color=INK, weight="semibold", va="top")
fig.text(0.012, 0.885,
         f"Aperture diameter: SD {sds[0]:.1f}% of its range. The other "
         f"parameters scatter six to fourteen times more widely.",
         fontsize=12.5, color=MUTED, va="top")
fig.savefig("search_convergence.png", dpi=300, facecolor="white")
print("wrote search_convergence.png")
