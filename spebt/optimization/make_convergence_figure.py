#!/usr/bin/env python3
"""Where eleven independent searches ended up, per design parameter.

Replaces a table of standard deviations with the thing the table is trying to
say: every search lands on the same aperture, and nothing else is pinned.
Each parameter is drawn on its own searched range normalised to 0..1, so the
six are comparable on one axis.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import analyze_search_convergence as asc
import mobo_agent as ma

UB_BLUE, ACCENT = "#005BBB", "#007681"
INK, MUTED, GRID = "#1a1a1a", "#5a5a5a", "#d8d7d2"

labels, X = asc.collect("results/results_summary_mobo.csv")
print(f"{len(labels)} searches")

names = ["Aperture\ndiameter", "Number of\napertures", "Ring 1\ncrystals",
         "Ring 2\ncrystals", "Ring 2 inner\ndiameter", "Ring 3 inner\ndiameter"]
fig, ax = plt.subplots(figsize=(10.4, 4.9))
fig.patch.set_facecolor("white"); ax.set_facecolor("white")

rng = np.random.default_rng(0)
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
    ax.text(1.03, y, f"{sd_pct:.1f}%", va="center", fontsize=12,
            color=UB_BLUE if hot else MUTED, weight="bold" if hot else "normal")

ax.set_yticks(range(6)); ax.set_yticklabels(names[::-1], fontsize=11.5, color=INK)
ax.set_xlim(-0.03, 1.12); ax.set_ylim(-0.6, 5.6)
ax.set_xticks([0, 0.5, 1.0])
ax.set_xticklabels(["range\nminimum", "midpoint", "range\nmaximum"],
                   fontsize=10.5, color=MUTED)
ax.text(1.03, 5.78, "spread", fontsize=10.5, color=MUTED, weight="bold")
for s in ("top", "right", "left"):
    ax.spines[s].set_visible(False)
ax.spines["bottom"].set_color(GRID)
ax.tick_params(axis="y", length=0)
ax.tick_params(axis="x", colors=MUTED)

fig.text(0.012, 0.965, "Each dot is where one independent search ended up",
         fontsize=13.5, color=INK, weight="semibold", va="top")
fig.text(0.012, 0.905,
         "Aperture diameter is recovered to 1.9% of its range. Every other "
         "parameter scatters six to fourteen times more widely.",
         fontsize=10.5, color=MUTED, va="top")
fig.subplots_adjust(left=0.135, right=0.9, top=0.80, bottom=0.14)
fig.savefig("search_convergence.png", dpi=200, facecolor="white")
print("wrote search_convergence.png")
