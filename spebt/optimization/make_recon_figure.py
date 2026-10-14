#!/usr/bin/env python3
"""Side-by-side reconstruction comparison for the MIC talk.

Dr. Yao's feedback was that reading CNR numbers off a slide does not show the
audience anything. This puts the reconstructions themselves on the slide:
phantom, reference geometry, optimized geometry, all from the SAME noise seed
so the only difference visible is the geometry.
"""
import glob
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

INK, MUTED = "#0b0b0b", "#52514e"
UB_BLUE = "#005BBB"

ph_raw = torch.load("../data/sai_10mm/hot_rods_phantom_10.0_mm_x_10.0_mm.pt",
                    weights_only=True)
if isinstance(ph_raw, dict):
    key = next(k for k, v in ph_raw.items()
               if hasattr(v, "shape") and tuple(v.shape)[-2:] == (200, 200))
    ph = np.asarray(ph_raw[key], dtype=float)
else:
    ph = np.asarray(ph_raw, dtype=float)
ph = np.squeeze(ph)

ref = np.load(glob.glob("results/tmi_reference_000/cnr_repeat_seed0/recon_mlem_T8.npz")[0])["final"]
our = np.load(glob.glob("results/mobo_0296*/cnr_repeat_seed0/recon_mlem_T8.npz")[0])["final"]

# Common scale across the two reconstructions so the comparison is honest:
# a per-panel scale would flatter whichever image is dimmer.
vmax = float(np.percentile(np.concatenate([ref.ravel(), our.ravel()]), 99.5))

panels = [
    (ph / ph.max() * vmax, "Phantom (ground truth)", None),
    (ref, "Published reference", "CNR  3.589 $\\pm$ 0.055"),
    (our, "MOBO optimized", "CNR  4.722 $\\pm$ 0.069"),
]

fig, axes = plt.subplots(1, 3, figsize=(13.0, 5.4))
fig.patch.set_facecolor("white")
for ax, (img, title, sub) in zip(axes, panels):
    ax.imshow(img, cmap="hot", vmin=0, vmax=vmax, interpolation="nearest")
    ax.set_title(title, fontsize=13, color=INK, pad=9, weight="semibold")
    if sub:
        ax.text(0.5, -0.07, sub, transform=ax.transAxes, ha="center",
                va="top", fontsize=12, color=UB_BLUE, weight="semibold")
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color("#d8d7d2")

# Scale bar on the first panel: 200 px over a 10 mm FOV = 0.05 mm/px.
ax = axes[0]
bar_px = 2.0 / 0.05
ax.plot([12, 12 + bar_px], [188, 188], color="white", lw=3, solid_capstyle="butt")
ax.text(12 + bar_px / 2, 183, "2 mm", color="white", fontsize=10,
        ha="center", va="bottom")

fig.text(0.5, 0.055,
         "Same noise realisation, same 150-iteration ML-EM, identical display "
         "scale. The only difference is the scanner geometry.",
         ha="center", fontsize=11.5, color=MUTED)
fig.subplots_adjust(left=0.02, right=0.98, top=0.90, bottom=0.24, wspace=0.06)
fig.savefig("recon_comparison.png", dpi=200, facecolor="white")
print("wrote recon_comparison.png")
print("display vmax =", round(vmax, 4))
