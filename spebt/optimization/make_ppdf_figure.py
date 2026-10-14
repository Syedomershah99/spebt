#!/usr/bin/env python3
"""What a PPDF actually looks like, from a real system matrix.

Slide 6 asserts that every design metric derives from "the response of one
detector element to every point in the field of view". That is an abstraction
until you see one. This draws a real beam out of a real PPDF file and marks
the width the FWHM metric measures, so the four metrics have something
concrete behind them.
"""
import glob
import h5py
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

UB_BLUE, ACCENT, CORAL = "#005BBB", "#007681", "#E56D54"
INK, MUTED, GRID = "#1a1a1a", "#5a5a5a", "#d8d7d2"
NPIX, MM_PER_PX = 200, 0.05

f = sorted(glob.glob("results/mobo_0296*/position_000_ppdfs_t8_00.hdf5"))
if not f:
    f = sorted(glob.glob("results/*/position_000_ppdfs_t8_00.hdf5"))
if not f:
    # PPDF files live on the cluster, not in the laptop checkout; run this there.
    raise SystemExit("ERROR: no position_000_ppdfs_t8_00.hdf5 under results/. "
                     "This script needs the raw PPDFs, which only exist on CCR.")
src = f[0]
with h5py.File(src, "r") as h:
    ppdfs = h["ppdfs"][:]
print(f"{src.split('/')[1][:40]}  shape {ppdfs.shape}")

tot = ppdfs.sum(axis=1)
order = np.argsort(tot)[::-1]
strong = order[0]
beam = ppdfs[strong].reshape(NPIX, NPIX)

fig = plt.figure(figsize=(11.2, 4.7))
fig.patch.set_facecolor("white")
gs = fig.add_gridspec(1, 3, width_ratios=[1.05, 1.05, 1.25], wspace=0.3)

# (a) one detector's beam
ax = fig.add_subplot(gs[0]); ax.set_facecolor("white")
ax.imshow(beam, cmap="magma", interpolation="nearest")
ax.set_title("(a)  One detector element", fontsize=12, color=INK, pad=7)
ax.set_xticks([]); ax.set_yticks([])
ax.text(0.5, -0.07, "its response across the whole field of view",
        transform=ax.transAxes, ha="center", va="top", fontsize=10, color=MUTED)

# (b) several beams, to show angular sampling
ax = fig.add_subplot(gs[1]); ax.set_facecolor("white")
pick = order[::max(len(order) // 40, 1)][:28]
stack = ppdfs[pick].reshape(len(pick), NPIX, NPIX).sum(axis=0)
ax.imshow(stack, cmap="magma", interpolation="nearest")
ax.set_title("(b)  Many detectors", fontsize=12, color=INK, pad=7)
ax.set_xticks([]); ax.set_yticks([])
ax.text(0.5, -0.07, "beams crossing at many angles",
        transform=ax.transAxes, ha="center", va="top", fontsize=10, color=MUTED)

# (c) the profile across the beam, with the width FWHM measures
ax = fig.add_subplot(gs[2]); ax.set_facecolor("white")
row = int(np.argmax(beam.sum(axis=1)))
prof = beam[row]
x = np.arange(NPIX) * MM_PER_PX
pk = prof.max()
peak = int(np.argmax(prof))

# One detector sees SEVERAL beams, through different apertures, so a cut
# across the FOV crosses several peaks. Taking first-to-last above half
# maximum spans all of them and reported 8.40 mm, against a real beam width
# near 0.5 mm. Walk outward from the single tallest peak instead.
i = peak
while i > 0 and prof[i - 1] >= pk / 2:
    i -= 1
j = peak
while j < len(prof) - 1 and prof[j + 1] >= pk / 2:
    j += 1
lo, hi = x[i], x[j]
fwhm = hi - lo
if not (0.05 < fwhm < 2.0):
    raise SystemExit(f"ERROR: FWHM came out {fwhm:.2f} mm; beam widths in this "
                     f"system are near 0.5 mm. The profile cut is wrong.")

# Zoom on that beam so the width is legible rather than a spike on a wide axis.
pad = 14
a0, a1 = max(peak - pad, 0), min(peak + pad, NPIX - 1)
ax.plot(x[a0:a1], prof[a0:a1], color=UB_BLUE, lw=2.2)
ax.axhline(pk / 2, color=CORAL, lw=1.3, ls="--")
ax.annotate("", xy=(lo, pk / 2), xytext=(hi, pk / 2),
            arrowprops=dict(arrowstyle="<->", color=CORAL, lw=1.7))
ax.text((lo + hi) / 2, pk * 0.63, f"FWHM {fwhm:.2f} mm",
        ha="center", fontsize=11.5, color=CORAL, weight="bold")
ax.set_xlim(x[a0], x[a1])
print(f"single-beam FWHM {fwhm:.3f} mm")
ax.set_title("(c)  Width of one beam", fontsize=12, color=INK, pad=7)
ax.set_xlabel("position across the field of view (mm)", fontsize=10, color=MUTED)
# This is one beam from one detector, and beams vary. The FWHM metric is the
# mean over all of them weighted by sensitivity, which is nearer 0.48 mm.
# Without saying so, a single sharp beam reads as the system resolution.
ax.text(0.5, -0.30, "one example beam; the metric averages this over all\n"
        "beams, weighted by the signal each one carries",
        transform=ax.transAxes, ha="center", va="top", fontsize=9.5, color=MUTED)
ax.set_ylabel("detection probability", fontsize=10, color=MUTED)
ax.tick_params(colors=MUTED, labelsize=9)
for sp in ("top", "right"):
    ax.spines[sp].set_visible(False)
for sp in ("left", "bottom"):
    ax.spines[sp].set_color(GRID)

fig.subplots_adjust(left=0.035, right=0.985, top=0.86, bottom=0.28)
fig.savefig("ppdf_beam.png", dpi=200, facecolor="white")
print("wrote ppdf_beam.png")
