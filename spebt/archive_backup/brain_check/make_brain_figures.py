"""Figures for Dr. Yao: the brain template's design maps (arg_design_maps.py, PR #5), read inside
the 200 mm FOV circle, the strips of one crystal per layer, and the per-layer strip statistics."""
import json
import sys

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.collections import PolyCollection
from matplotlib.patches import Circle, Rectangle

B = "/projects/academic/rutaoyao/Omer/brain_maps_check"
sys.path.insert(0, f"{B}/code/beam-analysis")
from arg_design_maps import load_matrix

OUT = f"{B}/figs"
D = f"{B}/data"
TENSOR = ("/vscratch/grp-rutaoyao/Tridev/brain_spect/geometry/"
          "brain_spect_layouts_e90446e80e2b0ae1da75fd55d98c8d37_rot2_ang1p0deg_trans1x1_step0p0x0p0.tensor")
N = 200
EXT = (-100, 100, -100, 100)
INK, MUTED, GRID = "#1f2328", "#57606a", "#d0d7de"
LAYER_COL = {"inner": "#2a78d6", "middle": "#2ba36b", "outer": "#eb6834"}
plt.rcParams.update({"font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10,
                     "axes.edgecolor": GRID, "xtick.color": MUTED, "ytick.color": MUTED})

ix, iy = np.divmod(np.arange(N * N), N)
cx, cy = ix + 0.5 - 100.0, iy + 0.5 - 100.0           # pixel centres (mm)
in_fov = np.hypot(cx, cy) <= 100.0

lay = torch.load(TENSOR, weights_only=False)["layouts"]["position 000"]
det = lay["detector units"].numpy()
plates = lay["plate segments"].numpy()
centre = det.mean(1)
# layer = depth behind the panel's inner face: the largest projection on the six panel normals
theta0 = np.arctan2(*centre[np.argmin(np.hypot(*centre.T))][::-1])
panel_angles = theta0 + np.arange(6) * np.pi / 3
normals = np.stack([np.cos(panel_angles), np.sin(panel_angles)], 1)
depth = (centre @ normals.T).max(1)
d_sorted = np.sort(depth)
gaps = np.flatnonzero(np.diff(d_sorted) > 1.0)
cuts = (d_sorted[gaps] + d_sorted[gaps + 1]) / 2
assert len(cuts) == 2, cuts
layer = np.array(["inner", "middle", "outer"])[np.searchsorted(cuts, depth)]
stats = {"layer_depth_mm": [float(np.median(depth[layer == k])) for k in LAYER_COL],
         "crystals_per_layer": {k: int((layer == k).sum()) for k in LAYER_COL}}


def fov_only(v):
    out = v.astype(float).copy()
    out[~in_fov] = np.nan
    return out.reshape(N, N)


def draw_geometry(ax):
    ax.add_collection(PolyCollection(plates, facecolor="#444c56", edgecolor="none"))
    ax.add_collection(PolyCollection(det, facecolor=[LAYER_COL[l] for l in layer], edgecolor="none"))
    ax.add_patch(Rectangle((-100, -100), 200, 200, fill=False, ec=MUTED, lw=0.8, ls=":"))
    ax.add_patch(Circle((0, 0), 100, fill=False, ec=INK, lw=1.0, ls="--"))
    ax.set_xlim(-150, 150); ax.set_ylim(-150, 150); ax.set_aspect("equal")
    ax.set_xlabel("x (mm)"); ax.set_ylabel("y (mm)")
    for name, colour in LAYER_COL.items():
        ax.plot([], [], "s", color=colour, label=f"{name} layer")
    ax.plot([], [], "s", color="#444c56", label="collimator, r 120 mm")
    ax.plot([], [], "--", color=INK, lw=1.0, label="FOV, 200 mm")
    ax.plot([], [], ":", color=MUTED, lw=1.0, label="PPDF grid, 200 mm square")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=2, fontsize=7.5, frameon=False,
              handletextpad=0.4, columnspacing=1.0)


# Figure 1: geometry and the four maps inside the FOV circle
with h5py.File(f"{B}/out/brain_maps.hdf5") as h:
    m = {k: h[k][:].ravel() for k in ("wfwhm", "asci", "sensitivity", "ppds")}
with h5py.File(f"{B}/out/brain_maps_nofilter.hdf5") as h:
    m["asci_all"] = h["asci"][:].ravel()
panels = [("wfwhm", "wFWHM", " mm", "wFWHM (mm)", ".1f"),
          ("asci_all", "ASCI, all strips", "%", "ASCI (%)", ".1f"),
          ("asci", "ASCI, strips ≤ 10 mm", "%", "ASCI (%)", ".1f"),
          ("sensitivity", "Sensitivity", "", "probability (2-D)", ".3f"),
          ("ppds", "PPDS", " mm⁻²", "mm⁻² (2-D)", ".2f")]
fig, axes = plt.subplots(2, 3, figsize=(13.2, 8.4), layout="constrained")
draw_geometry(axes[0, 0])
axes[0, 0].set_title("Brain Template Geometry")
stats["fov_means"] = {}
for ax, (key, title, unit, bar, fmt) in zip(axes.ravel()[1:], panels):
    v = m[key][in_fov]
    lo = np.nanpercentile(v, 0.5) if key == "wfwhm" else 0
    im = ax.imshow(fov_only(m[key]).T, origin="lower", extent=EXT, cmap="viridis",
                   vmin=lo, vmax=np.nanpercentile(v, 99.5), interpolation="nearest")
    ax.add_patch(Circle((0, 0), 100, fill=False, ec=INK, lw=0.6))
    ax.set_title(f"{title}, mean {np.nanmean(v):{fmt}}{unit}")
    ax.set_xlabel("x (mm)"); ax.set_ylabel("y (mm)")
    fig.colorbar(im, ax=ax, label=bar, shrink=0.85)
    stats["fov_means"][key] = {"mean": float(np.nanmean(v)), "min": float(np.nanmin(v)), "max": float(np.nanmax(v))}
fig.savefig(f"{OUT}/brain_design_maps.png", dpi=200)
plt.close(fig)

# Figure 2: one crystal per layer near the middle of a panel, layout 0
ppdf = load_matrix(f"{D}/position_000_ppdfs.hdf5", "ppdfs", np.float64)
mask = load_matrix(f"{D}/beams_masks_configuration_00.hdf5", "beam_mask", np.int32)
with h5py.File(f"{D}/beams_properties_configuration_00.hdf5") as h:
    props = h["beam_properties"][:]
along = centre @ np.array([-normals[0, 1], normals[0, 0]])        # position along panel 0
on_panel0 = (centre @ normals[0]) >= depth - 1e-6
picks = {}
for name in LAYER_COL:
    cand = np.flatnonzero(on_panel0 & (layer == name))
    picks[name] = int(cand[np.argmin(np.abs(along[cand] - 10.0))])
fig, axes = plt.subplots(2, 3, figsize=(13.2, 8.6), layout="constrained")
for c, (name, i) in enumerate(picks.items()):
    img = ppdf[i].reshape(N, N)
    a0, a1 = axes[0, c], axes[1, c]
    a0.imshow(np.log10(img.T + 1e-12), origin="lower", extent=EXT, cmap="magma",
              vmin=np.log10(img.max()) - 3, vmax=np.log10(img.max()))
    rows = props[(props[:, 1] == i) & (props[:, 9] > 0)]
    lab = np.ma.masked_equal(mask[i].reshape(N, N).T, 0)
    a1.imshow(lab, origin="lower", extent=EXT, cmap="tab20", interpolation="nearest")
    for r in rows:
        a1.annotate(f"{r[4]:.1f}", (r[5], r[6]), fontsize=7, ha="center", color=INK,
                    bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none", alpha=0.8))
    for a in (a0, a1):
        a.add_patch(Circle((0, 0), 100, fill=False, ec="white" if a is a0 else INK, lw=0.7, ls="--"))
        a.plot(*centre[i], "^", color=LAYER_COL[name], ms=9, mec=INK)
        a.set_xlim(-115, 145); a.set_ylim(-130, 130); a.set_aspect("equal")
        a.set_xlabel("x (mm)"); a.set_ylabel("y (mm)")
    w = np.array([ppdf[i, mask[i] == r[2]].sum() for r in rows])
    a0.set_title(f"{name.title()} Layer Crystal {i}: PPDF (log, 3 decades)")
    a1.set_title(f"{len(rows)} strips, FWHM in mm; wFWHM {np.dot(w, rows[:, 4]) / w.sum():.1f} mm")
stats["example_crystals"] = picks
fig.savefig(f"{OUT}/brain_strips_by_layer.png", dpi=200)
plt.close(fig)

# Figure 3: strip FWHM by layer and each layer's share of the strips' PPDF, both layouts
rows_all, w_all = [], []
for L in (0, 1):
    pp = ppdf if L == 0 else load_matrix(f"{D}/position_001_ppdfs.hdf5", "ppdfs", np.float64)
    mk = mask if L == 0 else load_matrix(f"{D}/beams_masks_configuration_01.hdf5", "beam_mask", np.int32)
    with h5py.File(f"{D}/beams_properties_configuration_{L:02d}.hdf5") as h:
        p = h["beam_properties"][:]
    p = p[(p[:, 9] > 0) & np.isfinite(p[:, 3]) & (p[:, 4] > 0)]
    rows_all.append(p)
    w_all.append(np.array([pp[int(r[1]), mk[int(r[1])] == r[2]].sum() for r in p]))
p, w = np.vstack(rows_all), np.concatenate(w_all)
lyr = layer[p[:, 1].astype(int)]
fig, (h1, h2) = plt.subplots(1, 2, figsize=(12, 4.2), layout="constrained", width_ratios=[1.6, 1])
edges = np.logspace(np.log10(0.5), np.log10(70), 40)
stats["by_layer"] = {}
for name, colour in LAYER_COL.items():
    s = lyr == name
    h1.hist(p[s, 4], bins=edges, histtype="step", lw=1.6, color=colour,
            label=f"{name}: median {np.median(p[s, 4]):.1f} mm, wFWHM {np.dot(w[s], p[s, 4]) / w[s].sum():.1f} mm")
    stats["by_layer"][name] = {"strips": int(s.sum()), "median_fwhm": float(np.median(p[s, 4])),
                               "wfwhm": float(np.dot(w[s], p[s, 4]) / w[s].sum()),
                               "ppdf_share": float(w[s].sum() / w.sum())}
h1.set_xscale("log"); h1.set_xlabel("strip FWHM (mm)"); h1.set_ylabel("strips")
h1.set_title("Strip FWHM by Layer (both collimator positions)")
h1.legend(fontsize=8, frameon=False)
names = list(LAYER_COL)
shares = [stats["by_layer"][n]["ppdf_share"] * 100 for n in names]
h2.bar(names, shares, color=[LAYER_COL[n] for n in names])
for k, v in enumerate(shares):
    h2.text(k, v + 1, f"{v:.0f}%", ha="center", fontsize=9, color=INK)
h2.set_ylabel("% of all strip PPDF (counts)"); h2.set_ylim(0, max(shares) * 1.2)
h2.set_title("Where the Counts Come From")
fig.savefig(f"{OUT}/brain_layer_stats.png", dpi=200)
plt.close(fig)
stats["design_wfwhm_all_strips"] = float(np.dot(w, p[:, 4]) / w.sum())
json.dump(stats, open(f"{OUT}/brain_figure_stats.json", "w"), indent=1)
print(json.dumps(stats, indent=1))
