"""Diagnostics of the brain design maps (run after job_extract.sh): strip/mask consistency,
strip FWHMs and the 64 mm sentinel, the FOV circle against the square grid, and a QA figure
of single-detector PPDFs with their strips."""
import sys

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

B = "/projects/academic/rutaoyao/Omer/brain_maps_check"
sys.path.insert(0, f"{B}/code/beam-analysis")
from arg_design_maps import layout_maps, load_matrix, pixel_xy

cfg = yaml.safe_load(open(f"{B}/brain.yml"))
D = cfg["paths"]["data_output_dir"]
N, MM = (200, 200), (1.0, 1.0)
xy = pixel_xy(N, MM)
centre = xy + 0.5 - 100.0                       # pixel centres in mm, FOV centred at 0
rad = np.hypot(centre[:, 0], centre[:, 1])
in_fov, beyond = rad <= 100.0, rad > 122.0      # FOV circle; outside the collimator (120-122 mm)
SENTINEL = 64.0

t = torch.load(f"{cfg['paths']['scanner_layouts_dir']}/{cfg['paths']['scanner_layouts_filename']}", weights_only=False)
det_c = t["layouts"]["position 000"]["detector units"].mean(1).numpy()
inside_sq = (np.abs(det_c) <= 100).all(1)
print(f"detectors {det_c.shape[0]}, centres inside the 200 mm square: {inside_sq.sum()}")
print(f"pixels: FOV circle {in_fov.sum()}, beyond the collimator {beyond.sum()} of {rad.size}")

ppdfs, props, masks = {}, {}, {}
for L in (0, 1):
    ppdfs[L] = load_matrix(f"{D}/position_{L:03d}_ppdfs.hdf5", "ppdfs", np.float64)
    with h5py.File(f"{D}/beams_properties_configuration_{L:02d}.hdf5") as h:
        props[L] = h["beam_properties"][:]
    masks[L] = load_matrix(f"{D}/beams_masks_configuration_{L:02d}.hdf5", "beam_mask", np.int32)
    p, m = props[L], masks[L]
    det, bid = p[:, 1].astype(int), p[:, 2].astype(int)
    bad = 0
    for i in range(m.shape[0]):
        ids_m = set(np.unique(m[i][m[i] > 0]).tolist())
        ids_p = set(bid[det == i].tolist())
        bad += ids_m != ids_p
    fw = p[:, 4]
    good = np.isfinite(fw) & (fw > 0) & np.isfinite(p[:, 3])
    sent = good & (fw >= SENTINEL - 1e-6)
    w = np.array([ppdfs[L][d, m[d] == b].sum() for d, b in zip(det, bid)])
    print(f"layout {L}: {p.shape[0]} strips, detectors whose mask ids differ from their property rows: {bad}")
    print(f"  FWHM (mm) 5/25/50/75/95%: {np.round(np.percentile(fw[good], [5, 25, 50, 75, 95]), 2)}; "
          f"> 10 mm {np.mean(fw[good] > 10) * 100:.1f}%; at 64 mm {sent.sum()} strips "
          f"({sent.sum() / good.sum() * 100:.1f}%) carrying {w[sent].sum() / w[good].sum() * 100:.1f}% of the strips' PPDF")
    per_det = np.bincount(det[good], minlength=m.shape[0])
    print(f"  strips per detector: mean {per_det.mean():.2f}; detectors without strips {np.sum(per_det == 0)}; "
          f"inside-square detectors: mean {per_det[inside_sq].mean():.2f}, their median FWHM "
          f"{np.median(fw[good & inside_sq[det]]):.2f} mm vs {np.median(fw[good & ~inside_sq[det]]):.2f} mm")
    mask_px = (m > 0)
    print(f"  mask pixels beyond the collimator: {mask_px[:, beyond].sum() / mask_px.sum() * 100:.2f}% of all mask pixels")

with h5py.File(f"{B}/out/brain_maps.hdf5") as h:
    maps = {k: h[k][:].ravel() for k in ("wfwhm", "asci", "sensitivity", "ppds")}
with h5py.File(f"{B}/out/brain_maps_nofilter.hdf5") as h:
    asci_all = h["asci"][:].ravel()
for k, v in list(maps.items()) + [("asci_nofilter", asci_all)]:
    print(f"{k:14s} mean square {np.nanmean(v):.4g} | FOV circle {np.nanmean(v[in_fov]):.4g} "
          f"(min {np.nanmin(v[in_fov]):.4g}, max {np.nanmax(v[in_fov]):.4g}) | beyond collimator {np.nanmean(v[beyond]):.4g}; "
          f"NaN in FOV {int(np.isnan(v[in_fov]).sum())}")

# the 64 mm sentinel's effect: the same maps without those strips
sens, ppds, wn, wd = (np.zeros(rad.size) for _ in range(4))
cover = np.zeros((rad.size, 360), bool)
for L in (0, 1):
    p = props[L]
    keep = ~(np.isfinite(p[:, 4]) & (p[:, 4] >= SENTINEL - 1e-6))
    s, d, a, b = layout_maps(ppdfs[L], p[keep], masks[L], xy, 360, (0.0, 10.0), cover)
    sens += s / 2; ppds += d / 2; wn += a; wd += b
wf = np.divide(wn, wd, out=np.full(rad.size, np.nan), where=wd > 0)
print(f"without 64 mm strips, in the FOV circle: wFWHM {np.nanmean(wf[in_fov]):.4g} mm "
      f"(with {np.nanmean(maps['wfwhm'][in_fov]):.4g}); PPDS {np.mean(ppds[in_fov]):.4g} "
      f"(with {np.mean(maps['ppds'][in_fov]):.4g}); ASCI {np.mean(cover.sum(1)[in_fov]) / 3.6:.4g}")

# QA figure: three detectors (inner layer, outer layer, a centre inside the square) of layout 0
p0, m0 = props[0], masks[0]
r_det = np.hypot(det_c[:, 0], det_c[:, 1])
picks = [int(np.argmin(r_det)), int(np.argmax(r_det)), int(np.flatnonzero(inside_sq)[0])]
fig, axes = plt.subplots(2, 3, figsize=(13, 8.5), layout="constrained")
ext = (-100, 100, -100, 100)
for c, i in enumerate(picks):
    img = ppdfs[0][i].reshape(N)
    axes[0, c].imshow(np.log10(img.T + 1e-12), origin="lower", extent=ext, vmin=np.log10(img.max()) - 4, cmap="magma")
    axes[0, c].plot(*det_c[i], "c^", ms=8)
    axes[0, c].set_title(f"detector {i} (r = {r_det[i]:.0f} mm): log10 PPDF", fontsize=10)
    rows = p0[p0[:, 1] == i]
    axes[1, c].imshow(m0[i].reshape(N).T, origin="lower", extent=ext, cmap="tab20", interpolation="nearest")
    axes[1, c].plot(rows[:, 5], rows[:, 6], "wx")
    axes[1, c].plot(*det_c[i], "c^", ms=8)
    axes[1, c].set_title("strips: " + ", ".join(f"{f:.1f}" for f in rows[:, 4]) + " mm", fontsize=9)
    for ax in axes[:, c]:
        ax.add_patch(plt.Circle((0, 0), 100, fill=False, color="w", lw=0.8, ls="--"))
        ax.set_xlim(-150, 150); ax.set_ylim(-150, 150)
fig.savefig(f"{B}/out/brain_qa_detectors.png", dpi=110)
print("QA figure:", f"{B}/out/brain_qa_detectors.png", "detectors", picks)
