#!/usr/bin/env python3
"""Slide 15: rebuild the sensitivity map from per-file sums, split the factor 8, and
give per-position and 3-D-equivalent numbers for both colour bars.

  python analyze_maps.py
"""
import json
import math
import os

import numpy as np

W = os.path.dirname(os.path.abspath(__file__))
NPZ = "/Users/Omer/Desktop/RA/omer/spebt/archive_backup/design_maps"
MPP = 0.05
DA = MPP * MPP                     # pixel area, mm^2
D_AP = {"bench": 0.40, "mobo": 0.2718}
N_AP = {"bench": 180, "mobo": 211}
R_AP = 35.0                         # aperture-ring mid radius, mm (33.75 to 36.25)

ix, iy = np.arange(40000) // 200, np.arange(40000) % 200
x, y = (ix + 0.5) * MPP - 5, (iy + 0.5) * MPP - 5
inside = np.hypot(x, y) <= 5.0
CEN = [99 * 200 + 99, 99 * 200 + 100, 100 * 200 + 99, 100 * 200 + 100]


def stats(v):
    return dict(mean_square=float(v.mean()), mean_fov=float(v[inside].mean()),
                centre=float(v[CEN].mean()), max=float(v.max()), min_fov=float(v[inside].min()),
                p99_5=float(np.percentile(v, 99.5)))


def fmt(d, f="{:.4g}"):
    return {k: (f.format(v) if isinstance(v, float) else v) for k, v in d.items()}


out = {}
for tag in ("bench", "mobo"):
    z = np.load(os.path.join(NPZ, f"{tag}.npz"))
    c = np.load(os.path.join(W, "out", f"{tag}_colsums.npz"))
    col, ring_col, row = c["col"], c["ring_col"], c["row"]
    # 1. the slide's map = layouts averaged, poses summed
    rebuilt = col.sum(axis=1).mean(axis=0)
    o = {"rebuild_max_abs_diff_vs_npz_sens_all": float(np.abs(rebuilt - z["sens_all"]).max()),
         "npz_sens_all_mean": float(z["sens_all"].mean())}
    per_pos = col.mean(axis=(0, 1))                     # mean over the 16 frames
    o["sens_all_over_8_equals_frame_mean_max_abs_diff"] = float(np.abs(z["sens_all"] / 8 - per_pos).max())
    o["slide_sensitivity"] = stats(z["sens_all"])
    o["per_position_sensitivity"] = stats(per_pos)
    fm = col.reshape(16, -1)
    o["single_frame_mean_range"] = [float(fm.mean(1).min()), float(fm.mean(1).max())]
    o["single_frame_max_range"] = [float(fm.max(1).min()), float(fm.max(1).max())]
    o["single_frame_centre_range"] = [float(fm[:, CEN].mean(1).min()), float(fm[:, CEN].mean(1).max())]
    rc = ring_col.mean(axis=(0, 1))                     # (4, 40000) per-position, by ring
    o["per_position_by_ring_mean_square"] = [float(r.mean()) for r in rc]
    o["per_position_by_ring_centre"] = [float(r[CEN].mean()) for r in rc]
    # 2. PPDS (slide: ry_mask_all, x 1e-3 shown as "x10^3 mm^-2")
    dets = z["dets"]                                    # layout, det, ring1, k_all, k_win, A_s, A_x, S
    S_npz = dets[:, 7]
    S_files = np.concatenate([row[l].sum(axis=0) for l in (0, 1)])   # T8 sum per layout
    o["S_i_npz_vs_files_max_rel_diff"] = float(np.max(np.abs(S_npz - S_files) / np.maximum(S_files, 1e-12)))
    Ax = dets[:, 6]
    act = Ax > 0
    o["A_i_mm2_median_active"] = float(np.median(Ax[act]))
    o["S_i_T8sum_median_active"] = float(np.median(S_npz[act]))
    o["S_i_per_position_x_dA_median_mm2"] = float(np.median(S_npz[act]) / 8 * DA)
    ppds = z["ry_mask_all"]
    o["slide_ppds_per_mm2"] = stats(ppds)
    o["slide_ppds_colourbar_top_x1e3"] = float(np.percentile(ppds, 99.5) * 1e-3)
    o["per_position_ppds_per_mm2"] = stats(ppds / 8)
    o["per_position_ppds_dimensionless_x_dA"] = stats(ppds / 8 * DA)
    # 3. 2-D -> 3-D: pinhole d^2/(16 r^2) over slit d/(2 pi r) = pi d / (8 r)
    k3 = math.pi * D_AP[tag] / (8 * R_AP)
    o["k3d_pi_d_over_8r"] = k3
    o["open_fraction_2d_n_d_over_2pi_r"] = N_AP[tag] * D_AP[tag] / (2 * math.pi * R_AP)
    o["collimator_3d_fraction_n_d2_over_16r2"] = N_AP[tag] * D_AP[tag] ** 2 / (16 * R_AP ** 2)
    o["per_position_sensitivity_3d_estimate"] = stats(per_pos * k3)
    o["per_position_ppds_dimensionless_3d_estimate"] = stats(ppds / 8 * DA * k3)
    out[tag] = o

for tag in out:
    print(f"===== {tag}")
    for k, v in out[tag].items():
        print(f"  {k}: {fmt(v) if isinstance(v, dict) else v}")
b, m = out["bench"], out["mobo"]
print("===== MOBO / benchmark")
for key in ("per_position_sensitivity", "per_position_ppds_per_mm2", "per_position_sensitivity_3d_estimate"):
    print(f"  {key}: mean_square {m[key]['mean_square'] / b[key]['mean_square']:.3f}, "
          f"mean_fov {m[key]['mean_fov'] / b[key]['mean_fov']:.3f}, centre {m[key]['centre'] / b[key]['centre']:.3f}")
with open(os.path.join(W, "out", "analysis.json"), "w") as fh:
    json.dump(out, fh, indent=1)
