#!/usr/bin/env python3
"""Independent check of what sum_i PPDF_ij means, written without pymatcal code.

1. Geometry of both layouts from the .tensor files: crystals per ring, crystal width and
   depth, which crystal edge the (1, 5) subdivision runs along, plate (aperture ring)
   radii, aperture widths, and the in-plane open fraction seen from a point.
2. A 2-D ray cast from a point source: N rays uniform in angle, exact chords through every
   convex quad (Cyrus-Beck), attenuation exp(-mu L) in plates (3.5/mm) and crystals
   (0.5/mm), absorption 1 - exp(-mu L) in each crystal reached. Its mean over rays is the
   probability that an in-plane photon emitted at the point is absorbed in a crystal, and
   its per-crystal split is the "exact" PPDF column. Compared with the stored files.

  python raycast_check.py OUT_DIR
"""
import glob
import json
import math
import os
import sys

import h5py
import numpy as np
import torch

R = "/projects/academic/rutaoyao/Omer/spebt/spebt/spebt/optimization/results"
DESIGNS = {"bench": "tmi_reference_000", "mobo": "nd2fine_960"}
MU_PLATE, MU_CRY = 3.5, 0.5
MPP, NPIX = 0.05, 200
N_RAYS = 2_000_000


def layout(d):
    f = glob.glob(os.path.join(R, d, "scanner_layouts_*.tensor"))[0]
    return f, torch.load(f, weights_only=False)["layouts"]


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def geometry_report(P, C):
    out = {}
    cc = C.mean(1)
    rc = np.hypot(cc[:, 0], cc[:, 1])
    rings = np.round(rc).astype(int)
    vals, counts = np.unique(rings, return_counts=True)
    out["crystal_centre_radii_mm"] = vals.tolist()
    out["crystals_per_ring"] = counts.tolist()
    e01 = np.linalg.norm(C[:, 1] - C[:, 0], axis=1)
    e03 = np.linalg.norm(C[:, 3] - C[:, 0], axis=1)
    out["edge_v0v1_mm_min_max"] = [float(e01.min()), float(e01.max())]
    out["edge_v0v3_mm_min_max"] = [float(e03.min()), float(e03.max())]
    rhat = cc / rc[:, None]
    u3 = (C[:, 3] - C[:, 0]) / e03[:, None]
    out["v0v3_abs_cos_to_radial_min"] = float(np.abs((u3 * rhat).sum(1)).min())
    rv = np.hypot(C[..., 0], C[..., 1])
    out["crystal_vertex_radius_by_ring"] = {
        int(v): [float(rv[rings == v].min()), float(rv[rings == v].max())] for v in vals}
    # tangential packing: crystal width x count / circumference at the crystal centre
    out["tangential_packing_by_ring"] = {
        int(v): float(e01[rings == v].mean() * (rings == v).sum() / (2 * np.pi * v)) for v in vals}
    pr = np.hypot(P[..., 0], P[..., 1])
    out["plate_segments"] = int(P.shape[0])
    out["plate_vertex_radius_min_max"] = [float(pr.min()), float(pr.max())]
    return out


def open_fraction(P, point, n=4_000_000):
    """Fraction of in-plane directions from `point` that miss every plate polygon."""
    th = -np.pi + 2 * np.pi * (np.arange(n) + 0.5) / n
    blocked = np.zeros(n, bool)
    rel = P - point
    ang = np.arctan2(rel[..., 1], rel[..., 0])
    cang = np.arctan2(rel[..., 1].mean(1), rel[..., 0].mean(1))
    dd = wrap(ang - cang[:, None])
    lo, hi = cang + dd.min(1), cang + dd.max(1)
    for a, b in zip(lo, hi):
        k0 = int(math.ceil((a + np.pi) * n / (2 * np.pi) - 0.5))
        k1 = int(math.floor((b + np.pi) * n / (2 * np.pi) - 0.5))
        blocked[np.arange(k0, k1 + 1) % n] = True
    return 1.0 - blocked.mean()


def aperture_widths(P):
    """Gaps between neighbouring plate polygons along the ring, as straight-line distance
    between the facing edges at the inner and outer plate radius (approximate)."""
    cang = np.arctan2(P[..., 1].mean(1), P[..., 0].mean(1))
    order = np.argsort(cang)
    Q = P[order]
    ang = np.arctan2(Q[..., 1], Q[..., 0])
    r = np.hypot(Q[..., 0], Q[..., 1])
    gaps_in, gaps_out = [], []
    n = len(Q)
    for k in range(n):
        a, b = Q[k], Q[(k + 1) % n]
        ca = cang[order][k]
        # vertices of a on its leading side and of b on its trailing side
        da = wrap(np.arctan2(a[:, 1], a[:, 0]) - ca)
        db = wrap(np.arctan2(b[:, 1], b[:, 0]) - ca)
        ra, rb = np.hypot(a[:, 0], a[:, 1]), np.hypot(b[:, 0], b[:, 1])
        ia_in = np.argmax(np.where(ra < np.median(ra), da, -9))
        ia_out = np.argmax(np.where(ra >= np.median(ra), da, -9))
        ib_in = np.argmin(np.where(rb < np.median(rb), db, 9))
        ib_out = np.argmin(np.where(rb >= np.median(rb), db, 9))
        gaps_in.append(np.linalg.norm(a[ia_in] - b[ib_in]))
        gaps_out.append(np.linalg.norm(a[ia_out] - b[ib_out]))
    g_in, g_out = np.array(gaps_in), np.array(gaps_out)
    return g_in, g_out


def chords(polys, point, n_rays):
    """All (ray, polygon) pairs with a positive chord: ray index, polygon index, t_enter, L."""
    rel = polys - point
    ang = np.arctan2(rel[..., 1], rel[..., 0])
    cang = np.arctan2(rel[..., 1].mean(1), rel[..., 0].mean(1))
    dd = wrap(ang - cang[:, None])
    lo, hi = cang + dd.min(1), cang + dd.max(1)
    k0 = np.ceil((lo + np.pi) * n_rays / (2 * np.pi) - 0.5).astype(np.int64)
    k1 = np.floor((hi + np.pi) * n_rays / (2 * np.pi) - 0.5).astype(np.int64)
    cnt = np.maximum(k1 - k0 + 1, 0)
    pid = np.repeat(np.arange(len(polys)), cnt)
    start = np.repeat(k0, cnt)
    off = np.arange(cnt.sum()) - np.repeat(np.cumsum(cnt) - cnt, cnt)
    kk = start + off
    th = -np.pi + 2 * np.pi * (kk + 0.5) / n_rays
    kk = kk % n_rays
    u = np.stack((np.cos(th), np.sin(th)), 1)
    # outward normals per polygon edge
    v = polys
    w = np.roll(v, -1, axis=1)
    e = w - v
    area = 0.5 * (v[..., 0] * w[..., 1] - w[..., 0] * v[..., 1]).sum(1)
    sgn = np.where(area > 0, 1.0, -1.0)
    nrm = np.stack((e[..., 1], -e[..., 0]), -1) * sgn[:, None, None]   # outward for CCW
    t_in = np.zeros(len(kk))
    t_out = np.full(len(kk), np.inf)
    ok = np.ones(len(kk), bool)
    for i in range(4):
        n_i = nrm[pid, i]
        num = ((v[pid, i] - point) * n_i).sum(1)
        den = (u * n_i).sum(1)
        with np.errstate(divide="ignore", invalid="ignore"):
            t = num / den
        t_in = np.where(den < 0, np.maximum(t_in, t), t_in)
        t_out = np.where(den > 0, np.minimum(t_out, t), t_out)
        ok &= ~((den == 0) & (num < 0))
    L = np.where(ok, t_out - t_in, 0.0)
    keep = L > 0
    return kk[keep], pid[keep], t_in[keep], L[keep]


def raycast(P, C, point, n_rays=N_RAYS):
    polys = np.concatenate([P, C], 0)
    mu = np.r_[np.full(len(P), MU_PLATE), np.full(len(C), MU_CRY)]
    k, pid, t0, L = chords(polys, point, n_rays)
    o = np.lexsort((t0, k))
    k, pid, L = k[o], pid[o], L[o]
    ex = mu[pid] * L
    cum = np.cumsum(ex)
    first = np.r_[True, k[1:] != k[:-1]]
    seg_start = np.maximum.accumulate(np.where(first, np.arange(len(k)), 0))
    before = cum - ex - (cum[seg_start] - ex[seg_start])
    p = np.exp(-before) * (1 - np.exp(-ex))
    cry = pid >= len(P)
    per_cry = np.bincount(pid[cry] - len(P), weights=p[cry], minlength=len(C)) / n_rays
    # geometric part only: directions whose ray reaches a crystal without crossing a plate
    plate_hit = np.zeros(n_rays, bool)
    plate_hit[k[~cry]] = True
    return per_cry, 1.0 - plate_hit.mean()


def pixel_xy(ix, iy, dx, dy):
    return np.array([(ix + 0.5) * MPP - 5 + dx, (iy + 0.5) * MPP - 5 + dy])


def main(out_dir):
    os.makedirs(out_dir, exist_ok=True)
    report = {}
    pixels = [(99, 99), (100, 100), (117, 92), (160, 100), (100, 40), (30, 170), (170, 170), (5, 100)]
    for tag, d in DESIGNS.items():
        f, L = layout(d)
        rep = {"layout_file": os.path.basename(f), "positions": sorted(L.keys())}
        for li, key in enumerate(sorted(L.keys())):
            P = L[key]["plate segments"].double().numpy()
            C = L[key]["detector units"].double().numpy()
            g = geometry_report(P, C)
            g_in, g_out = aperture_widths(P)
            n_ap = P.shape[0] // 2
            gi, go = np.sort(g_in), np.sort(g_out)
            g["gap_inner_mm_percentiles_0_25_50_75_100"] = np.percentile(g_in, [0, 25, 50, 75, 100]).round(4).tolist()
            g["gap_outer_mm_percentiles_0_25_50_75_100"] = np.percentile(g_out, [0, 25, 50, 75, 100]).round(4).tolist()
            g["mean_of_largest_half_gaps_inner_outer_mm"] = [float(gi[-n_ap:].mean()), float(go[-n_ap:].mean())]
            g["mean_of_smallest_half_gaps_inner_outer_mm"] = [float(gi[:n_ap].mean()), float(go[:n_ap].mean())]
            g["open_fraction_from_origin"] = open_fraction(P, np.zeros(2))
            rep[key] = g
            print(tag, key, json.dumps(g, indent=1), flush=True)
        # ray cast vs stored columns, layout 0 and 1, pose 0 and 4
        rc = []
        for li in (0, 1):
            key = f"position {li:03d}"
            P = L[key]["plate segments"].double().numpy()
            C = L[key]["detector units"].double().numpy()
            for pose in (0, 4):
                fn = os.path.join(R, d, f"position_{li:03d}_ppdfs_t8_{pose:02d}.hdf5")
                with h5py.File(fn, "r") as h:
                    dx, dy = float(h.attrs["dx_mm"]), float(h.attrs["dy_mm"])
                    cols = {px: h["ppdfs"][:, px[0] * NPIX + px[1]].astype(np.float64) for px in pixels}
                for px in pixels:
                    pt = pixel_xy(px[0], px[1], dx, dy)
                    per_cry, geo_open = raycast(P, C, pt)
                    stored = cols[px]
                    big = per_cry > 1e-4
                    rel = (stored[big] - per_cry[big]) / per_cry[big]
                    row = dict(layout=li, pose=pose, pixel=px, point_mm=pt.round(4).tolist(),
                               raycast_total=float(per_cry.sum()), stored_total=float(stored.sum()),
                               ratio=float(stored.sum() / per_cry.sum()),
                               open_fraction=float(geo_open),
                               n_cry_compared=int(big.sum()),
                               per_crystal_corr=float(np.corrcoef(stored[big], per_cry[big])[0, 1]),
                               per_crystal_rel_diff_median=float(np.median(rel)),
                               per_crystal_rel_diff_p05_p95=[float(np.percentile(rel, 5)), float(np.percentile(rel, 95))])
                    rc.append(row)
                    print(tag, json.dumps(row), flush=True)
                    if li == 0 and pose == 0 and px == (99, 99):
                        np.savez(os.path.join(out_dir, f"{tag}_percrystal_l0_p0_px99_99.npz"),
                                 raycast=per_cry, stored=stored, point=pt)
        rep["raycast"] = rc
        report[tag] = rep
    with open(os.path.join(out_dir, "raycast_report.json"), "w") as fh:
        json.dump(report, fh, indent=1, default=str)


if __name__ == "__main__":
    main(sys.argv[1])
