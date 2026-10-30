#!/usr/bin/env python3
"""Summarize the independent ray cast against the stored PPDF column sums."""
import json, os
import numpy as np
W = os.path.dirname(os.path.abspath(__file__))
rep = json.load(open(os.path.join(W, "out", "raycast_report.json")))
for tag in ("bench", "mobo"):
    rows = rep[tag]["raycast"]
    r = np.array([x["ratio"] for x in rows])
    rt = np.array([x["raycast_total"] for x in rows]); st = np.array([x["stored_total"] for x in rows])
    op = np.array([x["open_fraction"] for x in rows])
    print(f"{tag}: {len(rows)} points; stored/raycast ratio mean {r.mean():.3f}, median {np.median(r):.3f}, "
          f"min {r.min():.3f}, max {r.max():.3f}; pooled sum ratio {st.sum() / rt.sum():.3f}")
    print(f"   raycast totals {rt.min():.4f}..{rt.max():.4f}; capture given open (raycast/open) "
          f"near centre: " + ", ".join(f"{x['raycast_total'] / x['open_fraction']:.3f}" for x in rows
                                         if abs(x['point_mm'][0]) < 0.3 and abs(x['point_mm'][1]) < 0.3))
    g = rep[tag]["position 000"]
    print("   open fraction from origin", g["open_fraction_from_origin"], "plates", g["plate_segments"],
          "plate radii", [round(v, 3) for v in g["plate_vertex_radius_min_max"]],
          "aperture inner/outer", [round(v, 4) for v in g["mean_of_largest_half_gaps_inner_outer_mm"]])
    print("   crystals/ring", g["crystals_per_ring"], "centre radii", g["crystal_centre_radii_mm"],
          "packing", {k: round(v, 3) for k, v in g["tangential_packing_by_ring"].items()},
          "edges v0v1", [round(v, 4) for v in g["edge_v0v1_mm_min_max"]],
          "v0v3", [round(v, 4) for v in g["edge_v0v3_mm_min_max"]],
          "v0v3 radial |cos| min", round(g["v0v3_abs_cos_to_radial_min"], 6))
    z = np.load(os.path.join(W, "out", f"{tag}_percrystal_l0_p0_px99_99.npz"))
    c = np.load(os.path.join(W, "out", f"{tag}_colsums.npz"))
    e = c["ring_edges"]
    print("   pixel (99,99), layout 0, pose 0, point", z["point"].round(4).tolist())
    for k in range(4):
        a, b = e[k], e[k + 1]
        print(f"     ring {k + 1}: stored {z['stored'][a:b].sum():.4f}  raycast {z['raycast'][a:b].sum():.4f}  "
              f"crystals lit stored {(z['stored'][a:b] > 1e-6).sum()} raycast {(z['raycast'][a:b] > 1e-6).sum()}")
    print(f"     total: stored {z['stored'].sum():.4f}  raycast {z['raycast'].sum():.4f}")
rr = json.load(open(os.path.join(W, "out", "recompute_rows.json")))
for tag, rows in rr.items():
    print(tag, "recomputed rows, max abs diff", max(x["max_abs_diff"] for x in rows),
          "largest stored entry", max(x["stored_max"] for x in rows))
