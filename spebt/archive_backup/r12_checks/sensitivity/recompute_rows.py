#!/usr/bin/env python3
"""Recompute a few PPDF rows with today's pymatcal code (the same calls as
arg_ppdf_t8.compute_pose) and compare them with the stored files, so the code read for
claim (a) is shown to be the code that produced the data. Read-only on results/.

  python recompute_rows.py OUT_DIR
"""
import glob
import json
import os
import sys

import h5py
import numpy as np
from torch import device, tensor

B = "/projects/academic/rutaoyao/Omer/spebt/spebt/spebt"
sys.path.insert(0, os.path.join(B, "pymatcal"))
from scanner_modeling._raytracer_2d._local_functions import (  # noqa: E402
    ppdf_2d_local, reduced_edges_2d_local, sfov_properties, subdivision_grid_rectangle)
from scanner_modeling.geometry_2d import (  # noqa: E402
    fov_tensor_dict, load_scanner_geometry_from_layout, load_scanner_layouts)

R = os.path.join(B, "optimization", "results")
DESIGNS = {"bench": ("tmi_reference_000", [0, 300, 700, 1500, 2500, 3359]),
           "mobo": ("nd2fine_960", [0, 350, 900, 1800, 2700, 3701])}


def main(out_dir):
    os.makedirs(out_dir, exist_ok=True)
    dev = device("cpu")
    mu = tensor([3.5, 0.5])
    grid = subdivision_grid_rectangle((1, 5))
    res = {}
    for tag, (d, crystals) in DESIGNS.items():
        lf = glob.glob(os.path.join(R, d, "scanner_layouts_*.tensor"))[0]
        layouts, _ = load_scanner_layouts(os.path.dirname(lf), os.path.basename(lf))
        pv, cv, pe, ce = load_scanner_geometry_from_layout(0, layouts)
        fn = os.path.join(R, d, "position_000_ppdfs_t8_00.hdf5")
        with h5py.File(fn, "r") as h:
            dx, dy = float(h.attrs["dx_mm"]), float(h.attrs["dy_mm"])
            stored = {c: h["ppdfs"][c].astype(np.float64) for c in crystals}
        fov = fov_tensor_dict((200, 200), (10, 10), (dx, dy), (5, 5))
        ids, pix, corners = sfov_properties(fov)
        ids1d = ids[:, :, 0] * fov["n pixels"][0] + ids[:, :, 1]
        out = []
        for c in crystals:
            row = np.zeros(40000)
            for s in range(int(fov["n subdivisions"].prod())):
                rp, rc = reduced_edges_2d_local(s, c, corners, pv, pe, cv, ce, dev)
                row[ids1d[s].numpy()] = ppdf_2d_local(s, c, pix, cv, rp, rc, grid, mu, dev).double().numpy()
            st = stored[c]
            diff = np.abs(row - st).max()
            out.append(dict(crystal=c, stored_sum=float(st.sum()), recomputed_sum=float(row.sum()),
                            max_abs_diff=float(diff), stored_max=float(st.max())))
            print(tag, json.dumps(out[-1]), flush=True)
        res[tag] = out
    with open(os.path.join(out_dir, "recompute_rows.json"), "w") as fh:
        json.dump(res, fh, indent=1)


if __name__ == "__main__":
    main(sys.argv[1])
