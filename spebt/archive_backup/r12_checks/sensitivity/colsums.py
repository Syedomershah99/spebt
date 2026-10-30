#!/usr/bin/env python3
"""Per-file (layout, T8 pose) sums of the stored PPDFs, read-only.

For each design and each of the 16 files position_LLL_ppdfs_t8_PP.hdf5 this saves
  col[l, p, j]      = sum_i PPDF_ij            (the per-position sensitivity map)
  ring_col[l,p,r,j] = the same over ring r only
  row[l, p, i]      = sum_k PPDF_ik            (each detector's pixel sum)
  dxdy[l, p]        = the pose offset stored in the file
so the slide-15 maps (T8 poses SUMMED within a layout, then layouts averaged) can be
rebuilt and compared with per-position values.

  python colsums.py OUT_DIR
"""
import glob
import os
import sys
import time

import h5py
import numpy as np

R = "/projects/academic/rutaoyao/Omer/spebt/spebt/spebt/optimization/results"
DESIGNS = {"bench": ("tmi_reference_000", 480), "mobo": ("nd2fine_960", 582)}
N3, N4 = 960, 1200
BLOCK = 400


def ring_edges(n_det, n1):
    n2 = n_det - n1 - N3 - N4
    return np.cumsum([0, n1, n2, N3, N4])


def main(out_dir):
    os.makedirs(out_dir, exist_ok=True)
    for tag, (d, n1) in DESIGNS.items():
        files = sorted(glob.glob(os.path.join(R, d, "position_*_ppdfs_t8_*.hdf5")))
        assert len(files) == 16, (d, len(files))
        with h5py.File(files[0], "r") as h:
            n_det, n_pix = h["ppdfs"].shape
        edges = ring_edges(n_det, n1)
        col = np.zeros((2, 8, n_pix))
        ring_col = np.zeros((2, 8, 4, n_pix))
        row = np.zeros((2, 8, n_det))
        dxdy = np.zeros((2, 8, 2))
        n_neg = 0
        vmax = 0.0
        for f in files:
            t0 = time.time()
            with h5py.File(f, "r") as h:
                lay, pose = int(h.attrs["layout_idx"]), int(h.attrs["pose_idx"])
                dxdy[lay, pose] = (float(h.attrs["dx_mm"]), float(h.attrs["dy_mm"]))
                ds = h["ppdfs"]
                for a in range(0, n_det, BLOCK):
                    blk = ds[a:a + BLOCK].astype(np.float64)
                    n_neg += int((blk < 0).sum())
                    vmax = max(vmax, float(blk.max()))
                    col[lay, pose] += blk.sum(axis=0)
                    row[lay, pose, a:a + blk.shape[0]] = blk.sum(axis=1)
                    for r in range(4):
                        lo, hi = max(edges[r], a), min(edges[r + 1], a + blk.shape[0])
                        if hi > lo:
                            ring_col[lay, pose, r] += blk[lo - a:hi - a].sum(axis=0)
            print(f"{tag} {os.path.basename(f)} layout {lay} pose {pose} "
                  f"mean {col[lay, pose].mean():.6f} max {col[lay, pose].max():.6f} "
                  f"({time.time() - t0:.1f} s)", flush=True)
        np.savez_compressed(os.path.join(out_dir, f"{tag}_colsums.npz"), col=col, ring_col=ring_col,
                            row=row, dxdy=dxdy, ring_edges=edges, n_neg=n_neg, vmax=vmax)
        print(f"{tag}: n_det {n_det}, ring edges {edges.tolist()}, negative entries {n_neg}, "
              f"largest single PPDF entry {vmax:.6g}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
