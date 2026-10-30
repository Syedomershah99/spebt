"""Orientation check, run on CCR (read-only on the results directories).

Three questions, each answered as-is against transposed:
 1. phantom -> PPDF: does H @ phantom.reshape(-1) explain the archived seed-0 projections of the
    first PPDF file (projections_T8.npy[0]) better than H @ phantom.T.reshape(-1)?
 2. design maps -> PPDF: is sens_all in bench.npz / mobo.npz (same pixel loop as asci_win in
    compute_design_maps.py) equal to the PPDF column sums averaged over the two layouts?
 3. beam masks -> PPDF: what share of each detector's PPDF mass lies inside its strip mask
    (layout 000), as-is against transposed? asci_win is built from these masks.
"""
import glob
import os
import sys

import h5py
import numpy as np
import torch

R = "/projects/academic/rutaoyao/Omer/spebt/spebt/spebt"
WORK = {"bench": f"{R}/optimization/results/tmi_reference_000",
        "mobo": f"{R}/optimization/results/nd2fine_960"}
MAPS = {k: os.path.expanduser(f"~/r12_bias/{k}.npz") for k in WORK}
PHANTOM = f"{R}/data/sai_10mm/hot_rods_phantom_10.0_mm_x_10.0_mm.pt"
N = 200

ph = torch.load(PHANTOM, map_location="cpu", weights_only=False)["Phantom tensor"].numpy()
counts = np.where(ph > 0.1, 100.0, 20.0)            # T_sec * e_hot, T_sec * e_bg
c_as, c_tr = counts.reshape(-1), counts.T.reshape(-1)

for key, wd in WORK.items():
    files = sorted(glob.glob(f"{wd}/position_*_ppdfs_t8_*.hdf5"))
    flist = [ln.strip() for ln in open(f"{wd}/cnr_repeat_seed0/dataset_flist.csv") if ln.strip()]
    assert [os.path.basename(f) for f in files] == [os.path.basename(f) for f in flist]
    projs = np.load(f"{wd}/cnr_repeat_seed0/projections_T8.npy")
    z = np.load(MAPS[key])
    colsum = np.zeros(N * N)
    lay0 = None
    for n, f in enumerate(files):
        with h5py.File(f, "r") as h:
            a = h["ppdfs"][:]
        a = a.reshape(a.shape[0], -1)
        colsum += a.sum(axis=0, dtype=np.float64)
        if n == 0:
            q = projs[0].astype(np.float64)
            for tag, c in (("as-is", c_as), ("transposed", c_tr)):
                e = np.clip(a.astype(np.float64) @ c, 1e-12, None)
                print(f"[{key}] 1. projections of {os.path.basename(f)} vs H @ phantom {tag}: "
                      f"corr {np.corrcoef(q, e)[0, 1]:.6f}, mean (q-e)^2/e {np.mean((q - e) ** 2 / e):.3f}")
        if "position_000_" in os.path.basename(f):
            lay0 = a.astype(np.float64) if lay0 is None else lay0 + a
        del a
    sens = colsum / 2.0                               # maps are averaged over the 2 layouts
    sm = z["sens_all"]
    smT = sm.reshape(N, N).T.reshape(-1)
    print(f"[{key}] 2. sens_all vs PPDF column sums / 2: max rel diff as-is "
          f"{np.max(np.abs(sm - sens)) / sens.max():.2e}, transposed {np.max(np.abs(smT - sens)) / sens.max():.2e}; "
          f"corr as-is {np.corrcoef(sm, sens)[0, 1]:.6f}, transposed {np.corrcoef(smT, sens)[0, 1]:.6f}")
    with h5py.File(f"{wd}/beams_masks_configuration_000.hdf5", "r") as h:
        mk = h["beam_mask"][:]
    mk = mk.reshape(mk.shape[0], -1)
    tot = lay0.sum(axis=1)
    inside = np.array([lay0[i, mk[i] > 0].sum() for i in range(mk.shape[0])])
    mkT = mk.reshape(-1, N, N).transpose(0, 2, 1).reshape(mk.shape[0], -1)
    insideT = np.array([lay0[i, mkT[i] > 0].sum() for i in range(mk.shape[0])])
    ok = tot > 0
    print(f"[{key}] 3. share of layout-000 PPDF mass inside each detector's strip mask: "
          f"as-is {np.mean(inside[ok] / tot[ok]):.4f}, transposed {np.mean(insideT[ok] / tot[ok]):.4f} "
          f"(mask covers {np.mean((mk > 0).sum(axis=1)) / N / N:.4f} of the field on average)")
    del lay0, mk, mkT
    sys.stdout.flush()
