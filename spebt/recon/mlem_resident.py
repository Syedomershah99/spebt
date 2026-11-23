"""ML-EM with the system matrix kept in GPU memory.

The same arithmetic as run_recon_comparison.run_mlem, in the same summation order, but each PPDF file is
read and copied to the device once and the sensitivity map is summed once, instead of on every iteration.
On Empire AI (one RTX PRO 6000, the MIC 2026 design, 9.5 GB of PPDFs, 150 iterations) an iteration took
19 ms instead of 0.97 s, and the final image was identical (eai/gpu_profile/reports/results_both_150.json).
It needs the whole system matrix in device memory: about 4 to 10 GB for the small-animal designs.

Usage (on a GPU node; projections from run_recon_comparison.forward_project):
  python mlem_resident.py --ppdf_dir DIR --projs projections.npy --out recon.npz [--iters 150]
"""
import argparse
import os
import sys

import h5py
import numpy as np
import torch


def load_system(flist, sfov, device):
    """PPDF matrices of the files in flist, each viewed as (1, n_proj, sfov), and the clamped sensitivity map."""
    need = sum(os.path.getsize(f) for f in flist)
    if torch.device(device).type == "cuda":
        free, _ = torch.cuda.mem_get_info(device)
        if need > 0.9 * free:
            raise MemoryError(f"PPDFs need about {need / 1e9:.1f} GB, the GPU has {free / 1e9:.1f} GB free")
    mats = []
    for fname in flist:
        with h5py.File(fname, "r") as h5f:
            m = torch.from_numpy(h5f["ppdfs"][:]).to(device=device, dtype=torch.float32)
        mats.append(m.view(1, m.numel() // sfov, sfov))
    sens = torch.zeros(sfov, device=device, dtype=torch.float32)
    for m in mats:                                   # same order as run_mlem
        sens += torch.sum(m, dim=1).squeeze()
    return mats, torch.clamp(sens, min=1e-12)


def mlem(mats, sens, pdata, iters=150, tol=1e-4):
    """Unfiltered ML-EM from a uniform start; stops early, as run_mlem does, once the relative change < tol."""
    est = torch.ones_like(sens)
    diffs = []
    for _ in range(iters):
        prev = est.clone()
        bp = torch.zeros_like(sens)
        for i, m in enumerate(mats):
            y = torch.clamp(torch.matmul(m, est), min=1e-12)
            r = pdata[i].view(1, m.shape[1]) / y
            bp += torch.matmul(m.transpose(1, 2), r.unsqueeze(-1)).squeeze()
        est = est * (bp / sens)
        diffs.append(float((torch.norm(est - prev) / torch.norm(prev)).item()))   # one sync per iteration
        if diffs[-1] < tol:
            break
    return est, diffs


if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import run_recon_comparison as rrc

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ppdf_dir", required=True)
    ap.add_argument("--projs", required=True, help=".npy projections, one row per PPDF file")
    ap.add_argument("--out", required=True, help="output .npz (final image and per-iteration changes)")
    ap.add_argument("--iters", type=int, default=rrc.N_ITERATIONS)
    a = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, flist = rrc.generate_flist(a.ppdf_dir, os.path.dirname(os.path.abspath(a.out)))
    mats, sens = load_system(flist, rrc.SFOV, dev)
    pdata = torch.from_numpy(np.load(a.projs)).to(device=dev, dtype=torch.float32)
    est, diffs = mlem(mats, sens, pdata, a.iters, rrc.CONVERGENCE_TOL)
    np.savez_compressed(a.out, final=est.view(rrc.IMG_DIM, rrc.IMG_DIM).cpu().numpy(), diffs=np.array(diffs))
    print(f"{len(diffs)} iterations on {dev}; saved {a.out}")
