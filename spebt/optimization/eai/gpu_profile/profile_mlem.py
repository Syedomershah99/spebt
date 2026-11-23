"""Profile ML-EM on one GPU, Lab 3 style (Nsight Systems), for the MIC design.

Two versions on the same noisy projections:
  campaign  the campaign's own code: compute_cnr's in-memory PPDF cache and
            run_recon_comparison.run_mlem, which copies every system-matrix file
            host to device and re-sums the sensitivity map on every iteration
  resident  the same arithmetic with the matrices copied to the GPU once and the
            sensitivity map summed once
The script asserts the two final images agree before reporting timings.

Usage (on a GPU node):
  python profile_mlem.py --ppdf_dir DIR --phantom PT --out OUT --mode both --iters 150
  nsys profile -t cuda,nvtx,osrt -o REPORT python profile_mlem.py ... --mode campaign --iters 10
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
ap.add_argument("--code", default="/projects/su/ryao1/develop/oshah/code_a7e678e",
                help="code tree holding recon/ and optimization/ (the campaign's)")
ap.add_argument("--ppdf_dir", required=True)
ap.add_argument("--phantom", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--mode", choices=("campaign", "resident", "both"), default="both")
ap.add_argument("--iters", type=int, default=150)
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()

sys.path.insert(0, os.path.join(a.code, "recon"))
sys.path.insert(0, os.path.join(a.code, "optimization"))
import run_recon_comparison as rrc  # noqa: E402
from compute_cnr import CachedH5  # noqa: E402

nvtx = torch.cuda.nvtx
dev = torch.device("cuda")
assert torch.cuda.is_available(), "needs a GPU node"
os.makedirs(a.out, exist_ok=True)
rrc.h5py = CachedH5()                     # as compute_cnr does: each file read from disk once
rrc.N_ITERATIONS = a.iters

flist_path, flist = rrc.generate_flist(a.ppdf_dir, a.out)
torch.manual_seed(a.seed)
np.random.seed(a.seed)
projs_path = rrc.forward_project(flist, a.phantom, a.out, dev)   # campaign defaults: T 10 s, e_hot 10, e_bg 2


def timed(fn):
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    out = fn()
    torch.cuda.synchronize()
    return out, time.perf_counter() - t0, torch.cuda.max_memory_allocated() / 2**30


def run_campaign():
    out = os.path.join(a.out, "campaign")
    os.makedirs(out, exist_ok=True)
    nvtx.range_push("campaign run_mlem")
    path = rrc.run_mlem(flist, projs_path, out, dev)
    nvtx.range_pop()
    z = np.load(path)
    return np.asarray(z["final"]), len(z["diffs"])


def run_resident():
    sfov = rrc.SFOV
    nvtx.range_push("resident: load matrices to GPU")
    pdata = torch.from_numpy(np.load(projs_path)).to(device=dev, dtype=torch.float32)
    mats = []
    for fname in flist:
        with rrc.h5py.File(fname, "r") as h5f:
            m = torch.from_numpy(h5f["ppdfs"][:]).to(device=dev, dtype=torch.float32)
        mats.append(m.view(1, m.numel() // sfov, sfov))
    nvtx.range_pop()
    nvtx.range_push("resident: sensitivity once")
    sens = torch.zeros(sfov, device=dev, dtype=torch.float32)
    for m in mats:                         # same summation order as run_mlem
        sens += torch.sum(m, dim=1).squeeze()
    sens = torch.clamp(sens, min=1e-12)
    nvtx.range_pop()
    est = torch.ones(sfov, device=dev, dtype=torch.float32)
    n = 0
    for it in range(a.iters):
        nvtx.range_push(f"resident iter {it}")
        prev = est.clone()
        bp = torch.zeros(sfov, device=dev, dtype=torch.float32)
        for i, m in enumerate(mats):
            y = torch.clamp(torch.matmul(m, est), min=1e-12)
            r = pdata[i].view(1, m.shape[1]) / y
            bp += torch.matmul(m.transpose(1, 2), r.unsqueeze(-1)).squeeze()
        est = est * (bp / sens)
        diff = torch.norm(est - prev) / torch.norm(prev)
        n += 1
        stop = float(diff.item()) < rrc.CONVERGENCE_TOL   # one sync per iteration, as run_mlem
        nvtx.range_pop()
        if stop:
            break
    return est.view(rrc.IMG_DIM, rrc.IMG_DIM).cpu().numpy(), n


res = {"gpu": torch.cuda.get_device_name(0), "torch": torch.__version__, "cuda": torch.version.cuda,
       "files": len(flist), "matrix_GB": sum(os.path.getsize(f) for f in flist) / 1e9, "iters": a.iters}
images = {}
for mode, fn in (("campaign", run_campaign), ("resident", run_resident)):
    if a.mode in (mode, "both"):
        (img, n), sec, peak = timed(fn)
        images[mode] = img
        res[mode] = {"seconds": sec, "iterations_run": n, "seconds_per_iteration": sec / n,
                     "peak_gpu_GB": peak}
        print(f"{mode}: {sec:.1f} s for {n} iterations ({sec / n:.3f} s each), peak GPU {peak:.1f} GB")
if len(images) == 2:
    d = np.abs(images["campaign"] - images["resident"])
    res["max_abs_diff"] = float(d.max())
    res["max_rel_diff"] = float(d.max() / np.abs(images["campaign"]).max())
    res["speedup"] = res["campaign"]["seconds"] / res["resident"]["seconds"]
    print(f"images agree to {res['max_rel_diff']:.2e} (relative); speedup {res['speedup']:.1f}x")
    assert res["max_rel_diff"] < 1e-4, "the resident version changed the result"
with open(os.path.join(a.out, f"results_{a.mode}_{a.iters}.json"), "w") as f:
    json.dump(res, f, indent=2)
