#!/usr/bin/env python3
"""One run of the CNR-against-iterations study (submit_iter_count_pair.sh).

Calls compute_cnr's own pipeline, compute_cnr.compute_cnr_for_work_dir (seeded
Poisson forward projection, ML-EM, CNR), with two changes that alter no number:

  1. Each PPDF matrix is read from disk once and kept in memory. run_mlem
     re-reads all 16 files on every iteration, about 9 GB a pass, so the
     reads, not the arithmetic, set the run time: 11 to 20 s an iteration in
     the repeats this study re-creates (read from /vscratch), about 1 s with
     the cache. The same arrays reach the same operations, so the output is
     unchanged; tests/test_pipeline.py checks this bit for bit.
  2. run_mlem keeps every iterate (SAVE_EVERY = 1) instead of every fifth.
     ML-EM's first k iterations do not depend on how many follow, so iterate k
     of one 400-iteration run is exactly the image a k-iteration run ends on.

Every iterate is then scored with run_recon_comparison.compute_cnr, the
function that scores a run's final image, into dense_cnr.csv in the run
directory: iteration, sector-mean CNR, pooled CNR and the six sectors.

Usage (from optimization/):
  python run_iter_count_pair.py --work_dir results/nd2fine_960 \\
      --phantom_path ../data/sai_10mm/hot_rods_phantom_10.0_mm_x_10.0_mm.pt \\
      --out_subdir iterpair_N400_T10_seed0 --seed 0 --iterations 400 --T_sec 10
"""
import argparse
import contextlib
import io
import os
import sys
import tempfile

import h5py
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import compute_cnr  # noqa: E402  (imports ../recon/run_recon_comparison as rrc)

rrc = compute_cnr.rrc


def score_iterates(recon_path, phantom_path):
    """CNR of every saved iterate, each scored exactly as a final image is.
    With SAVE_EVERY = 1, saved iterate k is the image after iteration k + 1."""
    z = np.load(recon_path)
    rows = []
    with tempfile.TemporaryDirectory() as tmp:
        p = os.path.join(tmp, "iterate.npz")
        for k, est in enumerate(z["estimates"]):
            np.savez(p, final=est)
            with contextlib.redirect_stdout(io.StringIO()):
                res = rrc.compute_cnr(p, phantom_path, tmp)
            s = np.asarray(res["sector_cnrs"], dtype=float)
            rows.append(dict(iteration=k + 1, cnr_sector_mean=float(np.nanmean(s)),
                             cnr_pooled=float(res["overall_cnr"]),
                             **{f"sector{i}": float(v) for i, v in enumerate(s)}))
    return pd.DataFrame(rows), len(z["diffs"])


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--work_dir", required=True)
    ap.add_argument("--phantom_path", required=True)
    ap.add_argument("--out_subdir", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--iterations", type=int, required=True)
    ap.add_argument("--T_sec", type=float, required=True)
    a = ap.parse_args()

    rrc.h5py = compute_cnr.CachedH5()
    rrc.SAVE_EVERY = 1
    _, sector_mean, _ = compute_cnr.compute_cnr_for_work_dir(
        a.work_dir, a.phantom_path, iterations=a.iterations, T_sec=a.T_sec,
        out_subdir=a.out_subdir, seed=a.seed)
    if np.isnan(sector_mean):
        sys.exit(f"ERROR: no CNR for {a.work_dir}; see the messages above")
    rrc.h5py = h5py            # drops the cache (about 9 GB) before scoring

    run_dir = os.path.join(a.work_dir, a.out_subdir)
    dense, n_run = score_iterates(os.path.join(run_dir, "recon_mlem_T8.npz"), a.phantom_path)
    # The last iterate is the run's final image, so its score must be the run's own.
    if len(dense) != n_run or dense.cnr_sector_mean.iloc[-1] != sector_mean:
        sys.exit(f"ERROR: {len(dense)} iterates for {n_run} iterations, or the last one "
                 f"scores {dense.cnr_sector_mean.iloc[-1]!r}, not the run's {sector_mean!r}")
    dense.to_csv(os.path.join(run_dir, "dense_cnr.csv"), index=False)
    print(f"scored {n_run} iterates -> {os.path.join(run_dir, 'dense_cnr.csv')}")


if __name__ == "__main__":
    main()
