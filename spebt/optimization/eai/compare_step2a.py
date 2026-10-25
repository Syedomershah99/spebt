#!/usr/bin/env python3
"""Compare Empire AI step 2 node A (Grace, ARM) with CCR (x86).

Reads the node A output folder copied from /projects (results_step2a.csv,
timings.txt, node_usage.txt, ppdf_stats.csv, runs/<design>/iterpair_*/) and the
CCR references, and prints:
  1. FWHM, ASCI, PPDS and MPXI of the six designs, EAI against CCR
  2. PPDF statistics of the two references, file by file
  3. CNR of the two references over 20 EAI seeds against CCR's 5: equivalence
     (TOST, Welch, 90% interval inside +/- MARGIN), the gain at 150 iterations
     and with each design at its own best, and the peak iteration counts
  4. time per design and the node's peak load and memory (sets designs per node)
Exact equality is not expected: ARM and x86 round float32 differently, so the
Poisson draws differ and CNR can only agree statistically.

Usage: python compare_step2a.py STEP2A_DIR REFS_CSV CCR_METRICS_CSV CCR_PPDF_STATS CCR_DENSE_CSV
"""
import glob
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

MARGIN = 0.15
BENCH, MOBO = "tmi_reference_000", "nd2fine_960"
METRICS = ["fwhm_weighted_mean", "asci_pct_fwhm0p45", "ppds_ring1", "mpxi_windowed_active_mean"]


def welch_ci(a, b, level=0.90):
    """Difference mean(a) - mean(b) and its two-sided Welch interval."""
    va, vb = a.var(ddof=1) / len(a), b.var(ddof=1) / len(b)
    df = (va + vb) ** 2 / (va ** 2 / (len(a) - 1) + vb ** 2 / (len(b) - 1))
    h = stats.t.ppf(0.5 + level / 2, df) * np.sqrt(va + vb)
    d = a.mean() - b.mean()
    return d, d - h, d + h


def main():
    step, refs_csv, met_csv, stats_csv, dense_csv = sys.argv[1:6]
    eai = pd.read_csv(os.path.join(step, "results_step2a.csv")).set_index("config")
    ccr = pd.read_csv(met_csv).set_index("config")
    refs = pd.read_csv(refs_csv).set_index("config")

    print("1. Deterministic metrics, EAI against CCR (relative difference)")
    for c in refs.index:
        if c not in eai.index:
            print(f"  {c[:36]:36s} MISSING on EAI")
            continue
        cells = [f"{m.split('_')[0]} {eai.at[c, m]:.5g} vs {ccr.at[c, m]:.5g} "
                 f"({(eai.at[c, m] - ccr.at[c, m]) / abs(ccr.at[c, m]):+.1e})" for m in METRICS]
        print(f"  {c[:36]:36s} " + " | ".join(cells))

    print("\n2. PPDF statistics, EAI against CCR (largest relative difference over 16 files)")
    a = pd.read_csv(os.path.join(step, "ppdf_stats.csv")).set_index(["design", "file"])
    b = pd.read_csv(stats_csv).set_index(["design", "file"])
    for d in (BENCH, MOBO):
        x, y = a.loc[d], b.loc[d]
        same_shape = bool((x["shape"] == y["shape"]).all())
        rs = ((x["sum"] - y["sum"]).abs() / y["sum"].abs()).max()
        rm = ((x["max"] - y["max"]).abs() / y["max"].abs()).max()
        nz = int((x["nonzero"] - y["nonzero"]).abs().max())
        print(f"  {d}: {len(x)} files, shapes {'match' if same_shape else 'DIFFER'}, sum {rs:.1e}, "
              f"max {rm:.1e}, nonzero count differs by up to {nz}")

    print(f"\n3. CNR, 20 EAI seeds against CCR's 5 (equivalence margin +/-{MARGIN})")
    dense = []
    for f in glob.glob(os.path.join(step, "runs", "*", "iterpair_N400_T10_seed*", "dense_cnr.csv")):
        d = f.split(os.sep)[-3]
        s = int(f.split("seed")[-1].split(os.sep)[0])
        dense.append(pd.read_csv(f, float_precision="round_trip").assign(design=d, seed=s))
    dense = pd.concat(dense)
    ccr_dense = pd.read_csv(dense_csv)
    ccr_dense = ccr_dense[ccr_dense.T_sec == 10.0]
    peaks, at150 = {}, {}
    for d in (BENCH, MOBO):
        e = dense[(dense.design == d) & (dense.iteration == 150)].cnr_sector_mean.to_numpy()
        c = refs.loc[d, [f"cnr_seed{k}" for k in range(5)]].to_numpy(float)
        diff, lo, hi = welch_ci(e, c)
        ok = -MARGIN < lo and hi < MARGIN
        at150[d] = e
        print(f"  {d}: EAI {e.mean():.3f} +/- {e.std(ddof=1):.3f} (n={len(e)}) vs CCR {c.mean():.3f} "
              f"+/- {c.std(ddof=1):.3f} (n=5): {diff:+.3f}, 90% CI [{lo:+.3f}, {hi:+.3f}] "
              f"{'EQUIVALENT' if ok else 'NOT shown equivalent'}")
        mc = dense[dense.design == d].groupby("iteration").cnr_sector_mean.mean()
        cc = ccr_dense[ccr_dense.design == d].groupby("iteration").cnr_sector_mean.mean()
        peaks[d] = (int(mc.idxmax()), mc.max(), int(cc.idxmax()), cc.max())
        print(f"    peak: EAI {mc.max():.3f} at {int(mc.idxmax())} iterations, CCR {cc.max():.3f} at "
              f"{int(cc.idxmax())}")
    g150 = 100 * (at150[MOBO].mean() / at150[BENCH].mean() - 1)
    gbest = 100 * (peaks[MOBO][1] / peaks[BENCH][1] - 1)
    print(f"  gain at 150 iterations: EAI {g150:+.1f}% (CCR +31.6%); each at its own best: "
          f"EAI {gbest:+.1f}% (CCR +16.2%)")

    print("\n4. Time and node load")
    print(open(os.path.join(step, "timings.txt")).read().rstrip())
    u = pd.read_csv(os.path.join(step, "node_usage.txt"), sep=" ", names=["t", "load", "mem_gb"])
    print(f"  peak load {u.load.max():.0f} on 144 cores; mean load {u.load.mean():.0f}; "
          f"peak memory {u.mem_gb.max():.0f} GB of 478")


if __name__ == "__main__":
    main()
