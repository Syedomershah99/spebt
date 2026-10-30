"""Robustness check on CCR: inner/outer split on the iteration study's reconstructions
(read-only): both designs, scan times 2.5 to 40 s, 5 seeds, iterates 50, 150 and 400.

Each run's dense_cnr.csv was scored by run_recon_comparison.compute_cnr; the all-rod sector-mean
computed here must match it at every iterate used. The T10 iterate 150 is also compared with the
cnr_repeat_seed reconstruction of the same seed (the step 1 images).
"""
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.expanduser("~/r12_bias"))
from rods import load_phantom, masks, rods  # noqa: E402

R = "/projects/academic/rutaoyao/Omer/spebt/spebt/spebt"
RES = f"{R}/optimization/results"
ph, radii = load_phantom(f"{R}/data/sai_10mm/hot_rods_phantom_10.0_mm_x_10.0_mm.pt")
hot, bg, sectors = masks(ph)
RD, _ = rods(ph, radii)
dist = np.array([r["dist_mm"] for r in RD])
sec = np.array([r["sector"] for r in RD])
CUTS = {"2.50": 2.50, "2.60": 2.60, "median": float(np.median(dist))}


def pix_sets(sel):
    return [np.logical_or.reduce([RD[n]["mask"] for n in range(len(RD)) if sel[n] and sec[n] == k])
            for k in range(6)]


SETS = {"all": pix_sets(np.ones(len(RD), bool))}
for c, T in CUTS.items():
    SETS[f"in_{c}"] = pix_sets(dist < T)
    SETS[f"out_{c}"] = pix_sets(dist >= T)


def score(img):
    v = img[bg]
    mu, sd = v.mean(), v.std() + 1e-12
    return {k: float(np.mean([(img[p].mean() - mu) / sd for p in ps])) for k, ps in SETS.items()}


rows, worst, worst_rep = [], 0.0, 0.0
for design in ("tmi_reference_000", "nd2fine_960"):
    for T in ("T2p5", "T5", "T10", "T20", "T40"):
        for s in range(5):
            d = f"{RES}/{design}/iterpair_N400_{T}_seed{s}"
            z = np.load(f"{d}/recon_mlem_T8.npz")
            est = z["estimates"]
            dense = pd.read_csv(f"{d}/dense_cnr.csv").set_index("iteration")
            assert est.shape[0] == 400 and len(dense) == 400
            for it in (50, 150, 400):
                img = est[it - 1].astype(np.float64)
                sc = score(img)
                worst = max(worst, abs(sc["all"] - dense.loc[it, "cnr_sector_mean"]))
                rows.append(dict(design=design, T=T, seed=s, iteration=it, **sc))
            if T == "T10":
                rep = np.load(f"{RES}/{design}/cnr_repeat_seed{s}/recon_mlem_T8.npz")["final"]
                worst_rep = max(worst_rep, float(np.abs(est[149] - rep).max() / np.abs(rep).max()))
            del est, z
        print(f"done {design} {T}", flush=True)

df = pd.DataFrame(rows)
out = os.path.expanduser("~/r12_bias/iterpair_split.csv")
df.to_csv(out, index=False)
print(f"\nall-rod sector-mean vs dense_cnr.csv: max |diff| {worst:.2e}")
print(f"T10 iterate 150 vs cnr_repeat_seed final: max relative |diff| {worst_rep:.2e}")
print(f"wrote {out}\n")

for c in CUTS:
    print(f"cut {c} ({CUTS[c]:.3f} mm): mean over 5 seeds; gain = MOBO / benchmark")
    print(f"  {'T':>5} {'iter':>4} | {'bench in':>8} {'bench out':>9} | {'MOBO in':>7} {'MOBO out':>8} | "
          f"{'gain in':>7} {'gain out':>8} | {'all: bench':>10} {'MOBO':>5}")
    for T in ("T2p5", "T5", "T10", "T20", "T40"):
        for it in (50, 150, 400):
            q = df[(df["T"] == T) & (df.iteration == it)]
            b = q[q.design == "tmi_reference_000"]
            m = q[q.design == "nd2fine_960"]
            bi, bo, mi, mo = (b[f"in_{c}"].mean(), b[f"out_{c}"].mean(), m[f"in_{c}"].mean(), m[f"out_{c}"].mean())
            print(f"  {T:>5} {it:>4} | {bi:8.3f} {bo:9.3f} | {mi:7.3f} {mo:8.3f} | "
                  f"{mi / bi:7.3f} {mo / bo:8.3f} | {b['all'].mean():10.3f} {m['all'].mean():5.3f}")
    print()
