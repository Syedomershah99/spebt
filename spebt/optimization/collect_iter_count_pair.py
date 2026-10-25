#!/usr/bin/env python3
"""Collect the runs of submit_iter_count_pair.sh into three small files.

  runs.csv        one row per run: design, arm, count level T_sec, seed,
                  iterations run (ML-EM stops early if the relative change drops
                  below 1e-4), the last relative change, and CNR at 150
                  iterations and at the last one
  dense.csv       every run's dense_cnr.csv (CNR after each iteration, scored by
                  run_iter_count_pair.py) with design, T_sec and seed
  validation.txt  the gate, which analyze_iter_count_pair.py requires to end in
                  PASS. Fails on any expected run that is missing or incomplete,
                  and on any T_sec 10 run that does not re-create the stored
                  repeat of the same design and seed (cnr_repeat_seed<S>): the
                  projections must match byte for byte and the CNR at iteration
                  150 to --tol. The count arm may be absent (not yet run) but not
                  partial.

validation.txt is deleted first, so a crash part way leaves no PASS behind, and
it ends with the sha256 of runs.csv and dense.csv, which the analysis checks.
Exits 1 on FAIL.

Usage (from optimization/ in the project-directory checkout):
  python3 -B collect_iter_count_pair.py
"""
import argparse
import filecmp
import hashlib
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "recon"))
from run_recon_comparison import CONVERGENCE_TOL  # noqa: E402
DESIGNS = ["tmi_reference_000", "nd2fine_960"]
ARMS = {"iter": [10.0], "count": [2.5, 5.0, 20.0, 40.0]}    # as in the submit script
SEEDS = range(5)
N = 400
GATE_ITER = 150                                              # the stored repeats' count
# mobo_0296 itself: nd2fine_960 is its rebuild, so its stored repeats are shown
# for reference. They differ from nd2fine_960's by up to 6.4e-7 (float32
# rounding), so they are not part of the gate.
HEADLINE = "mobo_0296_ap0.2718_nap211_nd1_582_nd2_960_d2_393_d3_492"


def tag(t_sec, seed):
    return f"iterpair_N{N}_T{t_sec:g}_seed{seed}".replace(".", "p")


def sector_mean(npz_path):
    s = np.asarray(np.load(npz_path)["sector_cnrs"], dtype=float)
    return float(np.nanmean(s)), s


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--results_dir", default=os.path.join(HERE, "results"))
    # About 20 float32 steps at a CNR near 4.7, and over 1,000 times below the
    # seed-to-seed SD (0.05 to 0.07).
    ap.add_argument("--tol", type=float, default=1e-5)
    a = ap.parse_args()
    out_dir = os.path.join(a.results_dir, "iter_count_pair")
    os.makedirs(out_dir, exist_ok=True)
    val_path = os.path.join(out_dir, "validation.txt")
    if os.path.exists(val_path):
        os.remove(val_path)

    lines, bad, rows, dense = [], 0, [], []
    for arm, levels in ARMS.items():
        expected = [(d, t, s) for d in DESIGNS for t in levels for s in SEEDS]
        present = [os.path.isdir(os.path.join(a.results_dir, d, tag(t, s))) for d, t, s in expected]
        if arm == "count" and not any(present):
            lines.append("count arm: not run")
            continue
        for d, t, s in expected:
            run = os.path.join(a.results_dir, d, tag(t, s))
            need = ["projections_T8.npy", "recon_mlem_T8.npz", "cnr_results.npz", "dense_cnr.csv"]
            missing = [f for f in need if not os.path.exists(os.path.join(run, f))]
            if missing:
                bad += 1
                lines.append(f"MISSING {d} {tag(t, s)}: {', '.join(missing)}")
                continue
            diffs = np.asarray(np.load(os.path.join(run, "recon_mlem_T8.npz"))["diffs"])
            # round_trip: pandas' default parser can be one float64 step off.
            dn = pd.read_csv(os.path.join(run, "dense_cnr.csv"), float_precision="round_trip")
            n_run = len(diffs)
            if dn.iteration.tolist() != list(range(1, n_run + 1)):
                bad += 1
                lines.append(f"INCOMPLETE {d} {tag(t, s)}: dense_cnr.csv does not cover "
                             f"iterations 1 to {n_run}")
                continue
            if n_run != N:
                # ML-EM stops early only once the relative change is below the
                # tolerance; a short run that had not got there was cut off.
                converged = diffs[-1] < CONVERGENCE_TOL
                bad += not converged
                lines.append(f"{'stopped early (converged)' if converged else 'TRUNCATED'}: "
                             f"{d} {tag(t, s)} ran {n_run} of {N}")
            fin, _ = sector_mean(os.path.join(run, "cnr_results.npz"))
            at = dn.set_index("iteration")
            g = at.loc[GATE_ITER] if GATE_ITER in at.index else None
            rows.append(dict(design=d, arm=arm, T_sec=t, seed=s, n_run=n_run,
                             last_rel_change=float(diffs[-1]),
                             cnr150_sector_mean=np.nan if g is None else g.cnr_sector_mean,
                             cnr150_pooled=np.nan if g is None else g.cnr_pooled,
                             cnr_final_sector_mean=fin))
            dense.append(dn.assign(design=d, T_sec=t, seed=s))

            if t != 10.0:
                continue
            # The gate: this run must re-create the stored repeat of the same seed.
            ref = os.path.join(a.results_dir, d, f"cnr_repeat_seed{s}")
            ref_files = [os.path.join(ref, f) for f in ("projections_T8.npy", "cnr_results.npz",
                                                        "recon_mlem_T8.npz")]
            if not all(map(os.path.exists, ref_files)) or g is None:
                bad += 1
                lines.append(f"NO GATE {d} seed {s}: stored repeat missing, or the run "
                             f"stopped before iteration {GATE_ITER}")
                continue
            same_proj = filecmp.cmp(os.path.join(run, "projections_T8.npy"), ref_files[0],
                                    shallow=False)
            ref_cnr, ref_sec = sector_mean(ref_files[1])
            dc = g.cnr_sector_mean - ref_cnr
            dsec = np.nanmax(np.abs(g[[f"sector{i}" for i in range(len(ref_sec))]].to_numpy(float)
                                    - ref_sec))
            est = np.load(os.path.join(run, "recon_mlem_T8.npz"))["estimates"][GATE_ITER - 1]
            dimg = float(np.abs(est - np.load(ref_files[2])["final"]).max())
            ok = same_proj and abs(dc) <= a.tol
            bad += not ok
            lines.append(f"{'ok' if ok else 'DIFFERS'} {d} seed {s}: projections "
                         f"{'identical' if same_proj else 'DIFFER'}; CNR at {GATE_ITER} "
                         f"{g.cnr_sector_mean:.6f} vs stored {ref_cnr:.6f} ({dc:+.1e}, "
                         f"largest sector {dsec:.1e}); image max diff {dimg:.1e}")
            if d == "nd2fine_960":
                h = os.path.join(a.results_dir, HEADLINE, f"cnr_repeat_seed{s}", "cnr_results.npz")
                if os.path.exists(h):
                    lines.append(f"  for reference, mobo_0296 seed {s}: {sector_mean(h)[0]:.6f} "
                                 f"({g.cnr_sector_mean - sector_mean(h)[0]:+.1e})")

    if not rows:
        bad += 1
        lines.append("no runs found")
    else:
        pd.DataFrame(rows).sort_values(["design", "T_sec", "seed"]).to_csv(
            os.path.join(out_dir, "runs.csv"), index=False)
        pd.concat(dense).to_csv(os.path.join(out_dir, "dense.csv"), index=False)
        for name in ("runs.csv", "dense.csv"):
            with open(os.path.join(out_dir, name), "rb") as f:
                lines.append(f"sha256 {name} {hashlib.sha256(f.read()).hexdigest()}")
    lines.append("PASS" if bad == 0 else f"FAIL: {bad} problem(s)")
    text = "\n".join(lines)
    with open(val_path, "w") as f:
        f.write(text + "\n")
    print(f"{len(rows)} runs -> {out_dir}\n{text}")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
