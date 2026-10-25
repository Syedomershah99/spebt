#!/usr/bin/env python3
"""Create the search directories for an Empire AI campaign (run_node_searches.sh).

Warm starts (paired with CCR): the 21 starting designs of each distinct CCR
head-to-head set, already evaluated on CCR, copied with all their metrics into
the search's results_summary_mobo.csv. The three distinct sets are those of arms
r0 (= the 5-objective r0), r1 (= r4) and r2 (= r5).

Fresh starts: 21 new Latin hypercube designs per search (make_lhs6d_seeds.py,
one LHS seed per search), written to lhs_pending.csv for the launcher to
evaluate before the search's controller starts. Set MOBO_RING2_MAX first so the
designs respect the same box as the search.

Refuses to touch a directory that already exists.

Usage (from the repo's optimization/ directory):
  MOBO_RING2_MAX=960 python eai/make_searches.py --root /ddn/scratch/.../searches \\
      --warm 3 --fresh 9 --first_seed 10
"""
import argparse
import os
import subprocess
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
OPT = os.path.dirname(HERE)
H2H = os.path.join(OPT, "..", "archive_backup", "h2h")
WARM_SOURCES = ["results_h2h_2obj.csv", "results_h2h_2obj_r1.csv", "results_h2h_2obj_r2.csv"]
PARAMS = ["aperture_diam_mm", "n_apertures", "n_det_ring1", "n_det_ring2", "d2_inner_mm", "d3_inner_mm"]


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", required=True)
    ap.add_argument("--warm", type=int, default=3, choices=range(0, 4))
    ap.add_argument("--fresh", type=int, default=9)
    ap.add_argument("--first_seed", type=int, default=10)
    ap.add_argument("--prefix", default="", help="Prepended to every search name")
    a = ap.parse_args()
    if not os.environ.get("MOBO_RING2_MAX"):
        sys.exit("ERROR: set MOBO_RING2_MAX (e.g. 960) so fresh designs use the search's box")

    made = []
    for i, src in enumerate(WARM_SOURCES[:a.warm]):
        d = pd.read_csv(os.path.join(H2H, src))
        lhs = d[d.config.astype(str).str.startswith("lhs")]
        if len(lhs) != 21:
            sys.exit(f"ERROR: {src} has {len(lhs)} starting designs, expected 21")
        name = f"{a.prefix}warm_{['r0', 'r1', 'r2'][i]}"
        path = os.path.join(a.root, name)
        os.makedirs(path)               # fails if it exists: never overwrite a search
        lhs.to_csv(os.path.join(path, "results_summary_mobo.csv"), index=False)
        made.append((name, f"21 evaluated rows from {src}"))

    for j in range(a.fresh):
        seed = a.first_seed + j
        name = f"{a.prefix}fresh_s{seed:02d}"
        path = os.path.join(a.root, name)
        os.makedirs(path)
        out = os.path.join(path, "lhs_seeds.csv")
        subprocess.run([sys.executable, os.path.join(OPT, "make_lhs6d_seeds.py"), "--n_seeds", "21",
                        "--seed", str(seed), "--prefix", f"lhs6d_e{seed:02d}_", "--out", out],
                       check=True, cwd=OPT, capture_output=True)
        seeds = pd.read_csv(out)
        seeds[["config"] + PARAMS].to_csv(os.path.join(path, "lhs_pending.csv"), index=False)
        made.append((name, f"21 LHS designs, seed {seed}, n_det_ring2 <= {seeds.n_det_ring2.max()}"))

    for name, what in made:
        print(f"{name}: {what}")
    print(" ".join(os.path.join(a.root, n) for n, _ in made))


if __name__ == "__main__":
    main()
