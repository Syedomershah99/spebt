#!/usr/bin/env python3
"""Isolate ring-2 crystal count from ring-2 radius.

The 12 Aug `ring2_*` sweep walked the packing frontier: each seed raised
n_det_ring2 AND d2_inner together, because the packing limit scales with
radius. That confounds "more crystals" with "ring 2 further from the source",
so the sweep cannot say which one moved CNR.

Then the 5-seed recheck turned up something the sweep was not designed to
answer. `ring2_000` is `mobo_0296` with FOUR extra ring-2 crystals and an
identical d2, and it measures 4.509 +/- 0.124 against 4.722 +/- 0.069: a 0.21
drop at t = -3.4 from a 0.4% change in crystal count. The detector counts in
the PPDF files are exactly as requested, so it is not a malformed layout.

Either that is real and CNR is far more sensitive to ring-2 pitch than we have
been telling people, or something about sitting at 97% of the packing limit
(where PACKING_FRACTION puts these seeds) changes the geometry in a way the
count alone does not capture. It matters because the talk currently claims the
ring geometry is weakly constrained.

This sweep holds EVERY other parameter at mobo_0296, d2_inner included, and
moves only n_det_ring2. A flat line says the recheck caught a one-off and the
rings really are loose. A steep or structured line says pitch matters and the
claim needs qualifying.

Usage:
  python make_nd2_fine_seeds.py --results_csv results/results_summary_mobo.csv
  sbatch --array=0-5 --export=ALL,SEED_CSV=nd2_fine_seeds.csv,\\
      TASK_DIR=results/nd2_fine_seed_out submit_lhs6d_seeds.sh
"""
import argparse
import os
import sys

import pandas as pd

import mobo_agent as ma

# Bracket mobo_0296's 960 on both sides, with 964 included because that is the
# value the recheck measured. Spacing is deliberately fine: the question is
# whether a handful of crystals moves CNR, so a coarse grid would miss it.
ND2_VALUES = [944, 952, 960, 964, 972, 980]

BASE_CONFIG = "mobo_0296"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results_csv", required=True)
    ap.add_argument("--out", default="nd2_fine_seeds.csv")
    ap.add_argument("--prefix", default="nd2fine_")
    args = ap.parse_args()

    if not os.path.exists(args.results_csv):
        sys.exit(f"ERROR: results CSV not found: {args.results_csv}")

    df = pd.read_csv(args.results_csv)
    base = df[df["config"].astype(str).str.startswith(BASE_CONFIG)]
    if base.empty:
        sys.exit(f"ERROR: no row whose config starts with {BASE_CONFIG}")
    b = base.iloc[0]

    diam = float(b["aperture_diam_mm"])
    n_ap = int(b["n_apertures"])
    nd1 = int(b["n_det_ring1"])
    d2 = float(b["d2_inner_mm"])
    d3 = float(b["d3_inner_mm"])
    print(f"base {b['config']}")
    print(f"  aperture {diam:.6f} mm, {n_ap} apertures, ring1 {nd1}")
    print(f"  d2 {d2:.6f} mm, d3 {d3:.6f} mm  (both HELD FIXED)")

    cap = ma.max_crystals_on_ring(d2)
    print(f"  packing limit at this d2: {cap} crystals")

    rows, skipped = [], []
    for nd2 in ND2_VALUES:
        # Use the repo's own predicate rather than reimplementing the rule, so a
        # future change to the packing model cannot silently invalidate a sweep
        # that looks like it still runs.
        if not ma.is_ring_packing_ok(nd1, nd2, d2, d3):
            skipped.append(nd2)
            continue
        rows.append({
            "config": f"{args.prefix}{nd2}",
            "aperture_diam_mm": diam,
            "n_apertures": n_ap,
            "n_det_ring1": nd1,
            "n_det_ring2": nd2,
            "d2_inner_mm": d2,
            "d3_inner_mm": d3,
        })

    if skipped:
        print(f"  SKIPPED as unpackable at this d2: {skipped}")
    if not rows:
        sys.exit("ERROR: no requested n_det_ring2 value fits at this d2.")

    out = pd.DataFrame(rows)
    out.to_csv(args.out, index=False)
    print(f"\nwrote {args.out}: {len(out)} designs")
    print(out[["config", "n_det_ring2"]].to_string(index=False))
    print(f"\nsbatch --array=0-{len(out) - 1} --export=ALL,SEED_CSV={args.out},"
          f"TASK_DIR=results/nd2_fine_seed_out submit_lhs6d_seeds.sh")


if __name__ == "__main__":
    main()
