#!/usr/bin/env python3
"""How tightly independent searches agree on the optimum.

This produces the reproducibility result the MIC talk reports: across every
independent search we have run, the recovered aperture diameter agrees to
roughly 2% of the range searched, while the ring parameters scatter over ten
times that. It is the answer to "did you just find one lucky local optimum".

WHY THIS EXISTS AS A SCRIPT. The number was originally computed ad hoc, and on
2026-10-06 that turned out to matter. Merging the hand-placed `ring2_*` and
`seed6d_*` seed designs into the main archive changed which row has the highest
in-loop CNR -- `seed6d_02` reads 4.8744 against `mobo_0296`'s 4.7686 -- so a
naive re-run silently reported 1.7% where the talk says 1.8%, along with shifts
of up to 2.4 points in the other parameters.

Two separate things make the naive selection wrong:

  1. Seed designs were placed by hand to cover corners of the space. They are
     not things the optimizer found, so including one as a search's "optimum"
     does not describe search behaviour at all.
  2. In-loop CNR is a single Poisson draw (sigma 0.07 to 0.12). `seed6d_02`'s
     5-seed mean is 4.6850, BELOW `mobo_0296`'s 4.7224, so its archive-topping
     value is a high draw rather than a better design.

So selection is restricted to optimizer-proposed configs. Pass
--include_seeds to see the contaminated version.

Usage:
  python analyze_search_convergence.py
  python analyze_search_convergence.py --results_csv results/results_summary_mobo.csv
"""
import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd

import mobo_agent as ma

# Designs the optimizer proposed. Everything else in an archive is a seed:
# `lhs4d_*`, `d2d3_*`, `seed6d_*`, `ring2_*`, `nd2fine_*`, `nd2comm_*`,
# `tmi_reference_*`. `lhs6d_*` are the head-to-head starting seeds and are
# excluded for the same reason.
PROPOSED_PREFIXES = ("mobo_",)


def best_proposed(df: pd.DataFrame, source: str):
    """Highest in-loop CNR among optimizer-proposed, fully-measured designs."""
    need = ["cnr_sector_mean"] + ma.PARAM_NAMES
    missing = [c for c in need if c not in df.columns]
    if missing:
        print(f"  [warn] {source} lacks {missing}; skipped")
        return None
    sub = df[df["config"].astype(str).str.startswith(PROPOSED_PREFIXES)]
    sub = sub.dropna(subset=need)
    if sub.empty:
        print(f"  [warn] {source} has no optimizer-proposed design with "
              f"a CNR and a complete design vector; skipped")
        return None
    return sub.loc[sub["cnr_sector_mean"].idxmax()]


def collect(results_csv: str, include_seeds: bool = False):
    """One row per independent search: the main campaign and each h2h arm."""
    opt_dir = os.path.dirname(os.path.dirname(os.path.abspath(results_csv)))
    sources = sorted(glob.glob(os.path.join(opt_dir, "results_h2h_*",
                                            os.path.basename(results_csv))))
    sources.append(results_csv)

    rows, labels = [], []
    for f in sources:
        if not os.path.exists(f):
            continue
        df = pd.read_csv(f)
        if include_seeds:
            sub = df.dropna(subset=["cnr_sector_mean"] + ma.PARAM_NAMES)
            r = None if sub.empty else sub.loc[sub["cnr_sector_mean"].idxmax()]
        else:
            r = best_proposed(df, f)
        if r is None:
            continue
        rows.append([float(r[c]) for c in ma.PARAM_NAMES]
                    + [float(r["cnr_sector_mean"]),
                       float(r.get("fwhm_weighted_mean", np.nan))])
        labels.append(os.path.basename(os.path.dirname(f)) or "main")
    return labels, np.array(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results_csv", default="results/results_summary_mobo.csv")
    ap.add_argument("--include_seeds", action="store_true",
                    help="Also let hand-placed seed designs be a search's optimum. "
                         "Reproduces the contaminated numbers; not for reporting.")
    args = ap.parse_args()

    if not os.path.exists(args.results_csv):
        sys.exit(f"ERROR: not found: {args.results_csv}")

    labels, X = collect(args.results_csv, args.include_seeds)
    n = len(labels)
    if n < 3:
        sys.exit(f"ERROR: only {n} searches found; need at least 3 to quote a spread.")

    scope = "INCLUDING hand-placed seeds (contaminated)" if args.include_seeds \
            else "optimizer-proposed designs only"
    print(f"{n} independent searches, {scope}\n")

    print(f"{'parameter':22s} {'mean':>10s} {'sd':>9s} {'sd/range':>9s}")
    for j, c in enumerate(ma.PARAM_NAMES):
        v = X[:, j]
        rng = ma.BOUNDS_MAX[j] - ma.BOUNDS_MIN[j]
        print(f"{c:22s} {v.mean():10.3f} {v.std(ddof=1):9.3f} "
              f"{100 * v.std(ddof=1) / rng:8.1f}%")

    fw = X[:, -1]
    fw = fw[np.isfinite(fw)]
    if fw.size:
        print(f"\nweighted FWHM across the optima: {fw.min():.3f} to {fw.max():.3f} mm")
    print(f"best in-loop CNR per search: {X[:, -2].min():.3f} to {X[:, -2].max():.3f}")
    print("\n(in-loop CNR is a single draw; re-measure any design over seeds "
          "before quoting it)")


if __name__ == "__main__":
    main()
