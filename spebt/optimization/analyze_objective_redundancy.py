#!/usr/bin/env python3
"""
Is this really a five-objective problem?

After the Jul 2026 objective revision, four of the five correlate strongly with
reconstructed CNR: FWHM -0.93, ASCI@0.45mm +0.80, PPDS ring 1 +0.60. Only MPXI
is independent, at essentially zero. That raises a question worth answering
before the next campaign: if four objectives largely restate the same thing,
the multi-objective formulation may be doing less work than it appears, and a
smaller set would search the same space with a sharper acquisition signal.

Symptoms already visible: 94 of 208 designs are Pareto-optimal. With five
objectives a point is dominated only if another beats it on all five, which
almost never happens, so the front stops discriminating and design selection
falls back to CNR with the rest as constraints.

This tests the question against the existing archive, with no new simulation:

  1. Correlation structure. Pairwise Spearman between every objective, and the
     effective dimensionality from a PCA of the standardised objective matrix.
  2. Pareto overlap. Which designs are non-dominated under the full set versus
     under smaller subsets. If the five-objective front is mostly the union of
     the two-objective front plus points nothing dominates by accident, the
     extra objectives are not defining trade-offs.
  3. Ranking agreement. Whether a subset would have selected the same top
     designs, which is the question that actually matters for the campaign.

Usage:
  python analyze_objective_redundancy.py --results_csv results/results_summary_mobo.csv
"""
import argparse
import itertools
import os
import sys

import numpy as np
import pandas as pd

import mobo_agent as ma

# Subsets worth testing. CNR is the outcome we care about; MPXI is the only
# objective carrying information independent of it.
SUBSETS = {
    "all five": ma.OBJ_COLUMNS,
    "CNR + MPXI": ["cnr_sector_mean", "mpxi_mean"],
    "CNR + MPXI + FWHM": ["cnr_sector_mean", "mpxi_mean", "fwhm_weighted_mean"],
    "CNR only": ["cnr_sector_mean"],
    "no CNR (the 4 proxies)": [c for c in ma.OBJ_COLUMNS if c != "cnr_sector_mean"],
}


def pareto_mask(obj_max: np.ndarray) -> np.ndarray:
    """Boolean mask of non-dominated rows, all objectives maximized."""
    n = len(obj_max)
    keep = np.ones(n, dtype=bool)
    for i in range(n):
        if not keep[i]:
            continue
        dominated = np.all(obj_max >= obj_max[i], axis=1) & np.any(obj_max > obj_max[i], axis=1)
        if dominated.any():
            keep[i] = False
    return keep


def to_max(df: pd.DataFrame, cols) -> np.ndarray:
    """Objective matrix in maximization space."""
    dirs = [ma.OBJ_DIRECTIONS[ma.OBJ_COLUMNS.index(c)] for c in cols]
    return df[cols].values * np.asarray(dirs, dtype=float)


def main():
    ap = argparse.ArgumentParser(description="Test whether the objective set is redundant")
    ap.add_argument("--results_csv", required=True)
    args = ap.parse_args()

    if not os.path.exists(args.results_csv):
        print(f"ERROR: results CSV not found: {args.results_csv}")
        sys.exit(1)

    df = pd.read_csv(args.results_csv).dropna(subset=ma.OBJ_COLUMNS + ma.PARAM_NAMES)
    if {"d2_inner_mm", "d3_inner_mm"}.issubset(df.columns):
        df = df[df.apply(lambda r: ma.is_ring_ordering_ok(r.d2_inner_mm, r.d3_inner_mm), axis=1)]
    df = df.reset_index(drop=True)
    n = len(df)
    print(f"{n} designs with all five objectives and valid geometry\n")

    short = {c: l.split(" (")[0] for c, l in zip(ma.OBJ_COLUMNS, ma.OBJ_NAMES)}

    print("=" * 74)
    print("1. HOW MUCH DO THE OBJECTIVES OVERLAP?")
    print("=" * 74)
    print("\nPairwise Spearman, in maximization space (so + means they agree):\n")
    mx = pd.DataFrame(to_max(df, ma.OBJ_COLUMNS), columns=[short[c] for c in ma.OBJ_COLUMNS])
    corr = mx.corr(method="spearman")
    print(corr.round(2).to_string())

    off = corr.values[np.triu_indices(len(corr), k=1)]
    print(f"\nmean |rho| between objectives: {np.abs(off).mean():.2f}")
    print("(near 0 would mean genuinely independent objectives)")

    # PCA on the standardised objectives: how many directions carry the variance
    z = (mx - mx.mean()) / mx.std().replace(0, 1)
    ev = np.linalg.eigvalsh(np.cov(z.values.T))[::-1]
    ev = ev / ev.sum()
    print("\nVariance explained by each principal direction:")
    cum = 0.0
    for i, e in enumerate(ev, 1):
        cum += e
        print(f"  PC{i}: {e:6.1%}   cumulative {cum:6.1%}")
    n_eff = int(np.searchsorted(np.cumsum(ev), 0.90) + 1)
    print(f"\n-> {n_eff} directions capture 90% of the variance across 5 objectives")

    print()
    print("=" * 74)
    print("2. DOES A SMALLER SET PICK THE SAME DESIGNS?")
    print("=" * 74)
    full_mask = pareto_mask(to_max(df, ma.OBJ_COLUMNS))
    full_set = set(df.index[full_mask])
    print(f"\n{'objective set':<26} {'Pareto':>7} {'% of all':>9} {'overlap with 5-obj':>19}")
    print("-" * 66)
    for label, cols in SUBSETS.items():
        cols = [c for c in cols if c in df.columns]
        if not cols:
            continue
        m = pareto_mask(to_max(df, cols))
        s = set(df.index[m])
        overlap = len(s & full_set) / len(s) if s else float("nan")
        print(f"{label:<26} {len(s):>7} {len(s)/n:>8.1%} {overlap:>18.0%}")

    print("""
A five-objective front covering most of the archive is the tell: with five
objectives a design is dominated only if another beats it everywhere, so almost
nothing is dominated and the front stops being a filter.""")

    print()
    print("=" * 74)
    print("3. WOULD A SMALLER SET HAVE CHOSEN THE SAME TOP DESIGNS?")
    print("=" * 74)
    print("\nTop 5 by CNR, and their rank under each subset's hypervolume")
    print("contribution (higher contribution = more valuable to that front):\n")

    top = df.nlargest(5, "cnr_sector_mean")
    print(f"{'design':<30} {'CNR':>7} " +
          " ".join(f"{lab.split()[0]:>10}" for lab in SUBSETS))
    print("-" * (40 + 11 * len(SUBSETS)))
    for idx, row in top.iterrows():
        cells = []
        for label, cols in SUBSETS.items():
            cols = [c for c in cols if c in df.columns]
            m = pareto_mask(to_max(df, cols))
            cells.append("front" if m[df.index.get_loc(idx)] else "  --")
        cfg = str(row["config"])[:28]
        print(f"{cfg:<30} {row['cnr_sector_mean']:>7.3f} " +
              " ".join(f"{c:>10}" for c in cells))

    print("""
How to read this:
  - If the top CNR designs sit on the front under every subset, the extra
    objectives are not what put them there, and a smaller set would have found
    them too.
  - If mean |rho| between objectives is high and two principal directions carry
    most of the variance, the problem is closer to two-objective than five. That
    is not a failure -- it is worth knowing, because a smaller set gives the
    acquisition function a sharper signal and a front that actually
    discriminates.
  - The "no CNR" row is the useful control: if the four proxies alone pick the
    same designs, they are doing their job as cheap stand-ins for an expensive
    reconstruction.
""")


if __name__ == "__main__":
    main()
