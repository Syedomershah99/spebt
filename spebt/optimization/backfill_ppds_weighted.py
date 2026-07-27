#!/usr/bin/env python3
"""
Backfill the ring-weighted PPDS variant and test it against CNR.

Background: overall sensitivity is a poor objective because wide, poorly
collimated PPDFs raise it while blurring the image and degrading CNR. RY's
proposal (Jul 2026) is to weight each detector's PPDF contribution by its ring
-- 1, 2, 3, 4 from innermost to outermost -- so that counts arriving on the
better-collimated outer rings are rewarded more.

The plain (unweighted) PPDS did not correlate with CNR when it was first tried,
which is why it was shelved. Before making the weighted variant a MOBO
objective we should check whether the weighting actually fixes that. This
script does the check cheaply: every evaluated config still has its PPDF,
beam-property and beam-mask HDF5 files on disk, so both PPDS variants are
recomputable without any re-simulation.

Writes `ppds_weighted_mean` into the CSV, then reports Spearman correlations of
both variants against CNR so the objective decision can be made on evidence.

Usage:
  python backfill_ppds_weighted.py --results_csv results/results_summary_mobo.csv

  # Try a different weighting without touching code:
  python backfill_ppds_weighted.py --results_csv ... --ring_weights 1,2,4,8

  # Report correlations from an earlier run without recomputing:
  python backfill_ppds_weighted.py --results_csv ... --analyze_only
"""
import argparse
import os
import shutil
import sys
import time

import numpy as np
import pandas as pd

from compute_metrics import compute_ppds, DEFAULT_RING_WEIGHTS

WEIGHTED_COL = "ppds_weighted_mean"


def report_correlations(df: pd.DataFrame) -> None:
    """Spearman of each PPDS variant against CNR -- the GO/NO-GO evidence."""
    if "cnr_mean" not in df.columns:
        print("\n(no cnr_mean column; skipping correlation report)")
        return

    print()
    print("=" * 72)
    print("DOES WEIGHTED PPDS TRACK CNR?")
    print("=" * 72)

    candidates = [
        ("ppds_mean", "PPDS (unweighted)"),
        (WEIGHTED_COL, "PPDS (ring-weighted)"),
        ("sensitivity_mean", "sensitivity (current objective)"),
    ]
    print(f"\n{'metric':<34} {'n':>5} {'Spearman vs CNR':>18}")
    print("-" * 60)
    for col, label in candidates:
        if col not in df.columns:
            continue
        sub = df[[col, "cnr_mean"]].dropna()
        if len(sub) < 3:
            print(f"{label:<34} {len(sub):>5} {'too few points':>18}")
            continue
        rho = sub[col].corr(sub["cnr_mean"], method="spearman")
        print(f"{label:<34} {len(sub):>5} {rho:>+18.3f}")

    n = len(df[[WEIGHTED_COL, "cnr_mean"]].dropna()) if WEIGHTED_COL in df else 0
    if n >= 3:
        crit = 1.96 / np.sqrt(n - 1)   # large-sample 5% critical value
        print(f"\nApprox. 5% critical |rho| at n={n}: {crit:.3f}")

    print("""
How to read this:
  - We want ring-weighted PPDS to be strongly POSITIVE against CNR. That would
    mean it can stand in for sensitivity as an objective without rewarding the
    wide, blurring PPDFs that sensitivity rewards.
  - Sensitivity's own correlation is the benchmark to beat; it was -0.88 on the
    5-objective campaign data, i.e. actively pulling against CNR.
  - If the weighted variant is still flat or negative, the weighting did not
    fix the problem and that is worth reporting back before building a campaign
    on it. Try other weightings via --ring_weights before concluding.
""")


def main():
    parser = argparse.ArgumentParser(
        description="Backfill ring-weighted PPDS and correlate it against CNR")
    parser.add_argument("--results_csv", required=True,
                        help="Path to results CSV to backfill in place")
    parser.add_argument("--ring_weights", type=str, default=None,
                        help="Comma-separated weights for rings 1-4 "
                             f"(default: {','.join(str(w) for w in DEFAULT_RING_WEIGHTS)})")
    parser.add_argument("--force", action="store_true",
                        help="Recompute even if the row already has a value")
    parser.add_argument("--analyze_only", action="store_true",
                        help="Only print the correlation report; compute nothing")
    parser.add_argument("--limit", type=int, default=None,
                        help="Stop after this many computed rows. Useful for timing "
                             "a few configs before committing to the full set; the "
                             "script is idempotent so a later full run resumes.")
    args = parser.parse_args()

    if not os.path.exists(args.results_csv):
        print(f"ERROR: results CSV not found: {args.results_csv}")
        sys.exit(1)

    df = pd.read_csv(args.results_csv)
    print(f"Loaded {len(df)} rows from {args.results_csv}")

    if args.analyze_only:
        report_correlations(df)
        return

    if args.ring_weights:
        ring_weights = tuple(float(w) for w in args.ring_weights.split(","))
        if len(ring_weights) != 4:
            print(f"ERROR: --ring_weights needs 4 values, got {len(ring_weights)}")
            sys.exit(1)
    else:
        ring_weights = DEFAULT_RING_WEIGHTS
    print(f"Ring weights (inner -> outer): {ring_weights}")

    for required in ("work_dir", "n_det_ring1"):
        if required not in df.columns:
            print(f"ERROR: results CSV has no '{required}' column; "
                  f"cannot resolve ring membership.")
            sys.exit(1)

    if WEIGHTED_COL not in df.columns:
        df[WEIGHTED_COL] = float("nan")
        print(f"Added empty {WEIGHTED_COL} column.")

    stamp = time.strftime("%Y%m%d_%H%M%S")
    backup = args.results_csv.replace(".csv", f".bak.{stamp}.csv")
    shutil.copy(args.results_csv, backup)
    print(f"Backup written: {backup}\n")

    n_done = n_skip_have = n_skip_missing = n_failed = 0
    t_start = time.time()
    for i, row in df.iterrows():
        if args.limit is not None and n_done >= args.limit:
            print(f"\nReached --limit {args.limit}; stopping early.")
            break

        config = row.get("config", f"row_{i}")
        work_dir = row.get("work_dir")
        n_det_ring1 = row.get("n_det_ring1")

        if not args.force and pd.notna(row.get(WEIGHTED_COL)):
            n_skip_have += 1
            continue
        if not isinstance(work_dir, str) or not os.path.isdir(work_dir):
            n_skip_missing += 1
            continue
        if pd.isna(n_det_ring1):
            print(f"  [skip] {config}: no n_det_ring1 in CSV")
            n_skip_missing += 1
            continue

        try:
            ppds_w = compute_ppds(work_dir, n_det_ring1=int(n_det_ring1),
                                  ring_weights=ring_weights)
        except Exception as e:
            print(f"  [fail] {config}: compute_ppds raised: {e}")
            n_failed += 1
            continue

        df.at[i, WEIGHTED_COL] = ppds_w
        if pd.notna(ppds_w):
            print(f"  [ok]   {config}: weighted PPDS = {ppds_w:.4e}")
            n_done += 1
        else:
            print(f"  [nan]  {config}: NaN (missing PPDF or mask files)")
            n_failed += 1

        # Checkpoint so a crash mid-run does not lose the completed rows
        if (n_done + n_failed) % 10 == 0 and (n_done + n_failed) > 0:
            df.to_csv(args.results_csv, index=False)

    df.to_csv(args.results_csv, index=False)

    elapsed = time.time() - t_start
    print()
    print(f"Done. Computed: {n_done}, already had a value: {n_skip_have}, "
          f"skipped: {n_skip_missing}, failed: {n_failed}")
    if n_done:
        print(f"Elapsed: {elapsed / 60:.1f} min "
              f"({elapsed / n_done:.1f} s per config)")
    print(f"Updated CSV: {args.results_csv}")
    print(f"Backup:      {backup}")

    report_correlations(df)


if __name__ == "__main__":
    main()
