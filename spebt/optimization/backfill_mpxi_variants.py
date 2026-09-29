#!/usr/bin/env python3
"""
Merge the MPXI variant columns into the results CSV.

The objective set now uses mpxi_windowed_active_mean (RY approved Aug 2026).
compute_metrics.py writes it for new configs, but every design evaluated before
that change has only the old mpxi_mean. Without this backfill the optimizer sees
a NaN objective on the entire existing archive and drops all of it.

Reads the per-config file written by analyze_mpxi_variants.py --out, so the
expensive HDF5 pass is done once rather than repeated here.

Usage:
  python backfill_mpxi_variants.py \
      --results_csv results/results_summary_mobo.csv \
      --variants_csv results/mpxi_variants.csv
"""
import argparse
import os
import shutil
import sys
import time

import pandas as pd

VARIANT_COLS = ["mpxi_active_mean", "mpxi_windowed_mean",
                "mpxi_windowed_active_mean"]


def main():
    ap = argparse.ArgumentParser(description="Backfill MPXI variant columns")
    ap.add_argument("--results_csv", required=True)
    ap.add_argument("--variants_csv", required=True)
    ap.add_argument("--dry_run", action="store_true",
                    help="Report what would change without writing")
    args = ap.parse_args()

    for p in (args.results_csv, args.variants_csv):
        if not os.path.exists(p):
            print(f"ERROR: not found: {p}")
            sys.exit(1)

    res = pd.read_csv(args.results_csv)
    var = pd.read_csv(args.variants_csv)

    if "config" not in res.columns or "config" not in var.columns:
        print("ERROR: both CSVs need a 'config' column to join on")
        sys.exit(1)

    have = [c for c in VARIANT_COLS if c in var.columns]
    if not have:
        print(f"ERROR: {args.variants_csv} has none of {VARIANT_COLS}")
        sys.exit(1)

    var = var[["config"] + have].dropna(subset=["config"])
    if var["config"].duplicated().any():
        n = int(var["config"].duplicated().sum())
        print(f"[warn] {n} duplicate configs in the variants file; keeping the last")
        var = var.drop_duplicates(subset=["config"], keep="last")

    key = "mpxi_windowed_active_mean"
    before = int(res[key].notna().sum()) if key in res.columns else 0

    # FILL, do not REPLACE. This used to drop the existing variant columns and
    # left-join the incoming file, which is correct only when that file covers
    # the whole archive. Run on a subset -- the documented way to backfill a few
    # rows -- it silently erased every config the subset did not mention. On
    # 2026-09-30 that would have taken the archive from 279 usable MPXI values
    # to 6 and cut the GP's training set by 98%, reported as the unremarkable
    # line "331 still missing it".
    merged = res.copy()
    for c in have:
        incoming = var.set_index("config")[c]
        mapped = merged["config"].map(incoming)
        if c in merged.columns:
            merged[c] = mapped.combine_first(merged[c])
        else:
            merged[c] = mapped

    if len(merged) != len(res):
        print(f"ERROR: merge changed the row count ({len(res)} -> {len(merged)}); "
              f"aborting rather than corrupting the archive")
        sys.exit(1)

    after = int(merged[key].notna().sum())
    n_missing = len(merged) - after
    print(f"{len(merged)} configs in the archive")
    print(f"  {before} had {key} before, {after} after (+{after - before})")
    print(f"  {len(var)} config(s) supplied by {os.path.basename(args.variants_csv)}")
    print(f"  {n_missing} still missing it")
    if n_missing:
        missing = merged.loc[merged[key].isna(), "config"].head(10).tolist()
        print(f"  first few missing: {missing}")
        print("  (these will be dropped from training until their work_dirs are"
              "\n   readable and analyze_mpxi_variants.py is rerun)")

    # A backfill must never reduce what the archive knows. If it would, the
    # variants file is the wrong shape for this archive and the run is a
    # mistake, not a refresh.
    if after < before:
        print(f"\nERROR: this would REDUCE populated {key} values from {before} "
              f"to {after}.")
        print("       A backfill only ever adds. Check that --variants_csv "
              "covers the configs")
        print("       you intend to change, and rerun analyze_mpxi_variants.py "
              "if it does not.")
        sys.exit(1)

    if args.dry_run:
        print("\ndry run, nothing written")
        return

    backup = f"{args.results_csv}.bak_{time.strftime('%Y%m%d_%H%M%S')}"
    shutil.copy2(args.results_csv, backup)
    merged.to_csv(args.results_csv, index=False)
    print(f"\nwrote {args.results_csv}")
    print(f"backup at {backup}")


if __name__ == "__main__":
    main()
