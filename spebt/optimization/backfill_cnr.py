#!/usr/bin/env python3
"""
Backfill the cnr_mean column from existing ML-EM reconstruction outputs.

Scans <recon_root>/<config>/cnr_results.npz for each row of the results
CSV. If found, reads overall_cnr from the .npz and writes it into a
cnr_mean column on the matching row. Idempotent: rows that already have a
numeric cnr_mean are skipped unless --force is set.

This lets us populate the new CNR objective for the ~17 configurations we
have already reconstructed offline (baseline, lhs4d_3, lhs4d_10, mobo_0069,
mobo_0082, mobo_0084, mobo_0066, etc.) before the in-loop recon (via
compute_cnr.py) fills in new configs going forward.

Usage:
  python backfill_cnr.py \
      --results_csv results/results_summary_mobo.csv \
      --recon_root  results/recon_results
"""
import argparse
import os
import shutil
import sys
import time
import numpy as np
import pandas as pd


def _find_cnr_npz(recon_root: str, config_name: str):
    """Look for a cnr_results.npz under recon_root/<config>/ or recon_root/<config>_*."""
    direct = os.path.join(recon_root, config_name, "cnr_results.npz")
    if os.path.exists(direct):
        return direct
    # Some recon dirs use a slightly different name suffix (500iter, etc.)
    import glob
    candidates = sorted(glob.glob(os.path.join(recon_root, f"{config_name}*", "cnr_results.npz")))
    if candidates:
        # Prefer the longest-iteration run when multiple exist
        for c in candidates:
            if "500iter" in c:
                return c
        return candidates[0]
    return None


def main():
    parser = argparse.ArgumentParser(description="Backfill cnr_mean from existing recon outputs")
    parser.add_argument("--results_csv", required=True)
    parser.add_argument("--recon_root", required=True,
                        help="Directory holding per-config recon subfolders with cnr_results.npz")
    parser.add_argument("--force", action="store_true",
                        help="Recompute cnr_mean even if the row already has a value")
    args = parser.parse_args()

    if not os.path.exists(args.results_csv):
        print(f"ERROR: results CSV not found: {args.results_csv}")
        sys.exit(1)

    df = pd.read_csv(args.results_csv)
    n = len(df)
    print(f"Loaded {n} rows from {args.results_csv}")

    if "cnr_mean" not in df.columns:
        df["cnr_mean"] = float("nan")
        print("Added empty cnr_mean column.")

    stamp = time.strftime("%Y%m%d_%H%M%S")
    backup = args.results_csv.replace(".csv", f".bak.{stamp}.csv")
    shutil.copy(args.results_csv, backup)
    print(f"Backup written: {backup}")

    n_done = n_skip_have = n_skip_missing = n_failed = 0
    for i, row in df.iterrows():
        config = row.get("config")
        if not isinstance(config, str) or not config:
            n_skip_missing += 1
            continue

        existing = row.get("cnr_mean")
        if not args.force and pd.notna(existing):
            n_skip_have += 1
            continue

        npz_path = _find_cnr_npz(args.recon_root, config)
        if npz_path is None:
            n_skip_missing += 1
            continue

        try:
            nz = np.load(npz_path, allow_pickle=True)
            cnr_val = float(nz["overall_cnr"])
        except Exception as e:
            print(f"  [fail] {config}: failed to read {npz_path}: {e}")
            n_failed += 1
            continue

        df.at[i, "cnr_mean"] = cnr_val
        n_done += 1
        print(f"  [ok]   {config}: CNR = {cnr_val:.4f}   (from {os.path.basename(os.path.dirname(npz_path))})")

        if (n_done + n_failed) % 10 == 0 and (n_done + n_failed) > 0:
            df.to_csv(args.results_csv, index=False)

    df.to_csv(args.results_csv, index=False)

    print()
    print(f"Done. Computed: {n_done}, already had CNR: {n_skip_have}, "
          f"no recon output: {n_skip_missing}, failed: {n_failed}")
    print(f"Updated CSV: {args.results_csv}")
    print(f"Backup:      {backup}")


if __name__ == "__main__":
    main()
