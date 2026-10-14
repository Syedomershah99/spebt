#!/usr/bin/env python3
"""Recompute ASCI at a finer angular bin for every config that still has
its beam files.

At the 1-degree bin the production ASCI column saturates: 87% of designs sit
at 100%, so the metric cannot rank them and contributes nothing to the search.
At 0.1 degree none saturate (measured 2026-10-14, n=60). This produces the
finer column so the archive carries a version that discriminates.

Costs no ray tracing: ASCI comes from the beam property and mask files, which
survive for 459 configs even after the scratch purge took most PPDFs.

  python compute_asci_finebins.py --bins 3600 --out results/asci_finebins.csv
  python backfill_mpxi_variants.py --results_csv <archive> \\
      --variants_csv results/asci_finebins.csv

The backfill is shared rather than reimplemented: it fills without replacing
and refuses any run that would reduce what the archive knows.
"""
import argparse
import glob
import os
import sys

import pandas as pd

from analyze_asci_binsize import asci_at_bins, configs_with_beam_files


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results_csv", default="results/results_summary_mobo.csv")
    ap.add_argument("--results_dir", default="results")
    ap.add_argument("--bins", type=int, default=3600,
                    help="angular bins over 0..2pi; 3600 = 0.1 degree")
    ap.add_argument("--out", default="results/asci_finebins.csv")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    col = f"asci_pct_bin{360 / args.bins:g}deg".replace(".", "p")

    opt_dir = os.path.dirname(os.path.dirname(os.path.abspath(args.results_csv)))
    sources = [args.results_csv] + sorted(glob.glob(os.path.join(
        opt_dir, "results_h2h_*", os.path.basename(args.results_csv))))
    frames = []
    for f in sources:
        if os.path.exists(f):
            try:
                frames.append(pd.read_csv(f))
            except Exception as e:
                print(f"  [warn] unreadable archive {f}: {e}")
    if not frames:
        sys.exit(f"ERROR: no readable archive at {args.results_csv}")
    df = pd.concat(frames, ignore_index=True)
    df = df.dropna(subset=["aperture_diam_mm"]).drop_duplicates(
        subset=["config"], keep="first")

    # cnr is not needed to COMPUTE asci, only to analyse it later
    if "cnr_sector_mean" not in df.columns:
        df["cnr_sector_mean"] = float("nan")
    rows = configs_with_beam_files(df, args.results_dir)
    print(f"{len(frames)} archive(s), {len(df)} configs, "
          f"{len(rows)} with surviving beam files")
    if args.limit:
        rows = rows[:args.limit]
    if not rows:
        sys.exit("ERROR: no config has readable beam files; nothing to compute.")

    recs = []
    for i, (cfg, wd, _ap, _cnr) in enumerate(rows, 1):
        recs.append({"config": cfg, col: asci_at_bins(wd, args.bins)})
        if i % 25 == 0 or i == len(rows):
            print(f"  {i}/{len(rows)}")

    out = pd.DataFrame(recs)
    good = out[col].notna().sum()
    if good == 0:
        sys.exit(f"ERROR: every {col} value is NaN. Refusing to write a column "
                 f"of nothing; check the beam files and the bin range.")
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    out.to_csv(args.out, index=False)
    print(f"\nwrote {args.out}: {good} of {len(out)} configs have a value")
    v = out[col].dropna()
    sat = 100.0 * (v >= 99.0).mean()
    print(f"{col}: mean {v.mean():.2f}%, sd {v.std():.2f}, {sat:.0f}% saturated")
    if sat > 50:
        print("  [warn] still mostly saturated; a finer bin may be needed")


if __name__ == "__main__":
    main()
