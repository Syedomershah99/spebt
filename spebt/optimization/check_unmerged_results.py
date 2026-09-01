#!/usr/bin/env python3
"""Find finished batch evaluations that were never merged into the archive.

Seed sweeps (`make_ring2_seeds.py`, `make_d2d3_seed_list.py`,
`make_lhs6d_seeds.py`) write one CSV per array task into a
`results/<name>_seed_out/` directory. Merging those rows into
`results_summary_mobo.csv` is a separate manual step, and nothing ever checked
that it happened.

It did not happen once. The `ring2_000..007` sweep -- the whole point of which
was to test whether n_det_ring2 had headroom above 960 -- finished on 12 Aug
2026 with all 8 configs complete, and sat unread in `results/ring2_seed_out/`
for three weeks. The optimizer never trained on it and nobody saw the answer.

Run this after any seed sweep, and before quoting the archive as complete:

    python check_unmerged_results.py
    python check_unmerged_results.py --results_dir results --out_csv results/results_summary_mobo.csv

Exit status is 1 when unmerged rows exist, so it can gate a submit script.
"""
import argparse
import glob
import os
import sys

import pandas as pd


def batch_output_dirs(results_dir: str) -> list:
    """Directories holding per-task batch CSVs, oldest name order."""
    return sorted(
        d for d in glob.glob(os.path.join(results_dir, "*_out"))
        if os.path.isdir(d) and glob.glob(os.path.join(d, "task_*.csv"))
    )


def configs_in_batch_dir(batch_dir: str) -> set:
    """Config names recorded in one batch output directory.

    A task CSV that is empty or truncated (the array job died mid-write) is
    skipped with a warning rather than crashing the check: a partial batch is
    exactly when you most want the rest of the report.
    """
    names = set()
    for f in sorted(glob.glob(os.path.join(batch_dir, "task_*.csv"))):
        try:
            df = pd.read_csv(f)
        except Exception as e:
            print(f"  [warn] unreadable, skipping: {f} ({e})")
            continue
        if "config" not in df.columns:
            print(f"  [warn] no 'config' column, skipping: {f}")
            continue
        names.update(df["config"].dropna().astype(str))
    return names


def find_unmerged(results_dir: str, out_csv: str) -> dict:
    """Map batch dir -> sorted config names absent from the archive."""
    archive = set()
    if os.path.exists(out_csv):
        arc = pd.read_csv(out_csv)
        if "config" in arc.columns:
            archive = set(arc["config"].dropna().astype(str))

    unmerged = {}
    for d in batch_output_dirs(results_dir):
        missing = configs_in_batch_dir(d) - archive
        if missing:
            unmerged[d] = sorted(missing)
    return unmerged


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results_dir", default="results")
    ap.add_argument("--out_csv", default="results/results_summary_mobo.csv")
    args = ap.parse_args()

    unmerged = find_unmerged(args.results_dir, args.out_csv)
    if not unmerged:
        print(f"All batch outputs under {args.results_dir}/ are merged into "
              f"{args.out_csv}.")
        return 0

    total = sum(len(v) for v in unmerged.values())
    print(f"{total} evaluated config(s) are NOT in {args.out_csv}:\n")
    for d, names in unmerged.items():
        print(f"  {d}  ({len(names)})")
        for n in names:
            print(f"      {n}")
    print("\nThese cost compute and produced results nobody has read. Merge them "
          "or record why they are being left out.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
