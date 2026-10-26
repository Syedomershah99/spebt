#!/usr/bin/env python3
"""
Reclaim disk by deleting simulation output for designs we no longer need it for.

Aug 26 2026: /vscratch filled to 100% and every running job died. Pipeline jobs
could not write their output files, so they failed with no logs; controllers
could not write the manifest or their own stderr, so they died with no
traceback. Nothing was wrong with the code. The cluster simply had no bytes
left.

The arithmetic that caused it:

    PPDF files (position_*)        7,959 MB per design   89%
    beam masks and properties        998 MB per design
    ASCI histograms                  110 MB per design
    the metrics we actually keep        77 KB per CAMPAIGN

Six head-to-head arms at 80 designs each is roughly 4.7 TB to produce a few
hundred kilobytes of numbers.

WHY THIS IS A TOOL AND NOT AN AUTOMATIC PIPELINE STEP
-----------------------------------------------------
The obvious fix is for run_sai_pipeline.sh to delete PPDFs once metrics are
computed. That would have been wrong. compute_cnr.py reads PPDFs back out of
the work directory, which is what made the 5-seed repeat study possible, and
that study is where every error bar in the design result came from. Deleting on
write would have silently removed the ability to measure our own uncertainty.

So pruning is deliberate: keep the designs worth re-measuring, drop the rest.

WHAT YOU LOSE, AND WHY IT MAY MATTER MORE THAN THE DISK
-------------------------------------------------------
position_*_ppdfs_t8_*.hdf5 is read by three things: sensitivity (retired),
PPDS in any form, and the CNR forward projection. --keep_top covers CNR for
the designs worth re-measuring. It does NOT cover PPDS.

Ring-1 PPDS is still an objective, and redefining it is an open question: it
goes flat inside the small-aperture region where the best designs sit, and the
proposal on record is to test an FWHM-windowed version the way the ASCI window
was tested. That needs the raw PPDFs. Every metric in this project has been
redefined at least once (sensitivity removed, ASCI windowed, FWHM reweighted,
MPXI redefined and its sign reversed), so treat "we might redefine PPDS" as
likely rather than hypothetical.

Regenerating one design's PPDFs is a full pipeline run, about 40 minutes. At
250 designs that is roughly a week of compute to undo.

Prune when the disk actually demands it, not pre-emptively.

Usage:
  python prune_ppdfs.py --results_dir results --keep_top 10           # dry run
  python prune_ppdfs.py --results_dir results --keep_top 10 --delete
  python prune_ppdfs.py --results_dir results --keep_top 5 --mpxi_window 1.5:2.5 --delete

--mpxi_window also keeps the top --keep_top designs whose MPXI lies in the
window, for a search that steers MPXI into it: the designs worth re-measuring
are the best in-window ones, which need not be the best overall.
"""
import argparse
import os
import shutil
import sys
import time

import pandas as pd

# Deleted when a design is pruned. PPDFs dominate; masks and histograms are
# kept only because compute_mpxi_variants and analyze_asci_window re-read them,
# and they are a tenth the size.
PRUNE_GLOBS = ("position_",)
# Deleted as well with --masks: about 1.1 GB per design, which over a campaign
# of thousands of designs outgrows a 5 TB scratch quota on its own.
MASK_GLOBS = ("beams_masks_", "beams_properties_", "asci_histogram_")

RESULTS_NAME = "results_summary_mobo.csv"
OUTCOME = "cnr_sector_mean"
MPXI = "mpxi_windowed_active_mean"


def dir_size(path):
    total = 0
    for root, _, files in os.walk(path):
        for f in files:
            try:
                total += os.path.getsize(os.path.join(root, f))
            except OSError:
                pass
    return total


def prunable_bytes(work_dir, globs=PRUNE_GLOBS):
    """Bytes in files this tool would delete."""
    total = 0
    try:
        for f in os.listdir(work_dir):
            if f.startswith(globs):
                try:
                    total += os.path.getsize(os.path.join(work_dir, f))
                except OSError:
                    pass
    except OSError:
        pass
    return total


def newest_mtime(work_dir):
    """Most recent modification time of any file in the work directory."""
    t = 0.0
    for root, _, files in os.walk(work_dir):
        for f in files:
            try:
                t = max(t, os.path.getmtime(os.path.join(root, f)))
            except OSError:
                pass
    return t


def main():
    ap = argparse.ArgumentParser(description="Prune PPDF files from old designs")
    ap.add_argument("--results_dir", required=True,
                    help="A campaign results directory holding config work dirs")
    ap.add_argument("--keep_top", type=int, default=10,
                    help="Keep PPDFs for the N best designs by CNR, so they can "
                         "still be re-measured for error bars (default 10)")
    ap.add_argument("--delete", action="store_true",
                    help="Actually delete. Without this it only reports.")
    ap.add_argument("--mpxi_window", default=None, metavar="LO:HI",
                    help="Also keep the top --keep_top designs with "
                         "mpxi_windowed_active_mean in [LO, HI], and the top "
                         "--keep_top such designs chosen by the optimizer")
    ap.add_argument("--masks", action="store_true",
                    default=os.environ.get("MOBO_EXECUTOR") == "local",
                    help="Also delete the beam masks, properties and ASCI histograms "
                         "of pruned designs. On by default in a node campaign "
                         "(MOBO_EXECUTOR=local, run_node_searches.sh)")
    ap.add_argument("--min_age_hours", type=float, default=0.5,
                    help="Leave alone any design directory with a file modified "
                         "this recently: a pipeline may be running in it (half an "
                         "hour since 26 Oct 2026: no launcher re-runs a measured design "
                         "any more, and 2 h of finished designs filled 1.4 TB of scratch)")
    args = ap.parse_args()

    res_path = os.path.join(args.results_dir, RESULTS_NAME)
    if not os.path.exists(res_path):
        print(f"ERROR: no {RESULTS_NAME} in {args.results_dir}.")
        print("Refusing to prune a directory whose metrics are not recorded: the")
        print("CSV is the only thing that survives pruning, so without it the")
        print("simulation output is all there is.")
        sys.exit(1)

    df = pd.read_csv(res_path)
    if OUTCOME not in df.columns:
        print(f"ERROR: {res_path} has no {OUTCOME} column")
        sys.exit(1)

    work_dirs = [d for d in sorted(os.listdir(args.results_dir))
                 if os.path.isdir(os.path.join(args.results_dir, d))
                 and (d.startswith("mobo_") or d.startswith("lhs6d_")
                      or d.startswith("ring2_"))]

    # Rank each design once, and only designs whose files are here: a warm
    # start's copied rows (measured elsewhere) and repeated rows would otherwise
    # take keep slots and protect nothing.
    scored = df.dropna(subset=[OUTCOME]).drop_duplicates("config")
    scored = scored[scored["config"].astype(str).isin(work_dirs)]
    keep = set(scored.nlargest(args.keep_top, OUTCOME)["config"].astype(str))
    if args.mpxi_window:
        lo, hi = (float(v) for v in args.mpxi_window.split(":"))
        inside = scored[(scored[MPXI] >= lo) & (scored[MPXI] <= hi)]
        keep |= set(inside.nlargest(args.keep_top, OUTCOME)["config"].astype(str))
        # The re-measurement candidates are the best in-window designs the
        # optimizer chose, which starting designs could otherwise crowd out.
        chosen = inside[inside["config"].astype(str).str.startswith("mobo_")]
        keep |= set(chosen.nlargest(args.keep_top, OUTCOME)["config"].astype(str))
    print(f"Keeping PPDFs for {len(keep)} designs: the top {args.keep_top} by {OUTCOME}"
          + (f", and the top {args.keep_top} with MPXI in [{args.mpxi_window}], overall and "
             f"among the optimizer's" if args.mpxi_window else "")
          + (". Masks of pruned designs are deleted too" if args.masks else "") + ".\n")

    # Only designs whose CNR is recorded are finished with their PPDFs. A design
    # with no row is still running, and one with a row but no CNR may be in its
    # CNR step, which reads the PPDFs after the metrics row is written. A design
    # re-run after a restart has a CNR from its first run while its pipeline
    # rewrites its files, so recent activity also protects a directory.
    recorded = set(df.dropna(subset=[OUTCOME])["config"].astype(str))
    cutoff = time.time() - 3600 * args.min_age_hours
    globs = PRUNE_GLOBS + (MASK_GLOBS if args.masks else ())
    freed = kept_bytes = 0
    n_pruned = n_running = 0
    for d in work_dirs:
        wd = os.path.join(args.results_dir, d)
        size = prunable_bytes(wd, globs)
        if d not in recorded or newest_mtime(wd) > cutoff:
            n_running += size > 0
            continue
        if d in keep:
            kept_bytes += prunable_bytes(wd)
            continue
        if size == 0:
            continue
        n_pruned += 1
        freed += size
        if args.delete:
            for f in os.listdir(wd):
                if f.startswith(globs):
                    try:
                        os.remove(os.path.join(wd, f))
                    except OSError as e:
                        print(f"  [warn] {f}: {e}")

    verb = "Deleted" if args.delete else "Would delete"
    print(f"{verb} {'PPDFs and masks' if args.masks else 'PPDFs'} from {n_pruned} of "
          f"{len(work_dirs)} design directories"
          + (f" ({n_running} running, recently active or without a CNR yet, left alone)"
             if n_running else ""))
    print(f"  reclaimed: {freed / 1e12:.2f} TB ({freed / 1e9:.0f} GB)")
    print(f"  retained for the top designs: {kept_bytes / 1e9:.0f} GB")
    print(f"\nMetrics are untouched. {res_path} still holds every measurement.")
    if not args.delete:
        print("\nDry run. Re-run with --delete to actually remove them.")
    else:
        print("\nRe-measuring a pruned design now requires regenerating its PPDFs,")
        print("which is a full pipeline run of roughly 40 minutes.")


if __name__ == "__main__":
    main()
