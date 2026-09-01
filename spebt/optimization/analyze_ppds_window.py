#!/usr/bin/env python3
"""
Does an FWHM-windowed PPDS carry signal where the unwindowed one goes flat?

RY proposed this in Jul 2026. Ring-1 PPDS was adopted as the sensitivity
replacement at rho +0.60 against CNR, but split by aperture band it correlates
0.73 and 0.80 in the two larger bands and -0.03 in the smallest, 0.20 to
0.45 mm, which is exactly where every one of our best designs sits. So it
separates good regions from bad and says nothing inside the good one. By Aug it
had decayed to +0.21 over the full archive.

This is the last objective we have not re-examined, and every other one needed
it: sensitivity was removed, ASCI was windowed, FWHM was reweighted, MPXI was
redefined and its sign reversed.

Method, deliberately the same as the ASCI window sweep that produced the
0.45 mm threshold: recompute PPDS counting only beams narrower than each
candidate threshold, and correlate against reconstructed CNR. A threshold is
worth adopting only if it beats the unwindowed metric BOTH overall and inside
the small-aperture band, since that band is the one that matters.

Usage:
  python analyze_ppds_window.py --results_csv results/results_summary_mobo.csv
  python analyze_ppds_window.py --results_csv ... --limit 60
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

import compute_metrics as cm

THRESHOLDS = (0.40, 0.45, 0.50, 0.60, 0.80, 1.00)
OUTCOME = "cnr_sector_mean"
# The band containing every top design. Ring-1 PPDS is uninformative here, which
# is the whole reason for this test.
SMALL_APERTURE = (0.20, 0.45)


def main():
    ap = argparse.ArgumentParser(description="Sweep an FWHM window for PPDS")
    ap.add_argument("--results_csv", required=True)
    ap.add_argument("--limit", type=int, default=None,
                    help="Only process the first N configs (each takes ~20 s)")
    ap.add_argument("--out", default=None, help="Write per-config values here")
    args = ap.parse_args()

    df = pd.read_csv(args.results_csv)
    need = ["work_dir", "n_det_ring1", OUTCOME, "aperture_diam_mm"]
    missing = [c for c in need if c not in df.columns]
    if missing:
        print(f"ERROR: results CSV lacks {missing}")
        sys.exit(1)
    df = df.dropna(subset=need)

    # Only configs whose PPDF files still exist can be recomputed. The vscratch
    # purge stripped the raw files from the older lhs4d_* designs, leaving just
    # their cnr/ subdirectory, and head(limit) picks exactly those because they
    # are first in the CSV. Sampling the wrong subset produced a full table of
    # NaN that looked like a result.
    import glob as _glob
    has_ppdf = df.work_dir.astype(str).map(
        lambda w: bool(_glob.glob(os.path.join(w, "position_*_ppdfs_t8_*.hdf5"))))
    n_before = len(df)
    df = df[has_ppdf]
    print(f"{len(df)} of {n_before} configs still have their PPDF files "
          f"({n_before - len(df)} stripped by the scratch purge)")
    if df.empty:
        print("\nERROR: no config has the raw files this test needs. Nothing to do.")
        sys.exit(1)

    if args.limit and len(df) > args.limit:
        # Keep the small-aperture band intact: it is the band the test turns on,
        # and a plain head() or random sample can leave too few of them.
        small = df[(df.aperture_diam_mm >= SMALL_APERTURE[0])
                   & (df.aperture_diam_mm <= SMALL_APERTURE[1])]
        rest = df[~df.index.isin(small.index)]
        keep_small = min(len(small), max(args.limit // 2, 1))
        df = pd.concat([small.sample(keep_small, random_state=0),
                        rest.sample(min(len(rest), args.limit - keep_small),
                                    random_state=0)])
        print(f"sampled {len(df)}: {keep_small} in the 0.20-0.45 mm aperture band, "
              f"{len(df) - keep_small} outside it")

    print(f"recomputing PPDS at {len(THRESHOLDS)} thresholds for {len(df)} configs")
    print("(about 20 s per config per threshold, so this is not quick)\n", flush=True)

    rows = []
    for i, (_, r) in enumerate(df.iterrows(), 1):
        wd = str(r["work_dir"])
        if not os.path.isdir(wd):
            continue
        rec = {"config": r.get("config"), OUTCOME: r[OUTCOME],
               "aperture_diam_mm": r["aperture_diam_mm"]}
        base = cm.compute_ppds_per_ring(wd, int(r["n_det_ring1"]))
        rec["ppds_ring1"] = float(base[0]) if base is not None and len(base) else np.nan
        for t in THRESHOLDS:
            comps = cm.compute_ppds_per_ring(wd, int(r["n_det_ring1"]), fwhm_max=t)
            rec[f"ppds_ring1_w{t}"] = (float(comps[0]) if comps is not None
                                       and len(comps) else np.nan)
        rows.append(rec)
        if i % 10 == 0:
            print(f"  {i}/{len(df)}", flush=True)

    if not rows:
        print("ERROR: no readable work_dirs")
        sys.exit(1)
    out = pd.DataFrame(rows)
    if args.out:
        out.to_csv(args.out, index=False)
        print(f"\nwrote {args.out}")

    if out["ppds_ring1"].notna().sum() < 10:
        print(f"\nERROR: only {int(out['ppds_ring1'].notna().sum())} configs produced "
              f"a PPDS value.\nWithout usable inputs the correlations below would all "
              f"be NaN, which reads\nlike a result and is not one. Stopping.")
        sys.exit(1)

    small = out[(out.aperture_diam_mm >= SMALL_APERTURE[0])
                & (out.aperture_diam_mm <= SMALL_APERTURE[1])]
    print("\n" + "=" * 68)
    print("PPDS RING 1 vs CNR, BY FWHM WINDOW")
    print("=" * 68)
    print(f"\n{'window (mm)':<14}{'rho, all':>12}{'rho, small ap':>16}{'n small':>10}")
    print("-" * 52)

    def rho(frame, col):
        sub = frame[[col, OUTCOME]].dropna()
        return sub[col].corr(sub[OUTCOME], method="spearman") if len(sub) >= 10 else np.nan

    base_all, base_small = rho(out, "ppds_ring1"), rho(small, "ppds_ring1")
    print(f"{'unwindowed':<14}{base_all:>12.2f}{base_small:>16.2f}{len(small):>10}")
    best = None
    for t in THRESHOLDS:
        c = f"ppds_ring1_w{t}"
        a, s = rho(out, c), rho(small, c)
        print(f"{t:<14.2f}{a:>12.2f}{s:>16.2f}")
        if np.isfinite(s) and (best is None or s > best[1]):
            best = (t, s, a)

    print("""
How to read this. The 'small ap' column is the one that decides it: every top
design sits in the 0.20 to 0.45 mm aperture band, and unwindowed ring-1 PPDS is
uninformative there. A window is worth adopting only if it beats the unwindowed
metric in that column, not merely overall.""")
    if best and np.isfinite(base_small):
        t, s, a = best
        if s > base_small + 0.15:
            print(f"\n-> The {t:.2f} mm window lifts the small-aperture correlation "
                  f"from {base_small:+.2f} to {s:+.2f}. Worth adopting, and worth "
                  f"testing\n   the same way the ASCI window was: confirm it peaks "
                  f"rather than\n   rising monotonically, which would just mean the "
                  f"threshold is arbitrary.")
        else:
            print(f"\n-> No window helps materially. Best is {t:.2f} mm at "
                  f"{s:+.2f} against {base_small:+.2f} unwindowed.\n   Ring-1 PPDS "
                  f"appears to carry no signal inside the region we care about, "
                  f"windowed\n   or not, which is an argument for dropping it "
                  f"rather than redefining it.")


if __name__ == "__main__":
    main()
