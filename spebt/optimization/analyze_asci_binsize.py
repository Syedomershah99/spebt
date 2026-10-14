#!/usr/bin/env python3
"""Does ASCI saturate at 1-degree angular bins?

ASCI counts the fraction of (FOV pixel, angle bin) cells that any beam covers.
It is computed at 360 bins, i.e. one degree, and in that form it sits near
100% for almost every feasible design, which makes it useless for ranking:
an objective that cannot tell designs apart cannot steer a search.

Dr. Yao's suggestion (13 Oct 2026) is that the bin is simply too coarse. At
one degree a single beam blankets a whole bin, so coverage saturates long
before the angular sampling is actually complete. Finer bins should separate
designs again.

This recomputes ASCI at several bin widths from the SURVIVING beam property
and mask files, so it costs no ray tracing. For each width it reports how
saturated the metric is and how well it tracks reconstructed CNR, both overall
and inside the 0.20-0.45 mm aperture band where every top design sits. That
band is the one that matters: a metric only has to discriminate where the
optimizer actually converges.

Usage:
  python analyze_asci_binsize.py --limit 40
  python analyze_asci_binsize.py --bins 360 720 1800 3600
"""
import argparse
import glob
import os
import sys

import h5py
import numpy as np
import pandas as pd

FOV_NPIX = (200, 200)
N_FOV = FOV_NPIX[0] * FOV_NPIX[1]
SMALL_APERTURE = (0.20, 0.45)
SATURATED = 99.0          # percent; at or above this a design is indistinguishable


def configs_with_beam_files(df, results_dir):
    """Rows whose beam property AND mask files both survive the scratch purge."""
    keep = []
    for _, r in df.iterrows():
        wd = r.get("work_dir")
        if not isinstance(wd, str) or not os.path.isdir(wd):
            wd = os.path.join(results_dir, str(r["config"]))
        if not os.path.isdir(wd):
            continue
        props = sorted(glob.glob(os.path.join(wd, "beams_properties_configuration_*.hdf5")))
        masks = sorted(glob.glob(os.path.join(wd, "beams_masks_configuration_*.hdf5")))
        if props and masks and len(props) == len(masks):
            keep.append((r["config"], wd, r["aperture_diam_mm"], r["cnr_sector_mean"]))
    return keep


def stratified_sample(rows, limit, seed=0):
    """Half inside the small-aperture band, half outside.

    A plain head() or random draw leaves almost nothing in the band, and the
    band is the only place the answer matters.
    """
    if limit is None or len(rows) <= limit:
        return rows
    lo, hi = SMALL_APERTURE
    inside = [r for r in rows if lo <= r[2] <= hi]
    outside = [r for r in rows if not (lo <= r[2] <= hi)]
    rng = np.random.default_rng(seed)
    n_in = min(len(inside), max(limit // 2, 1))
    n_out = min(len(outside), limit - n_in)
    pick = ([inside[i] for i in rng.choice(len(inside), n_in, replace=False)] +
            [outside[i] for i in rng.choice(len(outside), n_out, replace=False)])
    return pick


def asci_at_bins(work_dir, n_bins):
    """Recompute ASCI for one config at a given angular bin count.

    Accumulates a boolean coverage grid rather than the integer histogram the
    production script writes: we only need whether a cell was touched, and at
    3600 bins the integer version is four times the memory for no extra
    information.
    """
    props = sorted(glob.glob(os.path.join(work_dir, "beams_properties_configuration_*.hdf5")))
    masks = sorted(glob.glob(os.path.join(work_dir, "beams_masks_configuration_*.hdf5")))
    if not props or len(props) != len(masks):
        return np.nan

    covered = np.zeros((N_FOV, n_bins), dtype=bool)
    edges = np.linspace(0.0, 2 * np.pi, n_bins + 1)   # beam angles are 0..2pi

    for pf, mf in zip(props, masks):
        try:
            with h5py.File(pf, "r") as f:
                bp = f["beam_properties"][:]
            with h5py.File(mf, "r") as f:
                bm = f["beam_mask"][:]
        except Exception as e:
            print(f"    [warn] unreadable beam file in {work_dir}: {e}")
            return np.nan

        angles = bp[:, 3].astype(np.float64)
        det_ids = bp[:, 1].astype(np.int64)
        beam_ids = bp[:, 2].astype(np.int64)
        ok = np.isfinite(angles)
        if not ok.any():
            continue

        # Fail loudly if the data does not fit the binning range. Binning
        # 0..2pi angles over -pi..pi silently made half the bins unreachable
        # and produced ASCI = 50.00% with zero variance across every design:
        # a clean-looking number that was pure artifact. np.clip would hide
        # the same mistake again, so check before clipping.
        # Angles are stored as float32, so one that is exactly 2*pi reads back
        # as 6.2831855 -- about 1.7e-07 above the float64 2*pi. Tolerate that
        # much; a real binning mistake (0..2pi data against -pi..pi bins) is
        # wrong by radians, not by a ten-millionth of one.
        TOL = 1e-6
        amin, amax = angles[ok].min(), angles[ok].max()
        if amin < edges[0] - TOL or amax > edges[-1] + TOL:
            raise ValueError(
                f"beam angles span [{amin!r}, {amax!r}] but bins cover "
                f"[{edges[0]!r}, {edges[-1]!r}] (overshoot low "
                f"{edges[0] - amin:.3e}, high {amax - edges[-1]:.3e}) in "
                f"{work_dir}. Fix the bin range rather than clipping.")

        bins = np.clip(np.digitize(angles, edges) - 1, 0, n_bins - 1)

        for k in np.flatnonzero(ok):
            d, b, ab = det_ids[k], beam_ids[k], bins[k]
            if d < 0 or d >= bm.shape[0]:
                continue
            covered[bm[d] == b, ab] = True

    return 100.0 * covered.sum() / covered.size


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results_csv", default="results/results_summary_mobo.csv")
    ap.add_argument("--results_dir", default="results")
    ap.add_argument("--bins", type=int, nargs="+", default=[360, 720, 1800, 3600])
    ap.add_argument("--limit", type=int, default=60)
    ap.add_argument("--out", default="results/asci_binsize.csv")
    args = ap.parse_args()

    from scipy.stats import spearmanr

    # Read every archive, not just the main one. The head-to-head arms keep
    # their own results CSV, and most surviving beam files belong to configs
    # recorded there: reading the main archive alone found 14 usable designs,
    # all inside the aperture band, so there was no out-of-band comparison at
    # all. The same widening that check_unmerged_results.py already does.
    frames = []
    for f in [args.results_csv] + sorted(glob.glob(
            os.path.join(os.path.dirname(os.path.dirname(
                os.path.abspath(args.results_csv))),
                "results_h2h_*", os.path.basename(args.results_csv)))):
        if not os.path.exists(f):
            continue
        try:
            frames.append(pd.read_csv(f))
        except Exception as e:
            print(f"  [warn] unreadable archive {f}: {e}")
    if not frames:
        sys.exit(f"ERROR: no readable archive at {args.results_csv}")
    df = pd.concat(frames, ignore_index=True)
    df = df.dropna(subset=["cnr_sector_mean", "aperture_diam_mm"])
    df = df.drop_duplicates(subset=["config"], keep="first")
    print(f"{len(frames)} archive(s), {len(df)} measured configs")
    rows = configs_with_beam_files(df, args.results_dir)
    print(f"{len(rows)} of them still have their beam files")
    if len(rows) < 8:
        sys.exit("ERROR: too few configs with surviving beam files to say anything.")

    rows = stratified_sample(rows, args.limit)
    lo, hi = SMALL_APERTURE
    n_in = sum(1 for r in rows if lo <= r[2] <= hi)
    print(f"sampled {len(rows)}: {n_in} inside the {lo}-{hi} mm aperture band, "
          f"{len(rows) - n_in} outside")
    print(f"bin widths to test: {[round(360/b, 3) for b in args.bins]} degrees\n")

    recs = []
    for i, (cfg, wd, ap_d, cnr) in enumerate(rows, 1):
        rec = {"config": cfg, "aperture_diam_mm": ap_d, "cnr_sector_mean": cnr}
        for nb in args.bins:
            rec[f"asci_{nb}"] = asci_at_bins(wd, nb)
        recs.append(rec)
        if i % 10 == 0 or i == len(rows):
            print(f"  {i}/{len(rows)}")

    out = pd.DataFrame(recs)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    out.to_csv(args.out, index=False)
    print(f"\nwrote {args.out}\n")

    band = out[out.aperture_diam_mm.between(lo, hi)]
    print("=" * 78)
    print("ASCI BIN WIDTH: SATURATION AND USEFULNESS")
    print("=" * 78)
    print(f"\n{'bin (deg)':>10} {'mean %':>9} {'sd':>8} {'saturated':>11} "
          f"{'rho all':>9} {'rho band':>9}")
    for nb in args.bins:
        col = f"asci_{nb}"
        v = out[col].dropna()
        if len(v) < 5:
            print(f"{360/nb:10.3f}   too few values ({len(v)})")
            continue
        sat = 100.0 * (v >= SATURATED).mean()
        r_all = spearmanr(out[col], out["cnr_sector_mean"],
                          nan_policy="omit").correlation
        r_band = (spearmanr(band[col], band["cnr_sector_mean"],
                            nan_policy="omit").correlation
                  if len(band.dropna(subset=[col])) >= 5 else np.nan)
        print(f"{360/nb:10.3f} {v.mean():9.2f} {v.std():8.3f} "
              f"{sat:10.0f}% {r_all:+9.2f} {r_band:+9.2f}")

    print("\nHow to read this. 'saturated' is the share of designs at or above "
          f"{SATURATED}%,\nwhich are indistinguishable from each other and so "
          "invisible to the optimizer.\n'rho band' is the one that decides it: "
          "a bin width is an improvement only if\nit discriminates inside the "
          "aperture range our best designs occupy.")


if __name__ == "__main__":
    main()
