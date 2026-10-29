#!/usr/bin/env python3
"""FWHM with the strip filter of windowed ASCI and MPXI, per design with beam files.

Dr. Yao (R9-rY, 28 Oct 2026) suggested the same FWHM filter for the FWHM objective
that ASCI and MPXI use: count only strips with FWHM <= 0.45 mm. The strip set here is
analyze_asci_window's: angle-valid strips above 1% of the layout's peak sensitivity,
then FWHM <= 0.45 mm. The archive's two FWHM columns are recomputed beside it as a
check (compute_metrics.compute_fwhm_and_asci). Reads beam_properties only; a design
with an unreadable layout file is left out whole.

  python compute_fwhm_filter.py --root /projects/academic/rutaoyao/Omer/spebt/spebt/spebt/optimization \\
      --out fwhm_filter.csv     (28 Oct 2026, CCR job 26644964: 330 designs, in archive_backup/)
"""
import argparse
import csv
import glob
import os

import h5py
import numpy as np

ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
ap.add_argument("--root", default=os.path.dirname(os.path.abspath(__file__)))
ap.add_argument("--out", default="fwhm_filter.csv")
ap.add_argument("--fwhm_max", type=float, default=0.45)
args = ap.parse_args()
T, FLOOR = args.fwhm_max, 0.01

out = []
for d in sorted(glob.glob(f"{args.root}/results*/*/")):
    props = sorted(glob.glob(d + "beams_properties_configuration_*.hdf5"))
    if not props:
        continue
    arm = os.path.basename(os.path.dirname(d.rstrip("/")))
    cfg = os.path.basename(d.rstrip("/"))
    fa, sa, ff, sf, bad = [], [], [], [], []
    for pf in props:
        try:
            with h5py.File(pf, "r") as f:
                bp = f["beam_properties"][:]
        except OSError:
            bad.append(os.path.basename(pf))
            continue
        if bp.shape[0] == 0:
            continue
        ang, fw, s = (bp[:, k].astype(np.float64) for k in (3, 4, 7))
        ok = np.isfinite(fw) & (fw > 0)              # compute_fwhm_and_asci's set
        w = s[ok].copy(); w[~np.isfinite(w)] = 0.0
        fa.append(fw[ok]); sa.append(w)
        keep = ~np.isnan(ang)                         # analyze_asci_window's set
        if keep.sum() == 0:
            continue
        smax = s[keep].max() if np.isfinite(s[keep]).any() else 0.0
        if smax > 0:
            keep &= s > smax * FLOOR
        keep &= np.isfinite(fw) & (fw > 0)
        ff.append(fw[keep]); sf.append(s[keep])
    if bad:
        print(f"skip {arm}/{cfg}: unreadable {bad}")
        continue
    if not fa:
        continue
    fa, sa = np.concatenate(fa), np.concatenate(sa)
    ff = np.concatenate(ff) if ff else np.array([]); sf = np.concatenate(sf) if sf else np.array([])
    win, nf = ff <= T, fa <= T
    out.append(dict(
        arm=arm, config=cfg, n_layouts=len(props), n_all=fa.size,
        fwhm_mean_chk=fa.mean(), fwhm_wmean_chk=(fa @ sa / sa.sum()) if sa.sum() > 0 else np.nan,
        n_floor=ff.size, n_win=int(win.sum()),
        fwhm_floor_mean=ff.mean() if ff.size else np.nan,
        fwhm_win_mean=ff[win].mean() if win.any() else np.nan,
        fwhm_win_wmean=(ff[win] @ sf[win] / sf[win].sum()) if win.any() and sf[win].sum() > 0 else np.nan,
        fwhm_win_nofloor_mean=fa[nf].mean() if nf.any() else np.nan,
    ))
with open(args.out, "w", newline="") as f:
    wr = csv.DictWriter(f, fieldnames=list(out[0]))
    wr.writeheader(); wr.writerows(out)
print("designs", len(out))
