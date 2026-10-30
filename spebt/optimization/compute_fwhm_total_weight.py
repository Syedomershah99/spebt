#!/usr/bin/env python3
"""Weighted FWHM per design with each strip weighted by its whole PPDF (Dr. Yao, R10-rY,
29 Oct 2026: "I am not clear about S_b, the mean PPDF of strip b. Check out the wFWHM
equation"), beside the searches' weight, the strip's mean PPDF.

beam_properties col 7 is the strip's mean PPDF over its pixels and col 9 its pixel count
(beam_property_io.stack_beams_properties), so the strip's whole PPDF, sum_k p_ibk, is
col 7 * col 9. Strips as compute_fwhm_and_asci counts them (finite FWHM > 0); the mean
weight reproduces the archive's fwhm_weighted_mean. Reads beam_properties only.

  python compute_fwhm_total_weight.py --root <optimization dir> --out fwhm_total_weight.csv
"""
import argparse
import csv
import glob
import os

import h5py
import numpy as np

ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
ap.add_argument("--root", default=os.path.dirname(os.path.abspath(__file__)))
ap.add_argument("--out", default="fwhm_total_weight.csv")
args = ap.parse_args()
out = []
for d in sorted(glob.glob(f"{args.root}/results*/*/")):
    props = sorted(glob.glob(d + "beams_properties_configuration_*.hdf5"))
    if not props:
        continue
    fw, sm, npx, bad = [], [], [], False
    for pf in props:
        try:
            with h5py.File(pf, "r") as f:
                bp = f["beam_properties"][:]
        except OSError:
            bad = True
            break
        ok = np.isfinite(bp[:, 4]) & (bp[:, 4] > 0)
        fw.append(bp[ok, 4]); sm.append(bp[ok, 7]); npx.append(bp[ok, 9])
    if bad or not fw:
        continue
    fw, sm, npx = (np.concatenate(v).astype(np.float64) for v in (fw, sm, npx))
    sm[~np.isfinite(sm)] = 0.0
    npx[~np.isfinite(npx)] = 0.0
    tot = sm * npx
    # FWHM exactly 64 mm is get_beam_width's sampling-line length: the width was not found
    ok = fw < 63.9
    out.append(dict(arm=os.path.basename(os.path.dirname(d.rstrip("/"))), config=os.path.basename(d.rstrip("/")),
                    n_strips=fw.size, n64=int((~ok).sum()),
                    fwhm_wmean_chk=fw @ sm / sm.sum() if sm.sum() > 0 else np.nan,
                    fwhm_wtotal=fw @ tot / tot.sum() if tot.sum() > 0 else np.nan,
                    fwhm_mean_no64=fw[ok].mean() if ok.any() else np.nan,
                    fwhm_wmean_no64=fw[ok] @ sm[ok] / sm[ok].sum() if sm[ok].sum() > 0 else np.nan,
                    npix_median=float(np.median(npx))))
with open(args.out, "w", newline="") as f:
    wr = csv.DictWriter(f, fieldnames=list(out[0])); wr.writeheader(); wr.writerows(out)
print("designs", len(out))
