#!/usr/bin/env python3
"""How much of each metric's ranking in the aperture band is multiplexing (R10, 28 Oct 2026).

In the band (apertures 0.20 to 0.45 mm, 796 of the 1020 scored designs) MPXI is tied to
CNR most strongly (rho -0.70), and ASCI rises with it (rho 0.91). This prints, per metric,
the raw Spearman rho with CNR, the rank partial correlation with MPXI held, and the rho
within five MPXI quantile bins; the ASCI bins at MPXI 1.8 to 2.1 (slide 11); and the FWHM
filter of Dr. Yao's R9-rY comment from archive_backup/fwhm_filter.csv. Pool as on slides 9
to 11: the main archive first, then the ten h2h files, mobo_NNNN names qualified by search.

  python analyze_band_partials.py
"""
import glob
import os

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

A = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "archive_backup")
C, MP = "cnr_sector_mean", "mpxi_windowed_active_mean"

frames = []
for f in [f"{A}/results_summary_mobo.csv"] + sorted(glob.glob(f"{A}/h2h/*.csv")):
    d = pd.read_csv(f)
    d["arm"] = "main" if "h2h" not in f else os.path.basename(f)[len("results_h2h_"):-4]
    frames.append(d)
a = pd.concat(frames, ignore_index=True)
a["key"] = [c if not str(c).startswith("mobo") else f"{r}:{c}" for c, r in zip(a.config, a.arm)]
u = a[a[C].notna()].drop_duplicates("key", keep="first")
band = u[(u.aperture_diam_mm >= 0.20) & (u.aperture_diam_mm <= 0.45)]

x = pd.read_csv(f"{A}/fwhm_filter.csv")
arm = x.arm.str.replace("results_h2h_", "", regex=False).replace({"results": "main"})
x["key"] = [c if not str(c).startswith("mobo") else f"{r}:{c}" for c, r in zip(x.config, arm)]
sub = band.merge(x[["key", "fwhm_win_mean", "fwhm_win_wmean", "fwhm_win_nofloor_mean"]], on="key")


def rho(df, m, y=C):
    d = df[[m, y]].dropna()
    return spearmanr(d[m], d[y]).correlation


def partial(df, m):
    """Rank partial correlation of m and CNR, MPXI held."""
    d = df[[m, C, MP]].dropna().rank()
    res = lambda v: v - np.polyval(np.polyfit(d[MP], v, 1), d[MP])
    return np.corrcoef(res(d[m]), res(d[C]))[0, 1]


def within(df, m, nb=5):
    d = df[[m, C, MP]].dropna().copy()
    d["q"] = pd.qcut(d[MP], nb, labels=False, duplicates="drop")
    rs = [(spearmanr(g[m], g[C]).correlation, len(g)) for _, g in d.groupby("q")]
    return np.average([r for r, _ in rs], weights=[n for _, n in rs])


print(f"{len(u)} scored designs, {len(band)} in the band; rho(MPXI, CNR) {rho(band, MP):+.2f}, "
      f"rho(ASCI, MPXI) {rho(band, 'asci_pct_fwhm0p45', MP):+.2f}")
print(f"{'metric':26s} {'raw':>6s} {'MPXI held':>10s} {'5 bins':>7s}")
for df, cols in ((band, ["fwhm_weighted_mean", "asci_pct_fwhm0p45", "ppds_ring1"]),
                 (sub, ["fwhm_weighted_mean", "fwhm_win_mean", "fwhm_win_wmean", "fwhm_win_nofloor_mean"])):
    print(f"-- {len(df)} designs")
    for m in cols:
        print(f"{m:26s} {rho(df, m):+6.2f} {partial(df, m):+10.2f} {within(df, m):+7.2f}")
s = band[(band[MP] >= 1.8) & (band[MP] < 2.1)]
bins = pd.cut(s.asci_pct_fwhm0p45, [35, 40, 45, 50, 55, 60, 65], right=False)
print(f"\nMPXI 1.8 to 2.1: {len(s)} designs, rho(ASCI, CNR) {rho(s, 'asci_pct_fwhm0p45'):+.2f}")
print(s.groupby(bins, observed=True)[C].agg(["count", "mean"]).round(2).to_string())
