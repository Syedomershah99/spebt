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

tw_path = f"{A}/fwhm_total_weight.csv"
if os.path.exists(tw_path):
    # Dr. Yao's weight (R10-rY, 29 Oct): each strip by its whole PPDF, col 7 x col 9
    t = pd.read_csv(tw_path)
    arm_t = t.arm.str.replace("results_h2h_", "", regex=False).replace({"results": "main"})
    t["key"] = [c if not str(c).startswith("mobo") else f"{r}:{c}" for c, r in zip(t.config, arm_t)]
    ut = u.merge(t[["key", "fwhm_wtotal", "fwhm_wmean_chk"]], on="key")
    bt = ut[(ut.aperture_diam_mm >= 0.20) & (ut.aperture_diam_mm <= 0.45)]
    print(f"\nstrip-total weight: {len(ut)} designs, {len(bt)} with small apertures")
    print(f"  check mean weight vs archive: max |diff| {np.abs(ut.fwhm_wmean_chk - ut.fwhm_weighted_mean).max():.2e}")
    for m in ("fwhm_weighted_mean", "fwhm_wtotal"):
        print(f"  {m:20s} all {rho(ut, m):+.2f}  small apertures {rho(bt, m):+.2f}  "
              f"MPXI held {partial(bt, m):+.2f}  5 bins {within(bt, m):+.2f}")
    print(f"  rho(total-weighted, mean-weighted) {rho(ut, 'fwhm_wtotal', 'fwhm_weighted_mean'):+.3f}")

pr_path = f"{A}/ppds_radial_recheck.csv"
if os.path.exists(pr_path):
    # The strip-length fix (30 Oct, recheck_ppds_radial.py): ring-I PPDS recomputed from the
    # PPDFs and strip files left on CCR, as the searches computed it and with the fix.
    r = pd.read_csv(pr_path)
    r = r[r.error.isna()]
    ur = u.merge(r[["key", "ppds_ring1_bug", "ppds_ring1_fix"]], on="key")
    br = ur[(ur.aperture_diam_mm >= 0.20) & (ur.aperture_diam_mm <= 0.45)]
    f = ur.ppds_ring1_fix / ur.ppds_ring1_bug
    fs, fl = f[br.index].to_numpy(), f.drop(br.index).to_numpy()
    print(f"\nstrip-length fix: {len(ur)} designs with PPDFs, {len(br)} with small apertures")
    print(f"  as searched vs archive: max relative diff {np.abs(ur.ppds_ring1_bug / ur.ppds_ring1 - 1).max():.1e}")
    print(f"  fixed / as searched: median {f.median():.3f}; small apertures {fs.min():.3f} to {fs.max():.3f}; "
          f"above 0.45 mm {fl.min():.3f} to {fl.max():.3f}")
    for m in ("ppds_ring1", "ppds_ring1_fix"):
        print(f"  {m:16s} all {rho(ur, m):+.2f}  small apertures {rho(br, m):+.2f}  "
              f"MPXI held {partial(br, m):+.2f}  5 bins {within(br, m):+.2f}")
    print(f"  rho(fixed, as searched) {rho(ur, 'ppds_ring1_fix', 'ppds_ring1'):+.4f}, "
          f"small apertures {rho(br, 'ppds_ring1_fix', 'ppds_ring1'):+.4f}")
    # The other 704 designs' PPDFs are purged. At small apertures the factor is nearly
    # constant, so those designs draw from the measured small-aperture factors. Above 0.45 mm it
    # grows with how fully the apertures fill the ring (0.97-0.98 at 0.73-0.76 mm x 255-270),
    # so the 197 unmeasured there are filled four ways. Bands: 2.5-97.5% of 300 draws under
    # each fill, the spread of the draws, not the uncertainty of the true value.
    known = u.merge(ur[["key"]].assign(f=f.to_numpy()), on="key", how="left")["f"].to_numpy()
    ap = u.aperture_diam_mm.to_numpy()
    small = (ap >= 0.20) & (ap <= 0.45)
    mm = ur.aperture_diam_mm.to_numpy()
    p06, p08 = f[(mm > 0.45) & (mm <= 0.6)].to_numpy(), f[mm > 0.6].to_numpy()
    cov, mcov = ap * u.n_apertures.to_numpy(), mm * ur.n_apertures.to_numpy()
    nearest = f.to_numpy()[np.abs(cov[:, None] - mcov[None, :]).argmin(axis=1)]
    rng = np.random.default_rng(0)
    fills = [(f"at {fl.min():.3f}, the lowest measured above 0.45 mm", lambda: fl.min()),
             (f"at {f.max():.3f}, the highest measured", lambda: f.max()),
             ("drawn by aperture bin (above 0.8 mm from 0.6-0.8)",
              lambda: np.where(ap <= 0.6, rng.choice(p06, len(u)), rng.choice(p08, len(u)))),
             ("from the measured design nearest in aperture x count", lambda: nearest)]
    print(f"  {len(u)} designs, unmeasured large-aperture designs filled:")
    for lab, large in fills:
        sims = []
        for _ in range(300):
            g = np.where(small, rng.choice(fs, len(u)), large())
            v = u.assign(p=u.ppds_ring1.to_numpy() * np.where(np.isnan(known), g, known))
            sims.append((rho(v, "p"), rho(v[small], "p"), partial(v[small], "p")))
        lo, hi = np.percentile(sims, [2.5, 97.5], axis=0)
        print(f"    {lab}: all {lo[0]:+.2f} to {hi[0]:+.2f}, small apertures {lo[1]:+.2f} to "
              f"{hi[1]:+.2f}, MPXI held {lo[2]:+.2f} to {hi[2]:+.2f}")
