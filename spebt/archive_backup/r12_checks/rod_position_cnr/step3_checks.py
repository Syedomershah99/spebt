"""Step 3: supporting checks for step 2.

 a. background mean and SD inside and outside the cut (explains the regional-background variant)
 b. inner/outer interaction: per seed, CNR_inner / CNR_outer of each design (they share the
    background, so this is a contrast-only ratio), Welch t-test across designs, more digits
 c. ASCI difference near the crossover (0.05 mm annuli) and share of the field where MOBO is higher
 d. descriptive: other per-pixel maps in the same files over the inner and outer rod pixels
"""
import os
import sys

import numpy as np
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
W = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from rods import IMG, MM, geometry, load_phantom, masks, rods  # noqa: E402

DESIGNS = [("benchmark", "tmi_reference_000", "bench"), ("MOBO", "nd2fine_960", "mobo")]
ph, radii = load_phantom()
hot, bg, sectors = masks(ph)
R, _ = rods(ph, radii)
dist = np.array([r["dist_mm"] for r in R]); sec = np.array([r["sector"] for r in R])
ii, jj, rr2, ang = geometry()
rmm = np.sqrt(rr2) * MM
recon = {n: [np.load(os.path.join(W, "data", d, f"cnr_repeat_seed{s}", "recon_mlem_T8.npz"))["final"]
             .astype(np.float64) for s in range(5)] for n, d, _ in DESIGNS}
maps = {n: np.load(f"/Users/Omer/Desktop/RA/omer/spebt/archive_backup/design_maps/{m}.npz") for n, _, m in DESIGNS}


def secmean_cnr(img, sel, mu, sd):
    return np.mean([(img[np.logical_or.reduce([R[n]["mask"] for n in range(len(R)) if sel[n] and sec[n] == k])].mean()
                     - mu) / sd for k in range(6)])


print("a. background (pipeline annulus minus rods), mean over 5 seeds")
for T in (2.50, 2.60):
    for n, _, _ in DESIGNS:
        g = [(x[bg].mean(), x[bg].std()) for x in recon[n]]
        i_ = [(x[bg & (rmm < T)].mean(), x[bg & (rmm < T)].std()) for x in recon[n]]
        o_ = [(x[bg & (rmm >= T)].mean(), x[bg & (rmm >= T)].std()) for x in recon[n]]
        f = lambda v: "mean %.2f SD %.2f" % tuple(np.mean(v, axis=0))
        print(f"  cut {T:.2f} {n:9s}: whole {f(g)} | 1.0-{T:.2f} mm {f(i_)} | {T:.2f}-4.5 mm {f(o_)}")

print("\nb. per-seed inner/outer CNR ratio (same background, so a contrast ratio)")
for T in (2.50, 2.55, 2.60, float(np.median(dist))):
    inner = dist < T
    r = {}
    for n, _, _ in DESIGNS:
        v = []
        for x in recon[n]:
            mu, sd = x[bg].mean(), x[bg].std() + 1e-12
            v.append(secmean_cnr(x, inner, mu, sd) / secmean_cnr(x, ~inner, mu, sd))
        r[n] = np.array(v)
    t = stats.ttest_ind(r["MOBO"], r["benchmark"], equal_var=False)
    print(f"  cut {T:.3f}: benchmark {np.round(r['benchmark'], 3)} ; MOBO {np.round(r['MOBO'], 3)} ; "
          f"Welch t {t.statistic:.2f}, p {t.pvalue:.4f}")

print("\nc. ASCI (asci_win) MOBO - benchmark in 0.05 mm annuli, 2.30 to 2.90 mm")
diff = maps["MOBO"]["asci_win"].reshape(IMG, IMG) - maps["benchmark"]["asci_win"].reshape(IMG, IMG)
for a in np.arange(2.30, 2.90, 0.05):
    m = (rmm >= a) & (rmm < a + 0.05)
    print(f"  {a:.2f}-{a + 0.05:.2f} mm: {diff[m].mean():+6.2f}  ({m.sum()} px)")
print(f"  MOBO higher on {100 * (diff > 0).mean():.1f}% of the 200 x 200 field, lower on {100 * (diff < 0).mean():.1f}%, "
      f"equal on {100 * (diff == 0).mean():.1f}%")
disk = rmm < 2.0
print(f"  within 2.0 mm: MOBO lower on {100 * (diff[disk] < 0).mean():.1f}% of pixels; "
      f"2.0-2.6 mm: lower on {100 * (diff[(rmm >= 2.0) & (rmm < 2.6)] < 0).mean():.1f}%")

print("\nd. other maps in the same files over rod pixels (descriptive), cut 2.55 mm")
inner = dist < 2.55
pin = np.logical_or.reduce([R[n]["mask"] for n in range(len(R)) if inner[n]])
pout = np.logical_or.reduce([R[n]["mask"] for n in range(len(R)) if not inner[n]])
for key, lab in (("sens_all", "sensitivity, all rings"), ("ry_mask_all", "PPDS (slide 15 form, in strips)"),
                 ("wfwhm_total", "weighted FWHM (slide 13), mm"), ("asci_win", "ASCI, %")):
    b = maps["benchmark"][key].reshape(IMG, IMG); m = maps["MOBO"][key].reshape(IMG, IMG)
    print(f"  {lab:34s} inner rods: bench {np.nanmean(b[pin]):.4g}, MOBO {np.nanmean(m[pin]):.4g} "
          f"(ratio {np.nanmean(m[pin]) / np.nanmean(b[pin]):.3f}) | outer rods: bench {np.nanmean(b[pout]):.4g}, "
          f"MOBO {np.nanmean(m[pout]):.4g} (ratio {np.nanmean(m[pout]) / np.nanmean(b[pout]):.3f})")
