import pandas as pd, numpy as np
from scipy.stats import spearmanr
d = pd.read_csv("results/asci_binsize.csv")
print(f"n = {len(d)}")
band = d[d.aperture_diam_mm.between(0.20, 0.45)]
print(f"in band: {len(band)}\n")

def partial(x, y, z):
    """Spearman of x,y controlling for z, via residuals of rank regression."""
    rx, ry, rz = (pd.Series(v).rank() for v in (x, y, z))
    bx = np.polyfit(rz, rx, 1); by = np.polyfit(rz, ry, 1)
    ex = rx - np.polyval(bx, rz); ey = ry - np.polyval(by, rz)
    return spearmanr(ex, ey).correlation

print("Is the ASCI-CNR link just aperture diameter in disguise?\n")
print(f"{'bin':>6} {'rho(ASCI,CNR)':>14} {'controlling for aperture':>26}")
for nb, lab in [(360,"1.0"),(1800,"0.2"),(3600,"0.1")]:
    c = f"asci_{nb}"
    sub = band.dropna(subset=[c, "cnr_sector_mean", "aperture_diam_mm"])
    if len(sub) < 10: continue
    raw = spearmanr(sub[c], sub.cnr_sector_mean).correlation
    par = partial(sub[c], sub.cnr_sector_mean, sub.aperture_diam_mm)
    print(f"{lab:>6} {raw:>14.2f} {par:>26.2f}")

print("\nWhat aperture itself does, in band:")
sub = band.dropna(subset=["asci_3600","cnr_sector_mean","aperture_diam_mm"])
print(f"  rho(aperture, CNR)        = {spearmanr(sub.aperture_diam_mm, sub.cnr_sector_mean).correlation:+.2f}")
print(f"  rho(aperture, ASCI 0.1deg)= {spearmanr(sub.aperture_diam_mm, sub.asci_3600).correlation:+.2f}")
