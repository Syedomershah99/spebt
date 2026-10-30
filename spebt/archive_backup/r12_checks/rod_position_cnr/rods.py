"""Masks and per-rod bookkeeping that copy run_recon_comparison.compute_cnr exactly.

compute_cnr (recon/run_recon_comparison.py, md5 87b861bc..., identical on the Mac and CCR):
  hot      = phantom_padded > 0.1
  bg       = 20 px <= r <= 90 px from (99.5, 99.5), minus hot  (1.0 to 4.5 mm)
  sector k = |wrap(atan2(i - 99.5, j - 99.5) - angle_k)| < 30 deg, & hot,
             angle_k in [60, 0, 300, 240, 180, 120] paired with radii 0.100 ... 0.225 mm
  CNR_k    = (mean(recon[sector k]) - mean(recon[bg])) / std(recon[bg], ddof=0) (+1e-12)
  objective = mean_k CNR_k  (cnr_sector_mean)
Here i is the first (row) index of the 200 x 200 image, j the second; pixel p = 200 i + j.
"""
import numpy as np
import scipy.ndimage as ndi
import torch

PHANTOM = "/Users/Omer/Desktop/RA/omer/spebt/data/sai_10mm/hot_rods_phantom_10.0_mm_x_10.0_mm.pt"
IMG = 200
MM = 0.05
C = (IMG - 1) / 2.0
ANGLES_DEG = [60, 0, 300, 240, 180, 120]


def load_phantom(path=PHANTOM):
    d = torch.load(path, map_location="cpu", weights_only=False)
    t = d["Phantom tensor"].numpy().astype(np.float64)
    assert t.shape == (IMG, IMG), t.shape          # no padding needed
    radii = np.asarray(d["Metadata"]["rods radii in mm"], dtype=float)
    return t, radii


def geometry():
    ii, jj = np.meshgrid(np.arange(IMG), np.arange(IMG), indexing="ij")
    rr2 = (ii - C) ** 2 + (jj - C) ** 2
    ang = np.arctan2(ii - C, jj - C)
    return ii, jj, rr2, ang


def masks(phantom):
    ii, jj, rr2, ang = geometry()
    hot = phantom > 0.1
    fov = IMG / 2.0
    bg = (rr2 >= (fov * 0.2) ** 2) & (rr2 <= (fov * 0.9) ** 2) & ~hot
    sectors = []
    for a in ANGLES_DEG:
        d = np.abs((ang - np.deg2rad(a) + np.pi) % (2 * np.pi) - np.pi)
        sectors.append((d < np.deg2rad(30)) & hot)
    return hot, bg, sectors


def rods(phantom, radii):
    """One record per connected hot component: centre, distance, sector, pixels."""
    hot, bg, sectors = masks(phantom)
    lab, n = ndi.label(hot, structure=np.ones((3, 3)))
    ii, jj, rr2, ang = geometry()
    out = []
    for k in range(1, n + 1):
        m = lab == k
        w = phantom[m]
        ci = float((ii[m] * w).sum() / w.sum())
        cj = float((jj[m] * w).sum() / w.sum())
        sec = [s for s in range(6) if (sectors[s] & m).any()]
        assert len(sec) == 1, f"rod {k} straddles sectors {sec}"
        s = sec[0]
        assert (sectors[s] & m).sum() == m.sum(), f"rod {k} partly outside its sector"
        out.append(dict(label=k, mask=m, ci=ci, cj=cj, npix=int(m.sum()), sector=s,
                        radius_mm=float(radii[s]),
                        dist_mm=float(np.hypot(ci - C, cj - C) * MM),
                        near_mm=float(np.sqrt(rr2[m].min()) * MM),
                        far_mm=float(np.sqrt(rr2[m].max()) * MM)))
    return out, lab


def cnr_parts(recon, hot_sel, bg):
    bv = recon[bg]
    mu, sd = bv.mean(), bv.std() + 1e-12
    return (recon[hot_sel].mean() - mu) / sd
