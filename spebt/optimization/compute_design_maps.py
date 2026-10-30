#!/usr/bin/env python3
"""Per-design system characterization for the benchmark / MOBO-optimized slides.

Dr. Yao's R10-rY (29 Oct 2026) adds three slides after the CNR result: FWHM and MPXI
histograms, ASCI maps, sensitivity and PPDS maps, benchmark beside MOBO-optimized.
This writes, for one design directory, everything those figures need, computed with
the deck's metric definitions and checked against the archive's stored values:

  strips   FWHM (col 4), sensitivity (col 7), detector, layout, angle-valid, the 1%
           sensitivity floor of windowed ASCI/MPXI, and the FWHM filter (<= 0.45 mm)
  mpxi     strips per detector, all (floor) and filtered, as compute_mpxi_variants
  asci     per-pixel ASCI, 1 degree bins: filtered (asci_pct_fwhm0p45) and all strips
  sens     Sens_j = sum_i PPDF_ij (strategy document Eq. 7), all rings and ring I
  ppds     Eq. 6, PPDS_j = sum_i PPDF_ij / A_i, and the R10-rY slide 6 form,
           sum over i with PPDF_ij > 0 of (sum_k PPDF_ik) / A_i, all rings and ring I
  wfwhm    Dr. Yao's wFWHM_j (R10-rY slide 9, 29 Oct): over the strips (i,b) whose PPDF
           is nonzero at pixel j, FWHM_ib weighted by the strip's whole PPDF, sum_k p_ibk;
           and the same map with the searches' weight, the strip's mean PPDF (col 7)

A_i = sum_b FWHM_tan * FWHM_rad over a detector's strips, as compute_metrics. The radial
FWHM is computed twice: as the searches' code did until 30 Oct 2026 (x on the second
pixel index, so strips near 0 and 90 degrees were measured across), and with x on the
first index, the convention of pymatana's pixels_coordinates, in which the stored strip
angle is measured (compute_metrics since the fix). The "as searched" Eq. 6 ring-I mean
must reproduce the archive's ppds_ring1.

Maps are averaged over the two layouts (T8 poses summed within a layout), as
arg_compute_ppds does; pixel p is (p // 200, p % 200) = (x, y) index, 0.05 mm pixels.

  python compute_design_maps.py WORK_DIR N_DET_RING1 OUT.npz
"""
import glob
import math
import os
import sys

import h5py
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from compute_metrics import _PX_COL, _PX_ROW, _ring_slices, ASCI_NBINS_ANGULAR, FOV_NPIX

FLOOR, WIN = 0.01, 0.45
N_PIX = FOV_NPIX[0] * FOV_NPIX[1]


def lid_of(path):
    for part in os.path.basename(path).replace(".hdf5", "").split("_"):
        if part.isdigit() and len(part) == 3:
            return part
    return None


def _radial(mask_row, ppdf_row, beam_id, angle, x_first):
    """PPDF-weighted 2.355 sigma of a strip's pixels along the direction `angle`."""
    pix = np.flatnonzero(mask_row == beam_id)
    if pix.size < 3:
        return 0.0
    x, y = (_PX_ROW[pix], _PX_COL[pix]) if x_first else (_PX_COL[pix], _PX_ROW[pix])
    r = x * math.cos(angle) + y * math.sin(angle)
    w = ppdf_row[pix].astype(np.float64)
    if w.sum() <= 0.0:
        return 0.0
    m = np.dot(w, r) / w.sum()
    v = np.dot(w, (r - m) ** 2) / w.sum()
    return 2.355 * math.sqrt(v) if v > 0 else 0.0


def _per_beam_radial_fwhm(mask_row, ppdf_row, beam_id, angle):
    """The searches' radial FWHM, x on the second pixel index (compute_metrics before 30 Oct)."""
    return _radial(mask_row, ppdf_row, beam_id, angle, x_first=False)


def radial_fwhm_x_first(mask_row, ppdf_row, beam_id, angle):
    """x on the first pixel index, as the stored angle is (compute_metrics since 30 Oct)."""
    return _radial(mask_row, ppdf_row, beam_id, angle, x_first=True)


def main(work_dir, n1, out):
    ppdf_files = sorted(glob.glob(os.path.join(work_dir, "position_*_ppdfs_t8_*.hdf5")))
    props = {lid_of(f): f for f in glob.glob(os.path.join(work_dir, "beams_properties_configuration_*.hdf5"))}
    masks = {lid_of(f): f for f in glob.glob(os.path.join(work_dir, "beams_masks_configuration_*.hdf5"))}
    lids = sorted(set(lid_of(f) for f in ppdf_files) & set(props) & set(masks))
    if not lids:
        sys.exit(f"no complete layout in {work_dir}")

    maps = {k: np.zeros(N_PIX) for k in ("sens_all", "sens_r1", "eq6_searched_all", "eq6_searched_r1",
                                         "eq6_all", "eq6_r1", "ry_all", "ry_r1",
                                         "ry_mask_all", "ry_mask_r1")}
    wnum, wden, mnum, mden = (np.zeros(N_PIX) for _ in range(4))
    cover_win = np.zeros((N_PIX, ASCI_NBINS_ANGULAR), dtype=bool)
    cover_all = np.zeros_like(cover_win)
    boundaries = np.arange(ASCI_NBINS_ANGULAR + 1) / 180.0 * np.pi
    strips, dets, chk = [], [], []
    for lid in lids:
        ppdf = None
        for f in [f for f in ppdf_files if lid_of(f) == lid]:
            with h5py.File(f, "r") as h:
                a = h["ppdfs"][:].astype(np.float64)
            ppdf = a if ppdf is None else ppdf + a
        with h5py.File(props[lid], "r") as h:
            bp = h["beam_properties"][:]
        with h5py.File(masks[lid], "r") as h:
            mk = h["beam_mask"][:]
        n_det = ppdf.shape[0]
        sl = _ring_slices(n_det, n1)
        r1 = np.zeros(n_det, dtype=bool); r1[sl[0][0]:sl[0][1]] = True

        det = bp[:, 1].astype(np.int64)
        if det.size and det.min() >= 1 and det.max() <= n_det:
            det = det - 1
        bid, ang, fw, sens = bp[:, 2].astype(np.int64), bp[:, 3], bp[:, 4], bp[:, 7]
        valid = ~np.isnan(ang)
        smax = sens[valid].max() if np.isfinite(sens[valid]).any() else 0.0
        floor = valid & (sens > smax * FLOOR) if smax > 0 else valid.copy()
        good = np.isfinite(fw) & (fw > 0)
        win = floor & good & (fw <= WIN)
        strips.append(np.c_[np.full(det.size, int(lid)), det, fw, sens, valid, floor & good, win, ang,
                            bp[:, 9]])

        # strips per detector: windowed MPXI counts floor & FWHM < 0.45 (compute_mpxi_variants)
        sfin = np.isfinite(sens); smax_all = sens[sfin].max() if sfin.any() else 0.0
        keep = sens > smax_all * FLOOR if smax_all > 0 else np.ones(det.size, bool)
        k_all = np.bincount(det[keep], minlength=n_det)
        k_win = np.bincount(det[keep & good & (fw < WIN)], minlength=n_det)
        dets.append(np.c_[np.full(n_det, int(lid)), np.arange(n_det), r1, k_all, k_win])

        # per-pixel angle coverage, as analyze_asci_window (floor, valid angle, FWHM > 0)
        b_of = np.searchsorted(boundaries, ang, side="left") - 1
        ok = floor & good & (b_of >= 0) & (b_of < ASCI_NBINS_ANGULAR)
        lut = {}
        for d_, b_, k_, f_ in zip(det[ok], bid[ok], b_of[ok], fw[ok]):
            lut[(int(d_), int(b_))] = (int(k_), float(f_))
        for i in range(n_det):
            row = mk[i]
            lit = np.flatnonzero(row)
            if lit.size == 0:
                continue
            ids = row[lit].astype(np.int64)
            for b in np.unique(ids):
                v = lut.get((i, int(b)))
                if v is None:
                    continue
                pix = lit[ids == b]
                cover_all[pix, v[0]] = True
                if v[1] <= WIN:
                    cover_win[pix, v[0]] = True

        # A_i both ways; Eq. 6 and the slide 6 form
        info = {(int(d_), int(b_)): (float(f_), float(a_))
                for d_, b_, a_, f_ in zip(det, bid, ang, fw)
                if 0 <= d_ < n_det and np.isfinite(a_) and np.isfinite(f_) and f_ > 0}
        A_s, A_x = np.zeros(n_det), np.zeros(n_det)
        sens_of = {(int(d_), int(b_)): float(s_) for d_, b_, s_ in zip(det, bid, sens) if np.isfinite(s_)}
        for i in range(n_det):
            row = mk[i]
            for b in np.unique(row[row > 0]):
                p = info.get((i, int(b)))
                if p is None:
                    continue
                pix = np.flatnonzero(row == b)
                pix = pix[ppdf[i, pix] > 0]
                tot = ppdf[i, pix].sum()          # sum_k p_ibk over the strip
                wnum[pix] += tot * p[0]; wden[pix] += tot
                sm = sens_of.get((i, int(b)), 0.0)
                mnum[pix] += sm * p[0]; mden[pix] += sm
                rs = _per_beam_radial_fwhm(row, ppdf[i], int(b), p[1])
                rx = radial_fwhm_x_first(row, ppdf[i], int(b), p[1])
                if rs > 0:
                    A_s[i] += p[0] * rs
                if rx > 0:
                    A_x[i] += p[0] * rx
                if len(chk) < 4000 and rs > 0 and rx > 0:
                    chk.append((p[1], p[0], rs, rx))
        S = ppdf.sum(axis=1)
        for tag, A in (("eq6_searched", A_s), ("eq6", A_x)):
            ok = A > 0
            c = np.zeros_like(ppdf)
            c[ok] = ppdf[ok] / A[ok, None]
            maps[f"{tag}_all"] += c.sum(axis=0)
            maps[f"{tag}_r1"] += c[r1].sum(axis=0)
        okx = A_x > 0
        dens = np.where(okx, S / np.where(okx, A_x, 1.0), 0.0)
        seen = ppdf > 0
        maps["ry_all"] += (seen * dens[:, None]).sum(axis=0)
        maps["ry_r1"] += (seen[r1] * dens[r1, None]).sum(axis=0)
        # PPDF_ij > 0 holds almost everywhere (tails through the metal), which makes the
        # slide 6 form flat; inside the detector's strips (the masks: 1% of the detector's
        # peak along an arc, beams_boundaries_radians) it is the map the formula describes
        inmask = mk > 0
        maps["ry_mask_all"] += (inmask * dens[:, None]).sum(axis=0)
        maps["ry_mask_r1"] += (inmask[r1] * dens[r1, None]).sum(axis=0)
        maps["sens_all"] += ppdf.sum(axis=0)
        maps["sens_r1"] += ppdf[r1].sum(axis=0)
        dets[-1] = np.c_[dets[-1], A_s, A_x, S]
        del ppdf, mk

    for k in maps:
        maps[k] /= len(lids)
    maps["wfwhm_total"] = np.where(wden > 0, wnum / np.where(wden > 0, wden, 1), np.nan)
    maps["wfwhm_mean"] = np.where(mden > 0, mnum / np.where(mden > 0, mden, 1), np.nan)
    np.savez_compressed(
        out, **maps, layouts=np.array([int(l) for l in lids]),
        asci_win=cover_win.sum(axis=1) / ASCI_NBINS_ANGULAR * 100.0,
        asci_all=cover_all.sum(axis=1) / ASCI_NBINS_ANGULAR * 100.0,
        strips=np.vstack(strips), dets=np.vstack(dets), radial_check=np.array(chk),
        strip_cols=np.array(["layout", "det", "fwhm", "sens", "valid", "floor", "win", "angle", "npix"]),
        det_cols=np.array(["layout", "det", "ring1", "k_all", "k_win", "A_searched", "A_xfirst", "S"]))
    k = np.vstack(dets)
    print(f"{work_dir}: layouts {lids}")
    print(f"  ppds_ring1 as searched (Eq. 6 mean): {maps['eq6_searched_r1'].mean():.6g}; "
          f"x-first {maps['eq6_r1'].mean():.6g}; slide-6 form {maps['ry_r1'].mean():.6g}, inside "
          f"the strips {maps['ry_mask_r1'].mean():.6g}")
    print(f"  asci_pct_fwhm0p45: {cover_win.mean() * 100:.3f}; all strips {cover_all.mean() * 100:.3f}")
    kw = k[:, 4]
    print(f"  mpxi_windowed_active_mean: {kw[kw > 0].mean():.3f}")
    s = np.vstack(strips)
    g = (s[:, 2] > 0) & np.isfinite(s[:, 2])
    w = np.where(np.isfinite(s[:, 3]), s[:, 3], 0.0)
    print(f"  fwhm_mean {s[g, 2].mean():.4f}, fwhm_weighted_mean {np.dot(s[g, 2], w[g]) / w[g].sum():.4f}")
    t = w * np.where(np.isfinite(s[:, 8]), s[:, 8], 0.0)
    print(f"  total-weighted FWHM {np.dot(s[g, 2], t[g]) / t[g].sum():.4f}; wFWHM_j map mean "
          f"{np.nanmean(maps['wfwhm_total']):.4f} (mean-weight map {np.nanmean(maps['wfwhm_mean']):.4f})")


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]), sys.argv[3])
