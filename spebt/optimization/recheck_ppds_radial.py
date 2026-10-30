#!/usr/bin/env python3
"""Recompute the archived PPDS with the strip-length fix (R11 QA, 30 Oct 2026).

compute_metrics._per_beam_radial_fwhm projected each strip's pixels with x on the second
pixel index until 30 Oct 2026, so strips near 0 and 90 degrees were measured across
(about 1 mm) instead of along (about 6.8 mm), and every archived PPDS (ppds_ring1, the
main search's sensitivity objective, included) came out about 2.6 times too large. This
recomputes, per design, PPDS per ring exactly as _ppds_components does (Eq. 6: PPDF_ij
over sum_b FWHM_tan * FWHM_rad, T8 poses summed per layout, the FOV mean averaged over
the layouts), once with the searches' projection ("bug", which must reproduce the
archive) and once with x on the first index ("fix"). Layouts are read one at a time.

  python recheck_ppds_radial.py DESIGNS.csv ROOT OUT.csv [--task K --ntasks N]

DESIGNS.csv: key, arm_dir, config, n_det_ring1, ppds_ring1 (archive), ppds_mean (archive);
the design directory is ROOT/arm_dir/config. Task K of N takes every N-th design.
"""
import argparse
import glob
import math
import os
import sys
import time

import h5py
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from compute_metrics import _PX_COL, _PX_ROW, _ring_slices


def lid_of(path):
    for part in os.path.basename(path).replace(".hdf5", "").split("_"):
        if part.isdigit() and len(part) == 3:
            return part
    return None


def radial(pix, w, angle, x_first):
    """2.355 times the PPDF-weighted SD of a strip's pixels along `angle` (0 if degenerate)."""
    if pix.size < 3:
        return 0.0
    x, y = (_PX_ROW[pix], _PX_COL[pix]) if x_first else (_PX_COL[pix], _PX_ROW[pix])
    r = x * math.cos(angle) + y * math.sin(angle)
    ws = w.sum()
    if ws <= 0.0:
        return 0.0
    m = np.dot(w, r) / ws
    v = np.dot(w, (r - m) ** 2) / ws
    return 2.355 * math.sqrt(v) if v > 0.0 else 0.0


def design_ppds(work_dir, n1):
    """{"bug"|"fix": per-ring PPDS (4) and all-detector PPDS}, averaged over layouts."""
    ppdf_files = sorted(glob.glob(os.path.join(work_dir, "position_*_ppdfs_t8_*.hdf5")))
    props = {lid_of(f): f for f in glob.glob(os.path.join(work_dir, "beams_properties_configuration_*.hdf5"))}
    masks = {lid_of(f): f for f in glob.glob(os.path.join(work_dir, "beams_masks_configuration_*.hdf5"))}
    lids = sorted({lid_of(f) for f in ppdf_files} - {None})
    per = {"bug": [], "fix": []}
    n_det = None
    for lid in lids:
        ppdf = None
        for f in [f for f in ppdf_files if lid_of(f) == lid]:
            with h5py.File(f, "r") as h:
                a = h["ppdfs"][:].astype(np.float64)
            ppdf = a if ppdf is None else ppdf + a
            del a
        pf = props.get(lid) or (next(iter(props.values())) if len(props) == 1 else None)
        mf = masks.get(lid) or (next(iter(masks.values())) if len(masks) == 1 else None)
        # shortcut: raises on a missing or mismatched layout, and averages in a layout with no
        # strip area, where compute_metrics skips both; fine for complete design directories
        # (all 316 had two), mirror compute_metrics before reusing it on partial ones.
        if pf is None or mf is None:
            raise RuntimeError(f"layout {lid}: no property or mask file")
        with h5py.File(pf, "r") as h:
            bp = h["beam_properties"][:]
        with h5py.File(mf, "r") as h:
            mk = h["beam_mask"][:]
        n_det, n_pix = ppdf.shape
        if mk.shape != ppdf.shape:
            raise RuntimeError(f"layout {lid}: masks {mk.shape} vs PPDFs {ppdf.shape}")
        det = bp[:, 1].astype(np.int64)
        if det.size and det.min() >= 1 and det.max() <= n_det:
            det = det - 1
        info = {(int(d), int(b)): (float(f_), float(a_))
                for d, b, a_, f_ in zip(det, bp[:, 2].astype(np.int64), bp[:, 3], bp[:, 4])
                if 0 <= d < n_det and np.isfinite(a_) and np.isfinite(f_) and f_ > 0.0}
        sumv = {"bug": np.zeros(n_det), "fix": np.zeros(n_det)}
        for i in range(n_det):
            row = mk[i]
            lit = np.flatnonzero(row > 0)
            if lit.size == 0:
                continue
            ids = row[lit]
            for b in np.unique(ids):
                p = info.get((i, int(b)))
                if p is None:
                    continue
                pix = lit[ids == b]
                w = ppdf[i, pix]
                for k, xf in (("bug", False), ("fix", True)):
                    fr = radial(pix, w, p[1], xf)
                    if fr > 0.0:
                        sumv[k][i] += p[0] * fr
        s = ppdf.sum(axis=1)                    # sum_j PPDF_ij: the FOV mean is sum_i S_i / V_i / N
        sl = _ring_slices(n_det, n1)
        if sl is None:
            raise RuntimeError(f"ring layout unresolved: n_det {n_det}, n1 {n1}")
        for k in ("bug", "fix"):
            v = sumv[k]
            q = np.where(v > 0, s / np.where(v > 0, v, 1.0), 0.0) / n_pix
            per[k].append([q[lo:hi].sum() for lo, hi in sl] + [q.sum()])
        del ppdf, mk
    out = {"n_det": n_det, "n_layouts": len(lids)}
    for k in ("bug", "fix"):
        m = np.asarray(per[k]).mean(axis=0)
        for r in range(4):
            out[f"ppds_ring{r + 1}_{k}"] = m[r]
        out[f"ppds_all_{k}"] = m[4]
    return out


def selfcheck():
    # a strip along x (first index) at angle 0: the fix measures its length, the bug its width
    pix = np.array([r * 200 + 100 for r in range(40, 160)])
    w = np.ones(pix.size)
    fix, bug = radial(pix, w, 0.0, True), radial(pix, w, 0.0, False)
    assert abs(fix - 2.355 * 6.0 / math.sqrt(12)) < 0.05 and bug == 0.0, (fix, bug)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("designs"); ap.add_argument("root"); ap.add_argument("out")
    ap.add_argument("--task", type=int, default=0); ap.add_argument("--ntasks", type=int, default=1)
    a = ap.parse_args()
    selfcheck()
    d = pd.read_csv(a.designs).iloc[a.task::a.ntasks]
    rows = []
    for _, r in d.iterrows():
        t0 = time.time()
        try:
            res = design_ppds(os.path.join(a.root, r.arm_dir, r.config), int(r.n_det_ring1))
            err = ""
        except Exception as e:                  # one bad design must not lose the others
            res, err = {}, f"{type(e).__name__}: {e}"
        row = dict(key=r.key, arm_dir=r.arm_dir, config=r.config, n_det_ring1=int(r.n_det_ring1),
                   archive_ppds_ring1=r.ppds_ring1, archive_ppds_mean=r.ppds_mean, error=err,
                   sec=round(time.time() - t0, 1), **res)
        rows.append(row)
        print(f"{r.key}: archive {r.ppds_ring1:.6g}, bug {res.get('ppds_ring1_bug', float('nan')):.6g}, "
              f"fix {res.get('ppds_ring1_fix', float('nan')):.6g}  ({row['sec']} s) {err}", flush=True)
    pd.DataFrame(rows).to_csv(a.out, index=False)


if __name__ == "__main__":
    main()
