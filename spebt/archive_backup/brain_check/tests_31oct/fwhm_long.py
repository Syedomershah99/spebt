"""Test 2: re-measure every strip's FWHM with pymatana's own functions on a 64 mm line (must
reproduce the stored values) and on a 300 mm line (spans the grid's 283 mm diagonal), then
recompute the maps with the 300 mm widths."""
import sys
import h5py
import numpy as np
import torch
import yaml

B = "/projects/academic/rutaoyao/Omer/brain_maps_check"
sys.path.insert(0, f"{B}/code/beam-analysis")
from scanner_modeling.beam_property_extract import (beam_sampling_line_batch, beam_samples_on_points_batch,
                                                    full_width_half_maximum_1d_batch)
from scanner_modeling.geometry_2d_io import load_scanner_layouts, load_scanner_layout_geometries
from scanner_modeling.geometry_2d_utils import fov_tensor_dict
from arg_design_maps import layout_maps, load_matrix, pixel_xy

cfg = yaml.safe_load(open(f"{B}/brain.yml"))
fov = fov_tensor_dict(n_pixels=(200, 200), mm_per_pixel=(1.0, 1.0))
layouts, _ = load_scanner_layouts(cfg["paths"]["scanner_layouts_dir"], cfg["paths"]["scanner_layouts_filename"])


def fwhm(p2, det_c, centers, masks, length, n):
    pts, dist = beam_sampling_line_batch(det_c, centers, n_samples=n, length=length)
    k = centers.shape[0]
    data = p2.view(-1).unsqueeze(0).expand(k, -1).clone().masked_fill_(~masks, 0).view(k, 200, 200)
    f, _ = full_width_half_maximum_1d_batch(dist.view(1, -1).expand(-1, n), beam_samples_on_points_batch(data, pts, fov))
    return f.numpy()


xy = pixel_xy((200, 200), (1.0, 1.0))
ix, iy = np.divmod(np.arange(40000), 200)
in_fov = np.hypot(ix + 0.5 - 100, iy + 0.5 - 100) <= 100
sens, ppds, wn, wd = (np.zeros(40000) for _ in range(4))
cover = np.zeros((40000, 360), bool)
res = {}
with torch.no_grad():
    for L in (0, 1):
        _, det = load_scanner_layout_geometries(L, layouts)
        det_c = det.mean(1)
        P = load_matrix(f"{B}/data/position_{L:03d}_ppdfs.hdf5", "ppdfs", np.float32)
        M = load_matrix(f"{B}/data/beams_masks_configuration_{L:02d}.hdf5", "beam_mask", np.int32)
        with h5py.File(f"{B}/data/beams_properties_configuration_{L:02d}.hdf5") as h:
            props = h["beam_properties"][:]
        f64, f300 = np.full(len(props), np.nan), np.full(len(props), np.nan)
        Pt, Mt = torch.from_numpy(P), torch.from_numpy(M)
        for i in np.unique(props[:, 1]).astype(int):
            rows = np.flatnonzero((props[:, 1] == i) & (props[:, 9] > 0))
            if rows.size == 0:
                continue
            centers = torch.tensor(props[rows][:, 5:7], dtype=torch.float32)
            masks = torch.stack([Mt[i] == int(b) for b in props[rows, 2]])
            p2 = Pt[i].view(200, 200)
            f64[rows] = fwhm(p2, det_c[i], centers, masks, 64.0, 4096)
            f300[rows] = fwhm(p2, det_c[i], centers, masks, 300.0, 19200)
        ok = props[:, 9] > 0
        stored = props[:, 4]
        d = np.abs(f64[ok] - stored[ok])
        print(f"layout {L}: {ok.sum()} strips with pixels; 64 mm line vs stored FWHM: max abs diff {d.max():.3g} mm, "
              f"{np.mean(d < 1e-3) * 100:.2f}% within 0.001 mm")
        sent = ok & (stored >= 64 - 1e-6)
        print(f"  stored 64 mm strips: {sent.sum()}; on the 300 mm line: median {np.median(f300[sent]):.1f} mm, "
              f"5/95%: {np.round(np.percentile(f300[sent], [5, 95]), 1)}; still at 300: {(f300[sent] >= 300 - 1e-3).sum()}")
        changed = ok & ~sent & (np.abs(f300 - stored) > 0.01)
        print(f"  other strips whose FWHM changes by > 0.01 mm: {changed.sum()} of {(ok & ~sent).sum()}"
              + (f"; largest change {np.max(np.abs(f300[changed] - stored[changed])):.2f} mm" if changed.any() else ""))
        w = np.array([P[int(r[1]), M[int(r[1])] == r[2]].sum() for r in props])
        print(f"  share of strip PPDF in the stored 64 mm strips: {w[sent].sum() / w[ok].sum() * 100:.3f}%")
        res[L] = (props, f64, f300)
        p300 = props.copy()
        p300[ok, 4] = f300[ok]
        s, dd, a, b = layout_maps(P.astype(np.float64), p300, M, xy, 360, (0.0, 10.0), cover)
        sens += s / 2; ppds += dd / 2; wn += a; wd += b
wf = np.divide(wn, wd, out=np.full(40000, np.nan), where=wd > 0)
with h5py.File(f"{B}/out/brain_maps.hdf5") as h:
    base = {k: h[k][:].ravel() for k in ("wfwhm", "asci", "ppds")}
asci = cover.sum(1) / 3.6
for k, v in (("wfwhm", wf), ("asci", asci), ("ppds", ppds)):
    print(f"maps with 300 mm widths, in the FOV: {k} mean {np.nanmean(v[in_fov]):.4g} (64 mm line: {np.nanmean(base[k][in_fov]):.4g}); "
          f"max abs change {np.nanmax(np.abs(v[in_fov] - base[k][in_fov])):.3g}")
np.savez_compressed(f"{B}/tests_31oct/fwhm_long.npz", **{f"f300_{L}": res[L][2] for L in res},
                    **{f"f64_{L}": res[L][1] for L in res}, wfwhm=wf, asci=asci, ppds=ppds)
