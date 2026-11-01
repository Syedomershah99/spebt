"""Share of the strip PPDF (counts) by strip width, corner-zeroed run, both collimator positions."""
import h5py
import numpy as np

T = "/projects/academic/rutaoyao/Omer/brain_maps_check/tests_31oct/corners/data"
fw, w = [], []
for L in (0, 1):
    with h5py.File(f"{T}/position_{L:03d}_ppdfs.hdf5") as h:
        P = h["ppdfs"][:]
    with h5py.File(f"{T}/beams_masks_configuration_{L:02d}.hdf5") as h:
        M = h["beam_mask"][:]
    with h5py.File(f"{T}/beams_properties_configuration_{L:02d}.hdf5") as h:
        p = h["beam_properties"][:]
    p = p[(p[:, 9] > 0) & np.isfinite(p[:, 3]) & (p[:, 4] > 0)]
    fw.append(p[:, 4])
    w.append(np.array([P[int(r[1]), M[int(r[1])] == r[2]].sum() for r in p]))
fw, w = np.concatenate(fw), np.concatenate(w)
for t in (5, 10, 20, 30):
    print(f"strips wider than {t:2d} mm: {np.mean(fw > t) * 100:5.1f}% of the strips, {w[fw > t].sum() / w.sum() * 100:5.1f}% of the counts")
print(f"median strip FWHM {np.median(fw):.2f} mm over {fw.size} strips; count-weighted median {fw[np.argsort(fw)][np.searchsorted(np.cumsum(w[np.argsort(fw)]) / w.sum(), 0.5)]:.1f} mm")
