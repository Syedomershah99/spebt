import collections
import h5py
import numpy as np

D = "/projects/academic/rutaoyao/Omer/brain_maps_check/data"
for L in (0, 1):
    with h5py.File(f"{D}/beams_properties_configuration_{L:02d}.hdf5") as h:
        p = h["beam_properties"][:]
    with h5py.File(f"{D}/beams_masks_configuration_{L:02d}.hdf5") as h:
        m = h["beam_mask"][:]
    det, bid = p[:, 1].astype(int), p[:, 2].astype(int)
    kinds = collections.Counter()
    examples = []
    for i in range(m.shape[0]):
        ids_m = np.unique(m[i][m[i] > 0])
        rows = p[det == i]
        ids_p = rows[:, 2].astype(int)
        if set(ids_m.tolist()) == set(ids_p.tolist()):
            continue
        if len(ids_m) == 0:
            kinds["mask empty, props present"] += 1
        elif len(ids_p) == 0:
            kinds["props empty, mask present"] += 1
        elif set(ids_m.tolist()) < set(ids_p.tolist()):
            kinds["mask ids a subset of props ids"] += 1
        elif set(ids_p.tolist()) < set(ids_m.tolist()):
            kinds["props ids a subset of mask ids"] += 1
        else:
            kinds["other"] += 1
        if len(examples) < 6:
            npx_m = {int(b): int((m[i] == b).sum()) for b in ids_m}
            examples.append((i, npx_m, [(int(r[2]), round(float(r[4]), 2), int(r[9]), bool(np.isnan(r[3]))) for r in rows]))
    print(f"layout {L}: {dict(kinds)}")
    for e in examples:
        print("  det", e[0], "mask {id: pixels}", e[1], "| props (id, FWHM, npix, angle NaN):", e[2])
    npx = p[:, 9]
    print(f"  property rows with 0 pixels: {(npx == 0).sum()}, NaN angle: {np.isnan(p[:, 3]).sum()}, NaN FWHM: {np.isnan(p[:, 4]).sum()}")
