"""Test 1 input: copies of the two PPDF files with every pixel outside the FOV circle set to 0."""
import os
import h5py
import numpy as np

B = "/projects/academic/rutaoyao/Omer/brain_maps_check"
T = f"{B}/tests_31oct/corners/data"
os.makedirs(T, exist_ok=True)
ix, iy = np.divmod(np.arange(200 * 200), 200)
outside = np.hypot(ix + 0.5 - 100, iy + 0.5 - 100) > 100
for L in (0, 1):
    with h5py.File(f"{B}/data/position_{L:03d}_ppdfs.hdf5", "r") as h:
        P = h["ppdfs"][:]
    share = P[:, outside].sum() / P.sum()
    P[:, outside] = 0
    with h5py.File(f"{T}/position_{L:03d}_ppdfs.hdf5", "w") as h:
        h["ppdfs"] = P
    print(f"layout {L}: zeroed {outside.sum()} pixels holding {share * 100:.2f}% of the PPDF")
