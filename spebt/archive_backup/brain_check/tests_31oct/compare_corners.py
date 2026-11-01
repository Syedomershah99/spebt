"""Test 1 result: maps from the corner-zeroed PPDFs against the baseline, inside the FOV circle."""
import h5py
import numpy as np

B = "/projects/academic/rutaoyao/Omer/brain_maps_check"
ix, iy = np.divmod(np.arange(40000), 200)
in_fov = np.hypot(ix + 0.5 - 100, iy + 0.5 - 100) <= 100
for name, base_f, test_f in (("window 10 mm", "brain_maps.hdf5", "corners_maps.hdf5"),
                             ("no window", "brain_maps_nofilter.hdf5", "corners_maps_nofilter.hdf5")):
    with h5py.File(f"{B}/out/{base_f}") as h, h5py.File(f"{B}/tests_31oct/corners/{test_f}") as g:
        for k in ("wfwhm", "asci", "sensitivity", "ppds"):
            if name == "no window" and k != "asci":
                continue
            a, b = h[k][:].ravel()[in_fov], g[k][:].ravel()[in_fov]
            print(f"{name:12s} {k:12s} FOV mean {np.nanmean(a):.4g} -> {np.nanmean(b):.4g} "
                  f"({(np.nanmean(b) / np.nanmean(a) - 1) * 100:+.2f}%); max abs change {np.nanmax(np.abs(b - a)):.3g}")
for L in (0, 1):
    with h5py.File(f"{B}/data/beams_properties_configuration_{L:02d}.hdf5") as h, \
         h5py.File(f"{B}/tests_31oct/corners/data/beams_properties_configuration_{L:02d}.hdf5") as g:
        p, q = h["beam_properties"][:], g["beam_properties"][:]
    for tag, x in (("baseline", p), ("zeroed", q)):
        ok = x[:, 9] > 0
        print(f"layout {L} {tag:8s}: strips with pixels {ok.sum()}, at 64 mm {(ok & (x[:, 4] >= 64 - 1e-6)).sum()}, "
              f"median FWHM {np.median(x[ok, 4]):.2f} mm, crystals without strips {720 - len(np.unique(x[ok, 1]))}")
