"""mlem_resident gives run_recon_comparison.run_mlem's image and iteration count (CPU, small random system)."""
import os
import sys

import h5py
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "recon"))
import mlem_resident  # noqa: E402
import run_recon_comparison as rrc  # noqa: E402


def test_matches_run_mlem(tmp_path, monkeypatch):
    rng = np.random.default_rng(0)
    flist = []
    for k in range(3):                                   # three PPDF files, 4 detectors each, on the 200 x 200 grid
        f = tmp_path / f"ppdf_{k}.hdf5"
        with h5py.File(f, "w") as h:
            h["ppdfs"] = rng.random((4, rrc.SFOV), dtype=np.float32) * (rng.random((4, rrc.SFOV)) < 0.05)
        flist.append(str(f))
    projs = tmp_path / "projs.npy"
    np.save(projs, rng.poisson(50, size=(3, 4)).astype(np.float32))
    monkeypatch.setattr(rrc, "N_ITERATIONS", 12)
    monkeypatch.setattr(rrc, "CONVERGENCE_TOL", 0.0)     # run every iteration in both
    z = np.load(rrc.run_mlem(flist, str(projs), str(tmp_path), torch.device("cpu")))
    mats, sens = mlem_resident.load_system(flist, rrc.SFOV, "cpu")
    pdata = torch.from_numpy(np.load(projs))
    est, diffs = mlem_resident.mlem(mats, sens, pdata, iters=12, tol=0.0)
    assert len(diffs) == len(z["diffs"]) == 12
    np.testing.assert_allclose(est.view(rrc.IMG_DIM, rrc.IMG_DIM).numpy(), z["final"], rtol=1e-6, atol=0)
    np.testing.assert_allclose(diffs, z["diffs"], rtol=1e-6)
