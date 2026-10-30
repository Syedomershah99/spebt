"""Step 1: reproduce the archived per-seed sector-mean CNR from the saved reconstructions.

(a) the pipeline's own function, run_recon_comparison.compute_cnr, imported read-only from the
    Mac repo (run with python -B so no bytecode is written there); its cnr_results.npz goes to
    this work folder, never to the results directories;
(b) the numpy re-implementation in rods.py that step 2 builds on.
Both are compared with the archived cnr_results.npz of each seed.
"""
import contextlib
import io
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
W = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, "/Users/Omer/Desktop/RA/omer/spebt/recon")
import run_recon_comparison as rrc  # noqa: E402
from rods import PHANTOM, load_phantom, masks, cnr_parts  # noqa: E402

DESIGNS = {"benchmark": "tmi_reference_000", "MOBO-optimized": "nd2fine_960"}

ph, radii = load_phantom()
hot, bg, sectors = masks(ph)
worst_a = worst_b = 0.0
summary = {}
for name, d in DESIGNS.items():
    arch, pipe, mine = [], [], []
    for s in range(5):
        src = os.path.join(W, "data", d, f"cnr_repeat_seed{s}")
        out = os.path.join(W, "repro", d, f"seed{s}")
        os.makedirs(out, exist_ok=True)
        with contextlib.redirect_stdout(io.StringIO()):
            res = rrc.compute_cnr(os.path.join(src, "recon_mlem_T8.npz"), PHANTOM, out)
        a = np.asarray(np.load(os.path.join(src, "cnr_results.npz"))["sector_cnrs"], float)
        p = np.asarray(res["sector_cnrs"], float)
        recon = np.load(os.path.join(src, "recon_mlem_T8.npz"))["final"].astype(np.float64)
        m = np.array([cnr_parts(recon, sec, bg) for sec in sectors])
        worst_a = max(worst_a, np.abs(p - a).max())
        worst_b = max(worst_b, np.abs(m - a).max())
        arch.append(a.mean()); pipe.append(p.mean()); mine.append(m.mean())
        print(f"{name:15s} seed{s}: archived {a.mean():.4f}  pipeline {p.mean():.4f}  "
              f"numpy {m.mean():.4f}   per-sector max |diff| {np.abs(p - a).max():.1e} / "
              f"{np.abs(m - a).max():.1e}")
    arch, pipe, mine = map(np.array, (arch, pipe, mine))
    summary[name] = arch
    print(f"{name:15s} mean +/- SD: archived {arch.mean():.4f} +/- {arch.std(ddof=1):.4f}; "
          f"pipeline {pipe.mean():.4f} +/- {pipe.std(ddof=1):.4f}; "
          f"numpy {mine.mean():.4f} +/- {mine.std(ddof=1):.4f}\n")
print(f"largest per-sector difference from the archive: pipeline {worst_a:.2e}, numpy {worst_b:.2e}")
ok = worst_a < 1e-4 and worst_b < 1e-4
print("REPRODUCED" if ok else "NOT REPRODUCED")
sys.exit(0 if ok else 1)
