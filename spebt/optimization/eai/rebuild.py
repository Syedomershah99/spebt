#!/usr/bin/env python3
"""Give every re-measurement design its PPDFs, then write remeasure.sh's list.

ANALYSIS_PLAN.md: a candidate whose PPDFs are missing is rebuilt from its
parameters and re-measured, never dropped (addendum 3), and the top designs of
CCR's six arms are rebuilt on Grace (addendum 2). Reads analyze_campaign.py's
candidates.csv and ccr_candidates.csv (name = search__design, work_dir, design
vector). A design whose work_dir holds its 16 PPDF files is used where it is:
its own pipeline finished, since its CNR is recorded. Any other is built by
run_sai_pipeline.sh from its design vector in ROOT/<search>/<design>, --par at
a time. A build counts as done only when its results_rebuild.csv holds a finite
in-loop CNR: the PPDF step creates all 16 files when it starts, so their number
proves nothing, while the CNR step reads them all. An unfinished build is wiped
and started again. Every build must reproduce its design's recorded values: an
Empire AI design its MPXI and in-loop CNR (the noise seed comes from the name;
same platform) to 1e-6, a CCR design its MPXI to 1e-3 (CCR is another CPU, and
its August CNR was unseeded). Each build's log and results are copied to --logs.

Writes --list (via a temporary file, so a failed run leaves no list), one
directory per line in the order of the input files, and exits 1 if any design
is still unbuilt or does not reproduce.

Usage (inside a Grace job: rebuild.sh):
  python rebuild.py --root /ddn/scratch/.../rebuild --list designs.txt --logs DIR candidates.csv ccr_candidates.csv
"""
import argparse
import glob
import math
import os
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

# run_sai_pipeline.sh variable -> design-vector column
VECTOR = {"APERTURE_DIAM": "aperture_diam_mm", "N_APERTURES": "n_apertures",
          "N_DET_RING1": "n_det_ring1", "N_DET_RING2": "n_det_ring2",
          "D2_INNER": "d2_inner_mm", "D3_INNER": "d3_inner_mm"}
PPDFS = "position_*_ppdfs_t8_*.hdf5"
BUILT = "results_rebuild.csv"


def n_ppdfs(d):
    return len(glob.glob(os.path.join(d, PPDFS))) if isinstance(d, str) else 0


def build_result(t):
    """The last row of a build's results, or None unless it has a finite CNR."""
    p = os.path.join(t, BUILT)
    if n_ppdfs(t) != 16 or not os.path.exists(p):
        return None
    r = pd.read_csv(p, float_precision="round_trip").iloc[-1]
    c = r.get("cnr_sector_mean")
    return r if c is not None and math.isfinite(c) else None


def target(r, root):
    """Where the design's PPDFs are, or will be built."""
    search, config = r["name"].split("__", 1)
    return r["work_dir"] if n_ppdfs(r.get("work_dir")) == 16 else os.path.join(root, search, config)


def build(r, code):
    """Wipe any unfinished attempt, then one pipeline run with the design's exact
    parameters (repr keeps every digit)."""
    t = r["target"]
    os.makedirs(t, exist_ok=True)
    for f in glob.glob(os.path.join(t, PPDFS)) + [os.path.join(t, BUILT)]:
        if os.path.exists(f):
            os.remove(f)
    env = dict(os.environ, CONFIG_NAME=r["name"].split("__", 1)[1], WORK_DIR=t,
               A_MM="0.2", B_MM="0.2", RESULTS_CSV=os.path.join(t, BUILT),
               **{k: str(int(r[c])) if c.startswith("n_") else repr(float(r[c])) for k, c in VECTOR.items()})
    with open(os.path.join(t, "rebuild.log"), "w") as log:
        rc = subprocess.run(["bash", os.path.join(code, "optimization", "run_sai_pipeline.sh")],
                            env=env, stdout=log, stderr=subprocess.STDOUT).returncode
    print(f"built {r['name']}: exit {rc}", flush=True)


def reproduces(r, b):
    """(ok, text): does build row b reproduce candidate r's recorded values?"""
    mpxi, cnr = b.get("mpxi_windowed_active_mean"), b.get("cnr_sector_mean")
    if str(r["name"]).startswith("ccr_"):
        ok = abs(mpxi - r["mpxi"]) <= 1e-3 * abs(r["mpxi"])
    elif pd.isna(r.get("mpxi")):                       # a reference: nothing recorded to compare
        ok = True
    else:
        ok = abs(mpxi - r["mpxi"]) <= 1e-6 * abs(r["mpxi"]) and abs(cnr - r["inloop_cnr"]) <= 1e-6
    return ok, f"MPXI {r.get('mpxi')!r} -> {mpxi!r}, in-loop CNR {r.get('inloop_cnr')!r} -> {cnr!r}"


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("candidates", nargs="+")
    ap.add_argument("--root", required=True, help="where missing designs are built")
    ap.add_argument("--list", required=True, help="remeasure.sh's DESIGNS file to write")
    ap.add_argument("--logs", help="folder (on /projects) for each build's log and results")
    ap.add_argument("--par", type=int, default=6, help="pipelines at a time (6 fit a Grace node)")
    ap.add_argument("--code", default=os.environ.get("CODE_DIR"), help="the spebt tree")
    a = ap.parse_args()
    if os.path.exists(a.list):
        os.remove(a.list)

    d = pd.concat([pd.read_csv(f, float_precision="round_trip") for f in a.candidates],
                  ignore_index=True).drop_duplicates("name")
    d["target"] = [target(r, a.root) for _, r in d.iterrows()]
    d["built_here"] = [t != r.get("work_dir") for t, (_, r) in zip(d.target, d.iterrows())]
    todo = d[[bh and build_result(t) is None for bh, t in zip(d.built_here, d.target)]]
    no_vector = todo[todo.reindex(columns=list(VECTOR.values())).isna().any(axis=1)]
    if len(no_vector):
        sys.exit(f"ERROR: no PPDFs and no design vector to build them from: {list(no_vector.name)}")
    print(f"{len(d)} designs: {int((~d.built_here).sum())} have their own PPDFs, "
          f"{int(d.built_here.sum()) - len(todo)} already rebuilt, {len(todo)} to build", flush=True)
    with ThreadPoolExecutor(a.par) as ex:
        list(ex.map(lambda r: build(r, a.code), [r for _, r in todo.iterrows()]))

    unbuilt, wrong = [], []
    for _, r in d[d.built_here].iterrows():
        b = build_result(r.target)
        if b is None:
            unbuilt.append(r["name"])
            continue
        ok, text = reproduces(r, b)
        print(f"{r['name']}: {text}{'' if ok else '  DOES NOT REPRODUCE'}")
        wrong += [] if ok else [r["name"]]
        if a.logs:
            os.makedirs(a.logs, exist_ok=True)
            for f, ext in (("rebuild.log", "log"), (BUILT, "csv")):
                shutil.copy(os.path.join(r.target, f), os.path.join(a.logs, f"{r['name']}.{ext}"))
    if unbuilt or wrong:
        sys.exit(f"ERROR: unbuilt {unbuilt}; not reproduced {wrong}")
    with open(a.list + ".tmp", "w") as f:
        f.writelines(t + "\n" for t in d.target)
    os.replace(a.list + ".tmp", a.list)
    print(f"wrote {len(d)} directories to {a.list}")


if __name__ == "__main__":
    main()
