#!/usr/bin/env python3
"""
Compute reconstruction CNR for a single SAI SC-SPECT configuration.

Wraps the ML-EM reconstruction pipeline (forward projection -> ML-EM ->
CNR) into a single CLI entry point that the SLURM job (run_sai_pipeline.sh)
calls AFTER compute_metrics.py, so each MOBO evaluation contributes a CNR
value the optimizer can target.

Adds the cnr_mean column to the results CSV in place. If the column does
not exist yet, it is created. If the row for --config_name does not exist
yet (e.g. compute_metrics.py was not run first), a minimal row is appended.

Usage:
  python compute_cnr.py \
      --work_dir <path-to-config-work-dir> \
      --phantom_path <path-to-hot_rods_phantom.pt> \
      --out_csv results/results_summary_mobo.csv \
      --config_name mobo_0099_ap... \
      [--iterations 150]

Design notes:
  - 150 ML-EM iterations is the default. CNR ranking is stable well before
    convergence; 500 iter is only needed for clean final plots.
  - Recon artefacts (projections, recon, CNR npz) are written to
    <work_dir>/cnr_inloop/ so the per-config folder remains self-contained.
  - The function returns NaN if PPDF files are missing rather than raising,
    so an infeasible config does not crash the SLURM job.
"""
import argparse
import os
import sys
import time
import numpy as np
import pandas as pd
import torch

# Reuse the recon implementation that already exists for offline comparison
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.normpath(os.path.join(_HERE, "..", "recon")))
import run_recon_comparison as rrc  # noqa: E402


def compute_cnr_for_work_dir(
    work_dir: str,
    phantom_path: str,
    iterations: int = 150,
    T_sec: float = 10.0,
    e_hot: float = 10.0,
    e_bg: float = 2.0,
    out_subdir: str = "cnr_inloop",
) -> float:
    """Run forward project -> ML-EM -> CNR for one work_dir, return overall CNR.

    Returns NaN if PPDF files are missing (e.g. infeasible config).
    """
    out_dir = os.path.join(work_dir, out_subdir)
    os.makedirs(out_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        flist_path, flist = rrc.generate_flist(work_dir, out_dir)
    except FileNotFoundError as e:
        print(f"  [cnr] no PPDF files in {work_dir}: {e}")
        return float("nan")
    if not flist:
        return float("nan")

    projs_path = rrc.forward_project(
        flist, phantom_path, out_dir, device,
        T_sec=T_sec, e_hot=e_hot, e_bg=e_bg,
    )

    # Temporarily override ML-EM iteration count for fast in-loop runs
    saved_iter = rrc.N_ITERATIONS
    rrc.N_ITERATIONS = int(iterations)
    try:
        recon_path = rrc.run_mlem(flist, projs_path, out_dir, device)
    finally:
        rrc.N_ITERATIONS = saved_iter

    results = rrc.compute_cnr(recon_path, phantom_path, out_dir)
    return float(results["overall_cnr"])


def _write_cnr_to_csv(out_csv: str, config_name: str, cnr: float) -> None:
    """Set cnr_mean for the row matching config_name. Create row/column if needed."""
    if not os.path.exists(out_csv):
        pd.DataFrame([{"config": config_name, "cnr_mean": cnr}]).to_csv(out_csv, index=False)
        return

    df = pd.read_csv(out_csv)
    if "cnr_mean" not in df.columns:
        df["cnr_mean"] = float("nan")

    if "config" in df.columns and (df["config"] == config_name).any():
        df.loc[df["config"] == config_name, "cnr_mean"] = cnr
    else:
        new_row = {col: np.nan for col in df.columns}
        new_row["config"] = config_name
        new_row["cnr_mean"] = cnr
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    df.to_csv(out_csv, index=False)


def main():
    parser = argparse.ArgumentParser(description="In-loop CNR computation for SAI SC-SPECT")
    parser.add_argument("--work_dir", required=True,
                        help="Per-config directory containing position_*_ppdfs_t8_*.hdf5 files")
    parser.add_argument("--phantom_path", required=True,
                        help="Hot-rod phantom .pt file used to simulate projections")
    parser.add_argument("--out_csv", required=True,
                        help="Metrics CSV to update with cnr_mean (created if missing)")
    parser.add_argument("--config_name", required=True,
                        help="Config identifier (must match the row written by compute_metrics.py)")
    parser.add_argument("--iterations", type=int, default=150,
                        help="ML-EM iteration count for in-loop recon (default: 150)")
    parser.add_argument("--T_sec", type=float, default=10.0,
                        help="Scan time in seconds for forward projection")
    parser.add_argument("--e_hot", type=float, default=10.0,
                        help="Emission rate per hot voxel (counts/sec)")
    parser.add_argument("--e_bg", type=float, default=2.0,
                        help="Emission rate per background voxel (counts/sec)")
    parser.add_argument("--force_nan", action="store_true",
                        help="Write NaN row (for infeasible configs flagged upstream)")
    parser.add_argument("--reason", type=str, default="")
    args = parser.parse_args()

    if args.force_nan:
        print(f"[{args.config_name}] FORCE_NAN: {args.reason}")
        _write_cnr_to_csv(args.out_csv, args.config_name, float("nan"))
        return

    t0 = time.time()
    cnr = compute_cnr_for_work_dir(
        args.work_dir, args.phantom_path,
        iterations=args.iterations,
        T_sec=args.T_sec, e_hot=args.e_hot, e_bg=args.e_bg,
    )
    elapsed = time.time() - t0

    if np.isnan(cnr):
        print(f"[{args.config_name}] CNR = NaN ({elapsed:.1f} s)")
    else:
        print(f"[{args.config_name}] CNR = {cnr:.4f}  ({elapsed:.1f} s, {args.iterations} ML-EM iter)")

    _write_cnr_to_csv(args.out_csv, args.config_name, cnr)


if __name__ == "__main__":
    main()
