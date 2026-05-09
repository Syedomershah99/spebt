#!/usr/bin/env python3
"""
Multi-Objective Bayesian Optimization Agent for SAI SC-SPECT.

4 objectives (all maximized internally via negation where needed):
  1. FWHM          — minimize (negate)
  2. ASCI          — maximize
  3. sensitivity   — maximize
  4. MPXI          — minimize (negate)

Uses ModelListGP (one SingleTaskGP per objective) + qLogNEHVI.

Design vector: (aperture_diam_mm, n_apertures, scint_radial_thickness_mm, ring_thickness_mm)

Usage:
  python mobo_agent.py --results_csv results/results_summary.csv
"""
import os
import logging
import warnings
import torch
import pandas as pd
from botorch.models import SingleTaskGP
from botorch.models.model_list_gp_regression import ModelListGP
from botorch.fit import fit_gpytorch_mll
from botorch.utils.transforms import normalize, unnormalize
from botorch.acquisition.multi_objective.logei import qLogNoisyExpectedHypervolumeImprovement
from botorch.optim import optimize_acqf
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.mlls.sum_marginal_log_likelihood import SumMarginalLogLikelihood

# --- Logging ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("MOBO_Agent_SAI")
warnings.filterwarnings("ignore", category=UserWarning)

# --- Design space bounds ---
# 4D: aperture_diam, n_apertures, n_det_ring1, n_det_ring2
# n_det values must be even (2 crystals per cell). Rounded after acquisition.
PARAM_NAMES = ["aperture_diam_mm", "n_apertures", "n_det_ring1", "n_det_ring2"]
BOUNDS_MIN = [0.2, 60.0, 120.0, 180.0]
BOUNDS_MAX = [1.0, 360.0, 660.0, 960.0]
DIM = len(PARAM_NAMES)

# --- Objective columns (as they appear in the CSV) ---
OBJ_COLUMNS = ["fwhm_mean", "asci_pct", "sensitivity_mean", "mpxi_mean"]
# Directions: +1 = maximize, -1 = minimize (we negate minimization objectives)
OBJ_DIRECTIONS = [-1.0, 1.0, 1.0, -1.0]
OBJ_NAMES = ["FWHM (min)", "ASCI (max)", "Sensitivity (max)", "MPXI (min)"]


def get_next_candidate(results_csv: str):
    """
    1. Load results CSV with all 4 objectives.
    2. Fit ModelListGP (one GP per objective).
    3. Optimize qLogNEHVI, q=1.
    4. Return next design point.
    """
    logger.info("--- Starting Multi-Objective BO Step ---")

    if not os.path.exists(results_csv):
        raise FileNotFoundError(f"Could not find {results_csv}")

    df = pd.read_csv(results_csv)

    # Drop rows where any objective is NaN (infeasible configs)
    df_valid = df.dropna(subset=OBJ_COLUMNS)
    n_total = len(df)
    n_valid = len(df_valid)
    n_infeasible = n_total - n_valid
    logger.info(f"Loaded {n_total} rows, {n_valid} feasible, {n_infeasible} infeasible (dropped)")

    if n_valid < 3:
        raise ValueError(f"Need at least 3 feasible points for MOBO, got {n_valid}")

    # --- Prepare tensors ---
    train_x = torch.tensor(df_valid[PARAM_NAMES].values, dtype=torch.double)
    train_y_raw = torch.tensor(df_valid[OBJ_COLUMNS].values, dtype=torch.double)

    # Apply direction signs (negate FWHM and MPXI so all objectives are maximized)
    directions = torch.tensor(OBJ_DIRECTIONS, dtype=torch.double)
    train_y = train_y_raw * directions

    bounds = torch.tensor([BOUNDS_MIN, BOUNDS_MAX], dtype=torch.double)
    train_x_norm = normalize(train_x, bounds)

    # Log current objective ranges
    for i, name in enumerate(OBJ_NAMES):
        col_vals = train_y_raw[:, i]
        logger.info(f"  {name}: min={col_vals.min():.4f}, max={col_vals.max():.4f}, "
                     f"mean={col_vals.mean():.4f}")

    # --- Standardize each objective independently ---
    y_mean = train_y.mean(dim=0, keepdim=True)
    y_std = train_y.std(dim=0, keepdim=True).clamp(min=1e-6)
    train_y_std = (train_y - y_mean) / y_std

    # --- Fit ModelListGP (one SingleTaskGP per objective) ---
    logger.info("Fitting ModelListGP (one GP per objective)...")
    models = []
    for i, name in enumerate(OBJ_NAMES):
        gp = SingleTaskGP(train_x_norm, train_y_std[:, i:i+1])
        mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
        fit_gpytorch_mll(mll)
        models.append(gp)

        # Log ARD lengthscales
        try:
            ls = gp.covar_module.base_kernel.lengthscale.detach().cpu().numpy().flatten()
            ls_str = ", ".join(f"{p}={l:.3f}" for p, l in zip(PARAM_NAMES, ls))
            logger.info(f"  GP[{name}] lengthscales: {ls_str}")
        except Exception:
            pass

    model = ModelListGP(*models)
    logger.info("ModelListGP trained successfully.")

    # --- Reference point (in standardized space) ---
    # Use worst observed value minus a small margin for each objective
    ref_point_std = train_y_std.min(dim=0).values - 0.1
    logger.info(f"Reference point (standardized): {ref_point_std.tolist()}")

    # --- Optimize qLogNEHVI ---
    logger.info("Optimizing qLogNEHVI acquisition function...")
    acqf = qLogNoisyExpectedHypervolumeImprovement(
        model=model,
        ref_point=ref_point_std.tolist(),
        X_baseline=train_x_norm,
        prune_baseline=True,
        cache_root=True,
    )

    candidate_norm, acq_value = optimize_acqf(
        acq_function=acqf,
        bounds=torch.stack([torch.zeros(DIM), torch.ones(DIM)]).double(),
        q=1,
        num_restarts=10,
        raw_samples=512,
    )

    # --- Un-normalize ---
    candidate_physical = unnormalize(candidate_norm, bounds)
    next_diam = candidate_physical[0, 0].item()
    next_n_ap = int(round(candidate_physical[0, 1].item()))
    next_n_det1 = int(round(candidate_physical[0, 2].item()))
    next_n_det2 = int(round(candidate_physical[0, 3].item()))
    # n_det values must be even (2 crystals per cell)
    if next_n_det1 % 2 != 0:
        next_n_det1 += 1
    if next_n_det2 % 2 != 0:
        next_n_det2 += 1

    logger.info(f"Acquisition value: {acq_value.item():.6f}")
    logger.info(f"SUGGESTION -> aperture_diam={next_diam:.4f} mm | "
                f"n_apertures={next_n_ap} | "
                f"n_det_ring1={next_n_det1} | n_det_ring2={next_n_det2}")

    return next_diam, next_n_ap, next_n_det1, next_n_det2


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="MOBO Agent for SAI SC-SPECT")
    parser.add_argument("--results_csv", type=str, required=True,
                        help="Path to results CSV with fwhm_mean, asci_pct, sensitivity_mean, mpxi_mean columns")
    args = parser.parse_args()

    diam, n_ap, n_det1, n_det2 = get_next_candidate(args.results_csv)
    print(f"\nSuggested next config:")
    print(f"  aperture_diam   = {diam:.4f} mm")
    print(f"  n_apertures     = {n_ap}")
    print(f"  n_det_ring1     = {n_det1}")
    print(f"  n_det_ring2     = {n_det2}")
