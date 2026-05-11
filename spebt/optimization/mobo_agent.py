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
import math
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

# --- Feasibility constraint ---
# Aperture overlap: aperture_diam < circumference / n_apertures
HR_R_CENTER = 67.5 / 2.0 + 2.5 / 2.0  # 35.0 mm
HR_CIRCUMFERENCE = 2.0 * math.pi * HR_R_CENTER  # ~219.9 mm
SAFETY_MARGIN = 0.95


def is_feasible(diam, n_ap):
    """Check if aperture config fits on the HR ring without overlap."""
    return diam < SAFETY_MARGIN * HR_CIRCUMFERENCE / n_ap


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

    # Separate feasible and failed rows
    df_valid = df.dropna(subset=OBJ_COLUMNS)
    df_failed = df[df[OBJ_COLUMNS].isna().any(axis=1)]
    n_total = len(df)
    n_valid = len(df_valid)
    n_failed = len(df_failed)
    logger.info(f"Loaded {n_total} rows, {n_valid} feasible, {n_failed} failed")

    if n_valid < 3:
        raise ValueError(f"Need at least 3 feasible points for MOBO, got {n_valid}")

    # --- Assign penalty values to failed configs so the GP learns to avoid them ---
    # Use worst observed value (in the "maximize" direction) with a margin
    if n_failed > 0 and len(df_failed[PARAM_NAMES].dropna()) > 0:
        valid_vals = df_valid[OBJ_COLUMNS].values
        # Penalty: for each objective, assign worst-feasible value made 20% worse
        penalty_row = []
        for i, d in enumerate(OBJ_DIRECTIONS):
            col = valid_vals[:, i]
            if d > 0:  # maximize → penalty is well below minimum
                penalty_row.append(float(col.min()) * 0.5)
            else:  # minimize → penalty is well above maximum
                penalty_row.append(float(col.max()) * 2.0)
        logger.info(f"Penalty values for {n_failed} failed configs: {penalty_row}")

        # Build combined training data: valid + failed-with-penalty
        df_failed_with_params = df_failed.dropna(subset=PARAM_NAMES)
        failed_x = df_failed_with_params[PARAM_NAMES].values
        failed_y = [penalty_row] * len(df_failed_with_params)

        train_x = torch.tensor(
            pd.concat([df_valid[PARAM_NAMES], df_failed_with_params[PARAM_NAMES]]).values,
            dtype=torch.double)
        train_y_raw = torch.tensor(
            list(df_valid[OBJ_COLUMNS].values) + failed_y,
            dtype=torch.double)
        logger.info(f"Training with {n_valid} feasible + {len(df_failed_with_params)} penalized = "
                     f"{len(train_x)} total points")
    else:
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

    # Feasibility constraint in normalized space:
    # aperture_diam < 0.95 * circumference / n_apertures
    # Physical: diam * n_ap < 0.95 * circumference = threshold
    diam_min, diam_range = BOUNDS_MIN[0], BOUNDS_MAX[0] - BOUNDS_MIN[0]
    nap_min, nap_range = BOUNDS_MIN[1], BOUNDS_MAX[1] - BOUNDS_MIN[1]
    threshold = SAFETY_MARGIN * HR_CIRCUMFERENCE

    def is_feasible_norm(x_norm):
        """Check feasibility in normalized [0,1] space. x_norm shape: (..., DIM)"""
        diam = diam_min + x_norm[..., 0] * diam_range
        n_ap = nap_min + x_norm[..., 1] * nap_range
        return diam * n_ap < threshold

    def feasible_ic_generator(acq_function, bounds, num_restarts, raw_samples, options=None, **kwargs):
        """Generate feasible initial conditions for constrained acquisition optimization."""
        # Oversample heavily to get enough feasible points
        n_candidates = max(raw_samples * 10, 10000)
        X_rnd = torch.rand(n_candidates, 1, DIM, dtype=bounds.dtype, device=bounds.device)
        # Filter feasible
        feasible_mask = is_feasible_norm(X_rnd.squeeze(1))
        X_feasible = X_rnd[feasible_mask]
        if len(X_feasible) < num_restarts:
            raise RuntimeError(f"Only {len(X_feasible)} feasible ICs found, need {num_restarts}")
        # Evaluate acquisition and pick top candidates
        n_eval = min(len(X_feasible), raw_samples)
        with torch.no_grad():
            acq_values = acq_function(X_feasible[:n_eval])
        top_indices = torch.argsort(acq_values, descending=True)[:num_restarts]
        return X_feasible[top_indices]

    candidate_norm, acq_value = optimize_acqf(
        acq_function=acqf,
        bounds=torch.stack([torch.zeros(DIM), torch.ones(DIM)]).double(),
        q=1,
        num_restarts=20,
        raw_samples=1024,
        ic_generator=feasible_ic_generator,
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

    # Final feasibility check (belt and suspenders)
    if not is_feasible(next_diam, next_n_ap):
        logger.warning(f"Candidate infeasible (d={next_diam:.4f}, n={next_n_ap}), clamping aperture_diam")
        next_diam = min(next_diam, SAFETY_MARGIN * HR_CIRCUMFERENCE / next_n_ap - 0.01)

    # --- Deduplication: reject if too close to a previously tried config ---
    all_x = torch.tensor(df[PARAM_NAMES].dropna().values, dtype=torch.double)
    candidate_vec = torch.tensor([next_diam, float(next_n_ap), float(next_n_det1), float(next_n_det2)],
                                 dtype=torch.double)
    # Normalize distances by parameter ranges to compare fairly
    ranges = bounds[1] - bounds[0]
    norm_dists = ((all_x - candidate_vec) / ranges).norm(dim=1)
    min_dist = norm_dists.min().item() if len(norm_dists) > 0 else 1.0

    if min_dist < 0.02:  # less than 2% of design space away = duplicate
        logger.warning(f"Candidate too close to existing config (dist={min_dist:.4f}), adding exploration noise")
        import random
        for _attempt in range(50):
            # Random feasible perturbation
            noise_diam = next_diam + random.uniform(-0.2, 0.2)
            noise_n_ap = next_n_ap + random.randint(-60, 60)
            noise_n_det1 = next_n_det1 + random.choice([-100, -50, 0, 50, 100])
            noise_n_det2 = next_n_det2 + random.choice([-100, -50, 0, 50, 100])
            # Clamp to bounds
            noise_diam = max(BOUNDS_MIN[0], min(BOUNDS_MAX[0], noise_diam))
            noise_n_ap = max(int(BOUNDS_MIN[1]), min(int(BOUNDS_MAX[1]), noise_n_ap))
            noise_n_det1 = max(int(BOUNDS_MIN[2]), min(int(BOUNDS_MAX[2]), noise_n_det1))
            noise_n_det2 = max(int(BOUNDS_MIN[3]), min(int(BOUNDS_MAX[3]), noise_n_det2))
            if noise_n_det1 % 2 != 0:
                noise_n_det1 += 1
            if noise_n_det2 % 2 != 0:
                noise_n_det2 += 1
            if is_feasible(noise_diam, noise_n_ap):
                perturbed_vec = torch.tensor([noise_diam, float(noise_n_ap),
                                              float(noise_n_det1), float(noise_n_det2)], dtype=torch.double)
                new_dists = ((all_x - perturbed_vec) / ranges).norm(dim=1)
                if new_dists.min().item() >= 0.02:
                    next_diam, next_n_ap = noise_diam, noise_n_ap
                    next_n_det1, next_n_det2 = noise_n_det1, noise_n_det2
                    logger.info(f"Perturbed to: d={next_diam:.4f} n={next_n_ap} nd1={next_n_det1} nd2={next_n_det2}")
                    break

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
