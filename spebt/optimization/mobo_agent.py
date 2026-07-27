#!/usr/bin/env python3
"""
Multi-Objective Bayesian Optimization Agent for SAI SC-SPECT.

5 objectives (all maximized internally via negation where needed):
  1. FWHM          — minimize (negate)
  2. ASCI          — maximize
  3. sensitivity   — maximize
  4. MPXI          — minimize (negate)
  5. CNR           — maximize  (reconstructed contrast-to-noise ratio from
                                the in-loop 150-iter ML-EM run on the hot-rod
                                phantom; computed by compute_cnr.py and
                                appended to each row by run_sai_pipeline.sh
                                step [4/4]. Direct reconstruction quality
                                objective per Dr. Yao's guidance — the four
                                proxy metrics did not always align with
                                CNR in the tested regime.)

(PPDS was evaluated earlier and put on hold — Spearman ρ vs reconstructed
CNR was not positive across 16 validated configurations. The PPDS
computation remains available in compute_metrics.py but is not used as
an objective here.)

Uses ModelListGP (one SingleTaskGP per objective) + qLogNEHVI.

Design vector: (aperture_diam_mm, n_apertures, n_det_ring1, n_det_ring2,
                d2_inner_mm, d3_inner_mm)

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

# --- Logging ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("MOBO_Agent_SAI")
# Selective warning suppression: BoTorch emits an InputDataWarning every fit
# because our train_x is normalised to [0,1] (which is what it wants but the
# check misfires). Everything else — numerical warnings from GPyTorch about
# jitter or ill-conditioning — we want to see.
warnings.filterwarnings(
    "ignore",
    message="Data (input features) is not contained to the unit cube.*",
    category=UserWarning,
)

# --- Design space bounds ---
# 6D: aperture_diam, n_apertures, n_det_ring1, n_det_ring2, d2_inner, d3_inner
# n_det values must be even (2 crystals per cell). Rounded after acquisition.
# n_apertures max: geometry generator enforces MIN_SPACING=0.8mm between
# aperture centers → chord = 2*R*sin(π/n) >= 0.8 → n_max ≈ 274 at R=35mm.
# Use 270 with safety margin.
# d2_inner / d3_inner are the inner diameters of detector rings 2 and 3; rings 1
# and 4 stay fixed at 260 / 650 mm.
# d3 lower bound is NOT the ring gap -- ring 3 carries a fixed 960 crystals, and
# below ~379 mm those cells overlap (see max_crystals_on_ring). 385 leaves a
# small margin over that hard floor.
PARAM_NAMES = ["aperture_diam_mm", "n_apertures", "n_det_ring1", "n_det_ring2",
               "d2_inner_mm", "d3_inner_mm"]
BOUNDS_MIN = [0.2, 60.0, 120.0, 180.0, 270.0, 385.0]
BOUNDS_MAX = [1.0, 270.0, 660.0, 960.0, 540.0, 640.0]
DIM = len(PARAM_NAMES)

# --- Feasibility constraints ---
# 1) Aperture overlap: aperture_diam < circumference / n_apertures
HR_R_CENTER = 67.5 / 2.0 + 2.5 / 2.0  # 35.0 mm
HR_CIRCUMFERENCE = 2.0 * math.pi * HR_R_CENTER  # ~219.9 mm
SAFETY_MARGIN = 0.95

# 2) Detector ring ordering: D1 < D2 < D3 < D4 with a minimum radial gap.
# RY (Jul 2026): 10 mm is reasonable given mechanical mounting and cooling.
# Rings stay concentric and share an axial extent — he wants the method
# established before more variables are added, so there is no axial term here.
D1_INNER_MM = 260.0
D4_INNER_MM = 650.0
MIN_RING_GAP_MM = 10.0

# 3) Detector cell packing. Each ring holds n_scint/2 cells, and every cell needs
# 2*W + gap of arc length. Shrinking a ring while keeping its crystal count
# raises the packing density, so diameter and detector count interact: the
# geometry generator raises ValueError once clearance reaches zero, which would
# burn a whole MOBO iteration on a config that could never be built. These
# constants mirror build_sc_spect_detector_rings in the geometry generator.
SCINT_TANGENTIAL_MM = 0.84
INTRA_CELL_GAP_MM = 0.84
SCINT_RADIAL_MM = 6.0
CELL_SPAN_MM = 2.0 * SCINT_TANGENTIAL_MM + INTRA_CELL_GAP_MM  # 2.52 mm
# Rings 3 and 4 carry fixed crystal counts (geometry generator: 40*12*2, 40*15*2)
N_DET_RING3 = 960
N_DET_RING4 = 1200


def is_feasible(diam, n_ap):
    """Check if aperture config fits on the HR ring without overlap."""
    return diam < SAFETY_MARGIN * HR_CIRCUMFERENCE / n_ap


def is_ring_ordering_ok(d2_inner, d3_inner):
    """Check D1 < D2 < D3 < D4 with at least MIN_RING_GAP_MM between rings."""
    return (
        d2_inner - D1_INNER_MM >= MIN_RING_GAP_MM
        and d3_inner - d2_inner >= MIN_RING_GAP_MM
        and D4_INNER_MM - d3_inner >= MIN_RING_GAP_MM
    )


def max_crystals_on_ring(inner_diam_mm):
    """Largest crystal count a ring of this diameter can hold without overlap."""
    r_center = inner_diam_mm / 2.0 + SCINT_RADIAL_MM / 2.0
    return 4.0 * math.pi * r_center / CELL_SPAN_MM


def is_ring_packing_ok(n_det1, n_det2, d2_inner, d3_inner):
    """Check every ring's crystals fit at its diameter.

    Ring 1 is fixed at 260 mm, which caps n_det_ring1 at ~663 -- the reason the
    existing bound is 660. Ring 4 is fixed in both diameter and count. Only
    ring 2 (both variable) and ring 3 (fixed count, variable diameter) can be
    driven infeasible by the D2/D3 expansion.
    """
    return (
        n_det1 < max_crystals_on_ring(D1_INNER_MM)
        and n_det2 < max_crystals_on_ring(d2_inner)
        and N_DET_RING3 < max_crystals_on_ring(d3_inner)
        and N_DET_RING4 < max_crystals_on_ring(D4_INNER_MM)
    )


def is_feasible_full(diam, n_ap, n_det1, n_det2, d2_inner, d3_inner):
    """All design constraints together."""
    return (
        is_feasible(diam, n_ap)
        and is_ring_ordering_ok(d2_inner, d3_inner)
        and is_ring_packing_ok(n_det1, n_det2, d2_inner, d3_inner)
    )


# --- Objective columns (as they appear in the CSV) ---
# cnr_mean is populated in-loop by compute_cnr.py (150-iter ML-EM) and by
# backfill_cnr.py for the 16 already-reconstructed configs.
OBJ_COLUMNS = ["fwhm_mean", "asci_pct", "sensitivity_mean", "mpxi_mean", "cnr_mean"]
# Directions: +1 = maximize, -1 = minimize (we negate minimization objectives)
OBJ_DIRECTIONS = [-1.0, 1.0, 1.0, -1.0, 1.0]
OBJ_NAMES = ["FWHM (min)", "ASCI (max)", "Sensitivity (max)", "MPXI (min)", "CNR (max)"]


def get_next_candidate(results_csv: str):
    """
    1. Load results CSV with all OBJ_COLUMNS objectives.
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
    # Use worst observed value (in the "maximize" direction) with a margin.
    #
    # IMPORTANT: this multiplicative scheme assumes every metric in OBJ_COLUMNS
    # is strictly positive across the observed data. All our current metrics
    # (fwhm_mean, asci_pct, sensitivity_mean, mpxi_mean, cnr_mean) satisfy that.
    # If a future metric can go negative, "col.min() * 0.5" would make the
    # penalty MORE favourable than the worst observation — subtract-a-margin
    # would be needed instead. Asserting positivity guards against a silent
    # regression here.
    if n_failed > 0 and len(df_failed[PARAM_NAMES].dropna()) > 0:
        valid_vals = df_valid[OBJ_COLUMNS].values
        assert (valid_vals >= 0).all(), (
            "penalty calculation assumes strictly non-negative objective values; "
            "negative value detected — rework the penalty scheme before proceeding"
        )
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
        cache_root=False,
    )

    # Feasibility constraint in normalized space:
    # aperture_diam < 0.95 * circumference / n_apertures
    # Physical: diam * n_ap < 0.95 * circumference = threshold
    diam_min, diam_range = BOUNDS_MIN[0], BOUNDS_MAX[0] - BOUNDS_MIN[0]
    nap_min, nap_range = BOUNDS_MIN[1], BOUNDS_MAX[1] - BOUNDS_MIN[1]
    nd2_min, nd2_range = BOUNDS_MIN[3], BOUNDS_MAX[3] - BOUNDS_MIN[3]
    d2_min, d2_range = BOUNDS_MIN[4], BOUNDS_MAX[4] - BOUNDS_MIN[4]
    d3_min, d3_range = BOUNDS_MIN[5], BOUNDS_MAX[5] - BOUNDS_MIN[5]
    threshold = SAFETY_MARGIN * HR_CIRCUMFERENCE

    def is_feasible_norm(x_norm):
        """Check feasibility in normalized [0,1] space. x_norm shape: (..., DIM)

        Three constraints:
          1. apertures must not overlap on the HR ring
          2. the detector rings must stay ordered with the minimum radial gap
          3. ring 2's crystals must fit at its proposed diameter

        The bounds already guarantee d2 >= D1 + gap, d3 <= D4 - gap, and that
        ring 3's fixed 960 crystals fit (d3 >= 385), so only the middle ordering
        inequality and ring 2's packing need checking here.
        """
        diam = diam_min + x_norm[..., 0] * diam_range
        n_ap = nap_min + x_norm[..., 1] * nap_range
        aperture_ok = diam * n_ap < threshold

        d2 = d2_min + x_norm[..., 4] * d2_range
        d3 = d3_min + x_norm[..., 5] * d3_range
        ordering_ok = (d3 - d2) >= MIN_RING_GAP_MM

        n_det2 = nd2_min + x_norm[..., 3] * nd2_range
        max_nd2 = 4.0 * math.pi * (d2 / 2.0 + SCINT_RADIAL_MM / 2.0) / CELL_SPAN_MM
        packing_ok = n_det2 < max_nd2

        return aperture_ok & ordering_ok & packing_ok

    def feasible_ic_generator(acq_function, bounds, num_restarts, raw_samples, options=None, **kwargs):
        """Generate feasible initial conditions for constrained acquisition optimization."""
        # Use Sobol sequence for better coverage of the 4D space
        n_candidates = 2048
        sobol = torch.quasirandom.SobolEngine(dimension=DIM, scramble=True)
        X_rnd = sobol.draw(n_candidates, dtype=bounds.dtype).unsqueeze(1)
        feasible_mask = is_feasible_norm(X_rnd.squeeze(1))
        X_feasible = X_rnd[feasible_mask]
        if len(X_feasible) < num_restarts:
            raise RuntimeError(f"Only {len(X_feasible)} feasible ICs found, need {num_restarts}")
        # Evaluate acquisition in batches to limit memory
        n_eval = min(len(X_feasible), 256)
        with torch.no_grad():
            acq_values = acq_function(X_feasible[:n_eval])
        top_indices = torch.argsort(acq_values, descending=True)[:num_restarts]
        return X_feasible[top_indices]

    candidate_norm, acq_value = optimize_acqf(
        acq_function=acqf,
        bounds=torch.stack([torch.zeros(DIM), torch.ones(DIM)]).double(),
        q=1,
        num_restarts=10,
        raw_samples=256,
        ic_generator=feasible_ic_generator,
    )

    # --- Un-normalize ---
    candidate_physical = unnormalize(candidate_norm, bounds)
    next_diam = candidate_physical[0, 0].item()
    next_n_ap = int(round(candidate_physical[0, 1].item()))
    next_n_det1 = int(round(candidate_physical[0, 2].item()))
    next_n_det2 = int(round(candidate_physical[0, 3].item()))
    next_d2 = candidate_physical[0, 4].item()
    next_d3 = candidate_physical[0, 5].item()
    # n_det values must be even (2 crystals per cell)
    if next_n_det1 % 2 != 0:
        next_n_det1 += 1
    if next_n_det2 % 2 != 0:
        next_n_det2 += 1

    # Final feasibility check (belt and suspenders)
    if not is_feasible(next_diam, next_n_ap):
        logger.warning(f"Candidate infeasible (d={next_diam:.4f}, n={next_n_ap}), clamping aperture_diam")
        next_diam = min(next_diam, SAFETY_MARGIN * HR_CIRCUMFERENCE / next_n_ap - 0.01)
    if not is_ring_ordering_ok(next_d2, next_d3):
        logger.warning(f"Candidate violates ring ordering (d2={next_d2:.1f}, d3={next_d3:.1f}), "
                       f"pushing d3 out to keep the {MIN_RING_GAP_MM:.0f} mm gap")
        next_d3 = min(next_d2 + MIN_RING_GAP_MM, D4_INNER_MM - MIN_RING_GAP_MM)
        next_d2 = min(next_d2, next_d3 - MIN_RING_GAP_MM)
    if not is_ring_packing_ok(next_n_det1, next_n_det2, next_d2, next_d3):
        # Trim ring 2's crystal count to what actually fits, rounding down to an
        # even number. Better a slightly smaller ring than a geometry crash that
        # costs the whole iteration.
        max_nd2 = int(max_crystals_on_ring(next_d2))
        trimmed = min(next_n_det2, max_nd2 - 2)
        trimmed -= trimmed % 2
        logger.warning(f"Ring 2 packing infeasible (n_det2={next_n_det2} at d2={next_d2:.1f} mm, "
                       f"max {max_nd2}); trimming n_det_ring2 to {trimmed}")
        next_n_det2 = max(int(BOUNDS_MIN[3]), trimmed)

    # --- Deduplication: reject if too close to a previously tried config ---
    all_x = torch.tensor(df[PARAM_NAMES].dropna().values, dtype=torch.double)
    candidate_vec = torch.tensor([next_diam, float(next_n_ap), float(next_n_det1),
                                  float(next_n_det2), next_d2, next_d3],
                                 dtype=torch.double)
    # Normalize distances by parameter ranges to compare fairly
    ranges = bounds[1] - bounds[0]
    norm_dists = ((all_x - candidate_vec) / ranges).norm(dim=1)
    min_dist = norm_dists.min().item() if len(norm_dists) > 0 else 1.0

    if min_dist < 0.02:  # less than 2% of design space away = duplicate
        logger.warning(f"Candidate too close to existing config (dist={min_dist:.4f}), searching for diverse alternative")

        # Generate Sobol set, filter for feasibility + distance from all existing
        sobol_dedup = torch.quasirandom.SobolEngine(dimension=DIM, scramble=True)
        n_dedup = 2048
        X_sobol = sobol_dedup.draw(n_dedup, dtype=torch.double)
        # Unnormalize
        X_phys = X_sobol * ranges + bounds[0]

        # Filter feasible (aperture overlap, ring ordering, ring packing)
        feasible_mask = torch.tensor(
            [is_feasible_full(X_phys[i, 0].item(), X_phys[i, 1].item(),
                              X_phys[i, 2].item(), X_phys[i, 3].item(),
                              X_phys[i, 4].item(), X_phys[i, 5].item())
             for i in range(n_dedup)]
        )
        X_feasible = X_phys[feasible_mask]
        if len(X_feasible) == 0:
            logger.warning("No feasible candidates for dedup — keeping original")
        else:
            # Compute min distance from each candidate to all existing configs
            all_x_norm = (all_x - bounds[0]) / ranges
            X_feas_norm = (X_feasible - bounds[0]) / ranges
            # Pairwise distances
            dists_to_existing = torch.cdist(X_feas_norm, all_x_norm)  # (n_feasible, n_existing)
            min_dists = dists_to_existing.min(dim=1).values  # (n_feasible,)

            # Evaluate acquisition on candidates with min_dist >= 0.02
            diverse_mask = min_dists >= 0.02
            if diverse_mask.sum() == 0:
                # Relax threshold
                diverse_mask = min_dists >= 0.01
                logger.warning("Relaxed dedup threshold to 0.01")

            if diverse_mask.sum() > 0:
                X_diverse = X_feasible[diverse_mask]
                diverse_dists = min_dists[diverse_mask]

                # Normalize for acquisition evaluation
                X_diverse_norm = ((X_diverse - bounds[0]) / ranges).unsqueeze(1)
                n_eval = min(len(X_diverse_norm), 256)
                with torch.no_grad():
                    acq_vals = acqf(X_diverse_norm[:n_eval])
                # Pick candidate with best acquisition value
                best_idx = acq_vals.argmax()
                best_cand = X_diverse[best_idx]

                next_diam = best_cand[0].item()
                next_n_ap = int(round(best_cand[1].item()))
                next_n_det1 = int(round(best_cand[2].item()))
                next_n_det2 = int(round(best_cand[3].item()))
                next_d2 = best_cand[4].item()
                next_d3 = best_cand[5].item()
                if next_n_det1 % 2 != 0:
                    next_n_det1 += 1
                if next_n_det2 % 2 != 0:
                    next_n_det2 += 1
                logger.info(f"Dedup: selected diverse candidate (acq={acq_vals[best_idx]:.4f}, "
                            f"dist={diverse_dists[best_idx]:.4f}): "
                            f"d={next_diam:.4f} n={next_n_ap} nd1={next_n_det1} "
                            f"nd2={next_n_det2} d2={next_d2:.1f} d3={next_d3:.1f}")
            else:
                logger.warning("No diverse candidates found — keeping original")

    logger.info(f"Acquisition value: {acq_value.item():.6f}")
    logger.info(f"SUGGESTION -> aperture_diam={next_diam:.4f} mm | "
                f"n_apertures={next_n_ap} | "
                f"n_det_ring1={next_n_det1} | n_det_ring2={next_n_det2} | "
                f"d2_inner={next_d2:.1f} mm | d3_inner={next_d3:.1f} mm")

    return next_diam, next_n_ap, next_n_det1, next_n_det2, next_d2, next_d3


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="MOBO Agent for SAI SC-SPECT")
    parser.add_argument("--results_csv", type=str, required=True,
                        help="Path to results CSV with all OBJ_COLUMNS "
                             "(fwhm_mean, asci_pct, sensitivity_mean, mpxi_mean, cnr_mean)")
    args = parser.parse_args()

    diam, n_ap, n_det1, n_det2, d2_inner, d3_inner = get_next_candidate(args.results_csv)
    print(f"\nSuggested next config:")
    print(f"  aperture_diam   = {diam:.4f} mm")
    print(f"  n_apertures     = {n_ap}")
    print(f"  n_det_ring1     = {n_det1}")
    print(f"  n_det_ring2     = {n_det2}")
    print(f"  d2_inner        = {d2_inner:.1f} mm")
    print(f"  d3_inner        = {d3_inner:.1f} mm")
