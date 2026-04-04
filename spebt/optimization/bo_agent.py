#!/usr/bin/env python3
"""
Bayesian Optimization Agent for SAI SC-SPECT.

Adapted from Kirtiraj's bo_agent.py for 3D design space:
  - Design vector: (aperture_diam_mm, a_mm, b_mm)
  - SingleTaskGP with Matérn 5/2 + ARD
  - qExpectedImprovement for batch BO (q candidates per round)

Usage:
  from bo_agent import get_next_candidates
  candidates = get_next_candidates(results_csv, q=8)
"""
import os
import logging
import warnings
import torch
import pandas as pd
import numpy as np
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.utils.transforms import normalize, unnormalize
from botorch.acquisition.monte_carlo import qExpectedImprovement
from botorch.optim import optimize_acqf
from gpytorch.mlls import ExactMarginalLogLikelihood

# --- Logging ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("BO_Agent_SAI")
warnings.filterwarnings("ignore", category=UserWarning)

# --- Design space bounds ---
# (aperture_diam_mm, a_mm, b_mm)
PARAM_NAMES = ["aperture_diam_mm", "a_mm", "b_mm"]
BOUNDS_MIN = [0.2, 0.1, 0.1]
BOUNDS_MAX = [1.0, 1.0, 1.0]
DIM = len(PARAM_NAMES)


def load_training_data(results_csv: str):
    """
    Load evaluated configs from results CSV.
    Returns (train_x, train_y) as torch tensors.
    """
    if not os.path.exists(results_csv):
        raise FileNotFoundError(f"Results file not found: {results_csv}")

    df = pd.read_csv(results_csv)

    # Filter valid results (JI > 0 and not NaN)
    df = df.dropna(subset=["JI"])
    df = df[df["JI"] > 1e-12]

    n_points = len(df)
    logger.info(f"Loaded {n_points} valid data points from {results_csv}")

    if n_points < 5:
        logger.warning("Very few data points (<5). GP may be unstable.")

    train_x = torch.tensor(
        df[PARAM_NAMES].values, dtype=torch.double
    )
    train_y = torch.tensor(
        df[["JI"]].values, dtype=torch.double
    )

    return train_x, train_y, df


def fit_gp(train_x: torch.Tensor, train_y: torch.Tensor):
    """
    Fit SingleTaskGP with Matérn 5/2 + ARD on normalized/standardized data.
    Returns (gp_model, train_x_norm, train_y_std, bounds, y_mean, y_std).
    """
    bounds = torch.tensor([BOUNDS_MIN, BOUNDS_MAX], dtype=torch.double)

    # Normalize inputs to [0, 1]
    train_x_norm = normalize(train_x, bounds)

    # Standardize outputs
    y_mean = train_y.mean()
    y_std = train_y.std() + 1e-6
    train_y_std = (train_y - y_mean) / y_std

    logger.info("Fitting SingleTaskGP (Matérn 5/2 + ARD)...")
    gp = SingleTaskGP(train_x_norm, train_y_std)
    mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
    fit_gpytorch_mll(mll)
    logger.info("GP fitted successfully.")

    # Log learned lengthscales (ARD)
    try:
        ls = gp.covar_module.base_kernel.lengthscale.detach().cpu().numpy().flatten()
        for name, l in zip(PARAM_NAMES, ls):
            logger.info(f"  ARD lengthscale {name}: {l:.4f}")
    except Exception:
        pass

    return gp, train_x_norm, train_y_std, bounds, y_mean, y_std


def get_next_candidates(
    results_csv: str,
    q: int = 8,
    num_restarts: int = 10,
    raw_samples: int = 512,
    mc_samples: int = 256,
):
    """
    Run one BO step: fit GP on existing data, optimize qEI, return q candidates.

    Args:
        results_csv: Path to results CSV with columns [aperture_diam_mm, a_mm, b_mm, JI]
        q: Number of candidates to propose (batch size)
        num_restarts: Number of L-BFGS-B restarts for acquisition optimization
        raw_samples: Number of random candidates for initialization
        mc_samples: Number of MC samples for qEI

    Returns:
        candidates: np.ndarray of shape (q, 3) in physical units
        metadata: dict with GP stats and acquisition values
    """
    logger.info(f"--- BO Step: proposing {q} candidates ---")

    # 1. Load data
    train_x, train_y, df = load_training_data(results_csv)
    current_best_ji = train_y.max().item()
    logger.info(f"Current best JI: {current_best_ji:.6e}")

    # 2. Fit GP
    gp, train_x_norm, train_y_std, bounds, y_mean, y_std = fit_gp(train_x, train_y)

    # 3. Optimize qEI
    logger.info(f"Optimizing qEI (q={q}, restarts={num_restarts})...")
    qEI = qExpectedImprovement(
        model=gp,
        best_f=train_y_std.max(),
        num_samples=mc_samples,
    )

    bounds_norm = torch.stack(
        [torch.zeros(DIM, dtype=torch.double),
         torch.ones(DIM, dtype=torch.double)]
    )

    candidates_norm, acq_values = optimize_acqf(
        acq_function=qEI,
        bounds=bounds_norm,
        q=q,
        num_restarts=num_restarts,
        raw_samples=raw_samples,
    )

    # 4. Un-normalize to physical units
    candidates_physical = unnormalize(candidates_norm, bounds)
    candidates_np = candidates_physical.detach().cpu().numpy()

    logger.info(f"Acquisition value: {acq_values.item():.6e}")
    for i in range(q):
        c = candidates_np[i]
        logger.info(
            f"  Candidate {i}: aperture_diam={c[0]:.4f}mm  "
            f"a={c[1]:.4f}mm  b={c[2]:.4f}mm"
        )

    metadata = {
        "n_training_points": len(train_x),
        "current_best_ji": current_best_ji,
        "acq_value": acq_values.item(),
        "y_mean": y_mean.item(),
        "y_std": y_std.item(),
    }

    return candidates_np, metadata


def evaluate_gp_r2(results_csv: str, test_frac: float = 0.2, seed: int = 42):
    """
    Evaluate GP R² on held-out data.
    Splits data into train/test, fits GP on train, predicts on test, returns R².
    """
    train_x, train_y, df = load_training_data(results_csv)
    n = len(train_x)
    n_test = max(1, int(n * test_frac))

    rng = np.random.RandomState(seed)
    perm = rng.permutation(n)
    test_idx = perm[:n_test]
    train_idx = perm[n_test:]

    x_tr, y_tr = train_x[train_idx], train_y[train_idx]
    x_te, y_te = train_x[test_idx], train_y[test_idx]

    gp, _, _, bounds, y_mean, y_std = fit_gp(x_tr, y_tr)

    # Predict on test set
    x_te_norm = normalize(x_te, bounds)
    gp.eval()
    with torch.no_grad():
        posterior = gp.posterior(x_te_norm)
        pred_std = posterior.mean * y_std + y_mean  # un-standardize

    # R² score
    ss_res = ((y_te - pred_std) ** 2).sum().item()
    ss_tot = ((y_te - y_te.mean()) ** 2).sum().item()
    r2 = 1.0 - ss_res / (ss_tot + 1e-12)

    logger.info(f"GP R² on held-out ({n_test}/{n} points): {r2:.4f}")
    return r2


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="BO Agent for SAI SC-SPECT")
    parser.add_argument("--results_csv", type=str, required=True,
                        help="Path to results_summary.csv")
    parser.add_argument("--q", type=int, default=8, help="Batch size")
    parser.add_argument("--eval_r2", action="store_true",
                        help="Evaluate GP R² instead of proposing candidates")
    args = parser.parse_args()

    if args.eval_r2:
        r2 = evaluate_gp_r2(args.results_csv)
        print(f"GP R² = {r2:.4f}")
    else:
        candidates, meta = get_next_candidates(args.results_csv, q=args.q)
        print(f"\nProposed {args.q} candidates:")
        for i, c in enumerate(candidates):
            print(f"  [{i}] aperture_diam={c[0]:.4f}  a={c[1]:.4f}  b={c[2]:.4f}")
