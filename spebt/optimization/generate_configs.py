#!/usr/bin/env python3
"""
Generate Latin Hypercube Sampling (LHS) configurations for 2D BO.

Design vector: (aperture_diam_mm, n_apertures)
  - aperture_diam_mm ∈ [0.2, 1.0]  (hardware: aperture diameter)
  - n_apertures      ∈ [60, 360]   (hardware: number of apertures on HR ring)

Output: configs_manifest.csv
"""
import argparse
import os
import numpy as np
from scipy.stats.qmc import LatinHypercube


# Default bounds
BOUNDS_MIN = np.array([0.2, 60.0])    # aperture_diam, n_apertures
BOUNDS_MAX = np.array([1.0, 360.0])
PARAM_NAMES = ["aperture_diam_mm", "n_apertures"]


def generate_lhs_configs(n_samples: int, seed: int = 42) -> np.ndarray:
    """Generate n_samples configs via LHS in 2D, scaled to physical bounds."""
    sampler = LatinHypercube(d=2, seed=seed)
    unit_samples = sampler.random(n=n_samples)  # (n_samples, 2) in [0,1]
    configs = BOUNDS_MIN + unit_samples * (BOUNDS_MAX - BOUNDS_MIN)
    # Round n_apertures to nearest integer
    configs[:, 1] = np.round(configs[:, 1]).astype(int)
    return configs


def main():
    parser = argparse.ArgumentParser(description="Generate LHS configs for BO")
    parser.add_argument("--n_samples", type=int, default=25,
                        help="Number of LHS samples (default: 25)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--output_dir", type=str, default="results",
                        help="Output directory for manifest CSV")
    parser.add_argument("--base_work_dir", type=str,
                        default="/vscratch/grp-rutaoyao/Omer/spebt/optimization",
                        help="Base work directory on HPC for config output dirs")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    configs = generate_lhs_configs(args.n_samples, args.seed)

    out_path = os.path.join(args.output_dir, "configs_manifest.csv")
    with open(out_path, "w") as f:
        f.write("idx,aperture_diam_mm,n_apertures,work_dir\n")
        for i, (ap_d, n_ap) in enumerate(configs):
            n_ap_int = int(n_ap)
            work_dir = os.path.join(
                args.base_work_dir,
                f"config_{i:04d}_ap{ap_d:.4f}_nap{n_ap_int}"
            )
            f.write(f"{i},{ap_d:.6f},{n_ap_int},{work_dir}\n")

    print(f"Generated {args.n_samples} LHS configs → {out_path}")
    print(f"Bounds: {dict(zip(PARAM_NAMES, zip(BOUNDS_MIN, BOUNDS_MAX)))}")

    print(f"\nSample statistics:")
    for j, name in enumerate(PARAM_NAMES):
        col = configs[:, j]
        print(f"  {name}: min={col.min():.4f}  max={col.max():.4f}  mean={col.mean():.4f}")


if __name__ == "__main__":
    main()
