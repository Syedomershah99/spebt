#!/usr/bin/env python3
"""
Generate Latin Hypercube Sampling (LHS) configurations for 3D BO.

Design vector: (aperture_diam_mm, a_mm, b_mm)
  - aperture_diam_mm ∈ [0.2, 1.0]  (hardware: aperture diameter)
  - a_mm             ∈ [0.1, 1.0]  (acquisition: T8 ellipse semi-axis X)
  - b_mm             ∈ [0.1, 1.0]  (acquisition: T8 ellipse semi-axis Y)

Output: configs_manifest.csv
"""
import argparse
import os
import numpy as np
from scipy.stats.qmc import LatinHypercube


# Default bounds (matching plan)
BOUNDS_MIN = np.array([0.2, 0.1, 0.1])   # aperture_diam, a, b
BOUNDS_MAX = np.array([1.0, 1.0, 1.0])
PARAM_NAMES = ["aperture_diam_mm", "a_mm", "b_mm"]


def generate_lhs_configs(n_samples: int, seed: int = 42) -> np.ndarray:
    """Generate n_samples configs via LHS in 3D, scaled to physical bounds."""
    sampler = LatinHypercube(d=3, seed=seed)
    unit_samples = sampler.random(n=n_samples)  # (n_samples, 3) in [0,1]
    # Scale to physical bounds
    configs = BOUNDS_MIN + unit_samples * (BOUNDS_MAX - BOUNDS_MIN)
    return configs


def main():
    parser = argparse.ArgumentParser(description="Generate LHS configs for BO")
    parser.add_argument("--n_samples", type=int, default=150,
                        help="Number of LHS samples (default: 150)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--output_dir", type=str, default="results",
                        help="Output directory for manifest CSV")
    parser.add_argument("--base_work_dir", type=str, default="/vscratch/grp-rutaoyao/Omer/spebt/optimization",
                        help="Base work directory on HPC for config output dirs")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    configs = generate_lhs_configs(args.n_samples, args.seed)

    # Write manifest CSV
    out_path = os.path.join(args.output_dir, "configs_manifest.csv")
    with open(out_path, "w") as f:
        f.write("idx,aperture_diam_mm,a_mm,b_mm,work_dir\n")
        for i, (ap_d, a, b) in enumerate(configs):
            work_dir = os.path.join(
                args.base_work_dir,
                f"config_{i:04d}_ap{ap_d:.4f}_a{a:.4f}_b{b:.4f}"
            )
            f.write(f"{i},{ap_d:.6f},{a:.6f},{b:.6f},{work_dir}\n")

    print(f"Generated {args.n_samples} LHS configs → {out_path}")
    print(f"Bounds: {dict(zip(PARAM_NAMES, zip(BOUNDS_MIN, BOUNDS_MAX)))}")

    # Print summary statistics
    print(f"\nSample statistics:")
    for j, name in enumerate(PARAM_NAMES):
        col = configs[:, j]
        print(f"  {name}: min={col.min():.4f}  max={col.max():.4f}  mean={col.mean():.4f}")


if __name__ == "__main__":
    main()
