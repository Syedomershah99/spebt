#!/usr/bin/env python3
"""
Generate Latin Hypercube Sampling (LHS) configurations for 3D MOBO.

Design vector: (aperture_diam_mm, n_apertures, n_det_ring1)
  - aperture_diam_mm  ∈ [0.2, 1.0]   (collimator aperture diameter)
  - n_apertures        ∈ [60, 360]    (number of apertures on HR ring)
  - n_det_ring1        ∈ [120, 480]   (crystals on detector ring 1, must be even)

Output: configs_manifest_3d.csv
"""
import argparse
import os
import numpy as np
from scipy.stats.qmc import LatinHypercube


BOUNDS_MIN = np.array([0.2, 60.0, 120.0])
BOUNDS_MAX = np.array([1.0, 360.0, 480.0])
PARAM_NAMES = ["aperture_diam_mm", "n_apertures", "n_det_ring1"]
DIM = len(PARAM_NAMES)


def generate_lhs_configs(n_samples: int, seed: int = 42) -> np.ndarray:
    """Generate n_samples configs via LHS in 3D, scaled to physical bounds."""
    sampler = LatinHypercube(d=DIM, seed=seed)
    unit_samples = sampler.random(n=n_samples)
    configs = BOUNDS_MIN + unit_samples * (BOUNDS_MAX - BOUNDS_MIN)
    # Round n_apertures to nearest int
    configs[:, 1] = np.round(configs[:, 1]).astype(int)
    # Round n_det_ring1 to nearest even int
    configs[:, 2] = (np.round(configs[:, 2] / 2) * 2).astype(int)
    return configs


def main():
    parser = argparse.ArgumentParser(description="Generate LHS configs for 3D MOBO")
    parser.add_argument("--n_samples", type=int, default=25,
                        help="Number of LHS samples (default: 25)")
    parser.add_argument("--seed", type=int, default=2026,
                        help="Random seed for reproducibility")
    parser.add_argument("--output_dir", type=str, default="results",
                        help="Output directory for manifest CSV")
    parser.add_argument("--base_work_dir", type=str,
                        default="/vscratch/grp-rutaoyao/Omer/spebt/spebt/optimization/results",
                        help="Base work directory on HPC")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    configs = generate_lhs_configs(args.n_samples, args.seed)

    out_path = os.path.join(args.output_dir, "configs_manifest_3d.csv")
    with open(out_path, "w") as f:
        f.write("idx,aperture_diam_mm,n_apertures,n_det_ring1,work_dir\n")
        for i, row in enumerate(configs):
            ap_d, n_ap, n_det = row
            n_ap_int = int(n_ap)
            n_det_int = int(n_det)
            work_dir = os.path.join(
                args.base_work_dir,
                f"lhs3d_{i:04d}_ap{ap_d:.4f}_nap{n_ap_int}_ndet{n_det_int}"
            )
            f.write(f"{i},{ap_d:.6f},{n_ap_int},{n_det_int},{work_dir}\n")

    print(f"Generated {args.n_samples} LHS configs -> {out_path}")
    print(f"Bounds: {dict(zip(PARAM_NAMES, zip(BOUNDS_MIN, BOUNDS_MAX)))}")
    print(f"\nSample statistics:")
    for j, name in enumerate(PARAM_NAMES):
        col = configs[:, j]
        print(f"  {name}: min={col.min():.4f}  max={col.max():.4f}  mean={col.mean():.4f}")


if __name__ == "__main__":
    main()
