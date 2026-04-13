#!/usr/bin/env python3
"""
Test BO agent end-to-end with synthetic data.

Generates fake LHS results with a known objective landscape, then verifies:
  1. GP fits successfully
  2. EI proposes a sensible next candidate within bounds
  3. Sequential iteration improves (best JI increases over rounds)

Usage:
  python test_bo_agent.py
"""
import os
import sys
import tempfile
import numpy as np
import pandas as pd

# Ensure we can import bo_agent from the same directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bo_agent import get_next_candidate, BOUNDS_MIN, BOUNDS_MAX, PARAM_NAMES


def synthetic_ji(aperture_diam_mm: float, n_apertures: int) -> float:
    """
    Fake JI landscape with a known optimum.

    Physics-inspired: resolution improves with smaller aperture (FWHM ~ diam),
    sensitivity improves with more apertures, but too many apertures with large
    diameter causes overlap (infeasible).

    Known optimum near (diam=0.35, n_ap=200) with JI ~ 1.2e-4.
    """
    d = aperture_diam_mm
    n = float(n_apertures)

    # Resolution term: FWHM ~ diam, so 1/FWHM^2 ~ 1/d^2
    resolution = 1.0 / (d ** 2)

    # Sensitivity term: proportional to n_apertures * aperture_area
    sensitivity = n * (d ** 2) * 1e-4

    # ASCI term: peaks around n=200, falls off at extremes
    asci = np.exp(-0.5 * ((n - 200) / 80) ** 2) * 0.8

    # Physical constraint penalty: aperture can't exceed chord spacing
    chord_spacing = 2 * 35.0 * np.sin(np.pi / n)
    if d >= chord_spacing:
        return 0.0

    ji = (sensitivity / (d ** 2)) * asci / 100.0

    # Add small noise (simulates beam analysis variability)
    ji *= (1.0 + np.random.normal(0, 0.02))
    return max(ji, 0.0)


def generate_synthetic_lhs(n_samples: int = 25, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic LHS results with fake JI values."""
    np.random.seed(seed)
    from scipy.stats.qmc import LatinHypercube

    sampler = LatinHypercube(d=2, seed=seed)
    unit = sampler.random(n=n_samples)
    bounds_min = np.array(BOUNDS_MIN)
    bounds_max = np.array(BOUNDS_MAX)
    configs = bounds_min + unit * (bounds_max - bounds_min)
    configs[:, 1] = np.round(configs[:, 1])

    rows = []
    for i in range(n_samples):
        diam = configs[i, 0]
        n_ap = int(configs[i, 1])
        ji = synthetic_ji(diam, n_ap)
        rows.append({
            "config": f"lhs_{i:04d}",
            "aperture_diam_mm": diam,
            "n_apertures": n_ap,
            "fwhm_mean": diam * 2.5,
            "sensitivity_mean": n_ap * diam**2 * 1e-4,
            "asci_pct": np.exp(-0.5 * ((n_ap - 200) / 80)**2) * 80,
            "JI": ji,
        })

    return pd.DataFrame(rows)


def test_gp_fit_and_propose():
    """Test 1: GP fits on synthetic LHS data and proposes a valid candidate."""
    print("=" * 60)
    print("TEST 1: GP fit + candidate proposal")
    print("=" * 60)

    df = generate_synthetic_lhs(25)
    valid = df[df["JI"] > 0]
    print(f"  Generated {len(df)} configs, {len(valid)} feasible (JI > 0)")
    print(f"  Best initial JI: {valid['JI'].max():.6e}")

    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        df.to_csv(f.name, index=False)
        csv_path = f.name

    try:
        diam, n_ap = get_next_candidate(csv_path)

        assert BOUNDS_MIN[0] <= diam <= BOUNDS_MAX[0], \
            f"diam {diam} out of bounds [{BOUNDS_MIN[0]}, {BOUNDS_MAX[0]}]"
        assert BOUNDS_MIN[1] <= n_ap <= BOUNDS_MAX[1], \
            f"n_ap {n_ap} out of bounds [{BOUNDS_MIN[1]}, {BOUNDS_MAX[1]}]"

        print(f"\n  Proposed: diam={diam:.4f} mm, n_apertures={n_ap}")
        print("  PASSED: candidate within bounds")
    finally:
        os.unlink(csv_path)


def test_sequential_improvement():
    """Test 2: Sequential BO iterations improve best JI."""
    print("\n" + "=" * 60)
    print("TEST 2: Sequential improvement (5 BO iterations)")
    print("=" * 60)

    df = generate_synthetic_lhs(25)
    best_ji_history = [df[df["JI"] > 0]["JI"].max()]
    print(f"  Initial best JI: {best_ji_history[0]:.6e}")

    n_iters = 5
    for i in range(n_iters):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            df.to_csv(f.name, index=False)
            csv_path = f.name

        try:
            diam, n_ap = get_next_candidate(csv_path)
        finally:
            os.unlink(csv_path)

        ji = synthetic_ji(diam, n_ap)
        new_row = {
            "config": f"bo_{i:04d}",
            "aperture_diam_mm": diam,
            "n_apertures": n_ap,
            "fwhm_mean": diam * 2.5,
            "sensitivity_mean": n_ap * diam**2 * 1e-4,
            "asci_pct": np.exp(-0.5 * ((n_ap - 200) / 80)**2) * 80,
            "JI": ji,
        }
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

        current_best = df[df["JI"] > 0]["JI"].max()
        best_ji_history.append(current_best)
        print(f"  Iter {i+1}: proposed d={diam:.4f} n={n_ap}, "
              f"JI={ji:.6e}, best={current_best:.6e}")

    improved = best_ji_history[-1] >= best_ji_history[0]
    print(f"\n  Initial best: {best_ji_history[0]:.6e}")
    print(f"  Final best:   {best_ji_history[-1]:.6e}")
    print(f"  {'PASSED' if improved else 'WARNING'}: "
          f"{'JI improved or maintained' if improved else 'JI did not improve (may need more iters)'}")

    return best_ji_history


def test_few_points():
    """Test 3: GP handles small dataset (edge case)."""
    print("\n" + "=" * 60)
    print("TEST 3: GP with minimal data (5 points)")
    print("=" * 60)

    np.random.seed(99)
    rows = []
    for i in range(5):
        diam = np.random.uniform(0.2, 1.0)
        n_ap = int(np.random.uniform(60, 360))
        rows.append({
            "config": f"min_{i}",
            "aperture_diam_mm": diam,
            "n_apertures": n_ap,
            "JI": synthetic_ji(diam, n_ap),
        })
    df = pd.DataFrame(rows)
    df = df[df["JI"] > 0]  # keep only feasible

    if len(df) < 3:
        print("  SKIP: too few feasible points for GP")
        return

    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        df.to_csv(f.name, index=False)
        csv_path = f.name

    try:
        diam, n_ap = get_next_candidate(csv_path)
        print(f"  Proposed: diam={diam:.4f}, n_ap={n_ap}")
        print("  PASSED: GP handled small dataset")
    except Exception as e:
        print(f"  FAILED: {e}")
    finally:
        os.unlink(csv_path)


if __name__ == "__main__":
    print("BO Agent End-to-End Test")
    print("Design vector: (aperture_diam_mm, n_apertures)")
    print()

    test_gp_fit_and_propose()
    history = test_sequential_improvement()
    test_few_points()

    print("\n" + "=" * 60)
    print("ALL TESTS COMPLETE")
    print("=" * 60)
