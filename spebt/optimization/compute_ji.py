#!/usr/bin/env python3
"""
Compute Joint Index (JI) for SAI SC-SPECT configurations.

Adapted from Kirtiraj's 6_calc_ji.py for SAI:
  - 200×200 FOV (not 280×280)
  - 16 HDF5 files per config (2 layouts × 8 T8 poses, not 15 rotations)
  - JI = (sensitivity_mean / FWHM²) × ASCI_pct / 100

Usage:
  python compute_ji.py --work_dir <path> --out_csv results/results_summary.csv --config_name config_0001
"""
import argparse
import os
import glob
import h5py
import numpy as np
import pandas as pd


# SAI constants
N_LAYOUTS = 2       # 2 collimator rotations (0° and 1°)
N_T8_POSES = 8      # 8 bed positions per layout
N_TOTAL_FILES = N_LAYOUTS * N_T8_POSES  # 16
FOV_NPIX = (200, 200)
ASCI_NBINS_ANGULAR = 360
TOTAL_ASCI_BINS = FOV_NPIX[0] * FOV_NPIX[1] * ASCI_NBINS_ANGULAR


def compute_sensitivity(work_dir: str):
    """
    Compute sensitivity by summing PPDFs across all 16 HDF5 files (2 layouts × 8 T8 poses).
    Returns (sensitivity_total, sensitivity_mean_per_file).
    """
    aggregated_ppdfs = None
    successful = 0

    # Match pattern: position_NNN_ppdfs_t8_PP.hdf5
    ppdf_pattern = os.path.join(work_dir, "position_*_ppdfs_t8_*.hdf5")
    ppdf_files = sorted(glob.glob(ppdf_pattern))

    if not ppdf_files:
        # Fallback: try Kirtiraj-style naming
        ppdf_pattern = os.path.join(work_dir, "scanner_layouts_*_layout_*_subvoxels.hdf5")
        ppdf_files = sorted(glob.glob(ppdf_pattern))

    for ppdf_file in ppdf_files:
        try:
            with h5py.File(ppdf_file, "r") as f:
                ppdfs = f["ppdfs"][:]
        except Exception as e:
            print(f"  [warn] Failed to read {ppdf_file}: {e}")
            continue

        if aggregated_ppdfs is None:
            aggregated_ppdfs = ppdfs.astype(np.float64)
        else:
            aggregated_ppdfs += ppdfs.astype(np.float64)
        successful += 1

    if aggregated_ppdfs is None or successful == 0:
        return np.nan, np.nan, 0

    # Sum over crystals (axis 0) → sensitivity per pixel
    per_pixel_sum = np.sum(aggregated_ppdfs, axis=0)
    sensitivity_total = float(np.mean(per_pixel_sum))
    sensitivity_mean = sensitivity_total / successful

    return sensitivity_total, sensitivity_mean, successful


def compute_fwhm_and_asci(work_dir: str):
    """
    Aggregate FWHM and ASCI from beam analysis outputs across all layouts.
    Returns (fwhm_mean, asci_pct).
    """
    all_fwhm_values = []
    combined_asci_hist = None

    # Search for beam properties and ASCI histogram files
    prop_files = sorted(glob.glob(os.path.join(work_dir, "beams_properties_configuration_*.hdf5")))
    asci_files = sorted(glob.glob(os.path.join(work_dir, "asci_histogram_*.hdf5")))

    # Also check for .pt ASCI files
    if not asci_files:
        asci_files = sorted(glob.glob(os.path.join(work_dir, "asci_histogram_*.pt")))

    # FWHM from beam properties
    for prop_file in prop_files:
        try:
            with h5py.File(prop_file, "r") as f:
                data = f["beam_properties"][:]
                if data.shape[0] > 0:
                    # Column 4 is FWHM (angle, width/FWHM, size, rel_sens, abs_sens)
                    fwhm_data = data[:, 4]
                    valid = fwhm_data[~np.isnan(fwhm_data)]
                    if len(valid) > 0:
                        all_fwhm_values.extend(valid.tolist())
        except Exception as e:
            print(f"  [warn] Failed to read {prop_file}: {e}")

    # ASCI from histogram files
    for asci_file in asci_files:
        try:
            if asci_file.endswith(".pt"):
                import torch
                hist = torch.load(asci_file, weights_only=True).numpy()
            else:
                with h5py.File(asci_file, "r") as f:
                    hist = f["asci_histogram"][:]

            if combined_asci_hist is None:
                combined_asci_hist = hist.astype(np.int64)
            else:
                combined_asci_hist += hist.astype(np.int64)
        except Exception as e:
            print(f"  [warn] Failed to read {asci_file}: {e}")

    # Compute averages
    fwhm_mean = float(np.mean(all_fwhm_values)) if all_fwhm_values else np.nan

    if combined_asci_hist is not None:
        asci_filled = np.count_nonzero(combined_asci_hist)
        asci_pct = (asci_filled / TOTAL_ASCI_BINS) * 100.0
    else:
        asci_pct = np.nan

    return fwhm_mean, asci_pct


def compute_ji(work_dir: str) -> dict:
    """
    Compute all metrics and JI for a single configuration.
    JI = (sensitivity_mean / FWHM²) × ASCI_pct / 100
    """
    sens_total, sens_mean, n_files = compute_sensitivity(work_dir)
    fwhm_mean, asci_pct = compute_fwhm_and_asci(work_dir)

    # JI formula
    ji = np.nan
    if (not np.isnan(fwhm_mean) and fwhm_mean > 0
            and not np.isnan(asci_pct)
            and not np.isnan(sens_mean)):
        ji = (sens_mean / (fwhm_mean ** 2)) * asci_pct / 100.0

    return {
        "fwhm_mean": fwhm_mean,
        "sensitivity_total": sens_total,
        "sensitivity_mean": sens_mean,
        "asci_pct": asci_pct,
        "n_ppdf_files": n_files,
        "JI": ji,
    }


def main():
    parser = argparse.ArgumentParser(description="Compute JI for SAI SC-SPECT config")
    parser.add_argument("--work_dir", type=str, required=True,
                        help="Directory containing PPDF HDF5 + beam analysis outputs")
    parser.add_argument("--out_csv", type=str, required=True,
                        help="Path to output CSV (appends if exists)")
    parser.add_argument("--config_name", type=str, default="config",
                        help="Config identifier for CSV row")
    # Optional: pass design parameters for CSV tracking
    parser.add_argument("--aperture_diam_mm", type=float, default=None)
    parser.add_argument("--n_apertures", type=int, default=None)
    parser.add_argument("--force_zero", action="store_true",
                        help="Write JI=0 row (for infeasible configs)")
    parser.add_argument("--reason", type=str, default="",
                        help="Reason for force_zero (logged)")
    args = parser.parse_args()

    if args.force_zero:
        results = {
            "fwhm_mean": float("nan"),
            "sensitivity_total": float("nan"),
            "sensitivity_mean": float("nan"),
            "asci_pct": float("nan"),
            "n_ppdf_files": 0,
            "JI": 0.0,
        }
        print(f"[{args.config_name}] FORCE_ZERO: {args.reason}")
    else:
        results = compute_ji(args.work_dir)
    results["config"] = args.config_name
    results["work_dir"] = args.work_dir

    if args.aperture_diam_mm is not None:
        results["aperture_diam_mm"] = args.aperture_diam_mm
    if args.n_apertures is not None:
        results["n_apertures"] = args.n_apertures

    # Append to CSV (create with header if new)
    df_new = pd.DataFrame([results])
    if os.path.exists(args.out_csv):
        df_new.to_csv(args.out_csv, mode="a", header=False, index=False)
    else:
        os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
        df_new.to_csv(args.out_csv, index=False)

    print(f"[{args.config_name}] FWHM={results['fwhm_mean']:.4f}  "
          f"ASCI={results['asci_pct']:.2f}%  "
          f"Sens={results['sensitivity_mean']:.4e}  "
          f"JI={results['JI']:.6e}  "
          f"({results['n_ppdf_files']} PPDF files)")


if __name__ == "__main__":
    main()
