#!/usr/bin/env python3
"""
Compute metrics for SAI SC-SPECT configurations.

Metrics: FWHM, ASCI, sensitivity, MPXI
  - 200×200 FOV (0.05 mm/px)
  - 16 HDF5 files per config (2 layouts × 8 T8 poses)

Usage:
  python compute_metrics.py --work_dir <path> --out_csv results/results_summary_mobo.csv --config_name config_0001
"""
import argparse
import math
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
MM_PER_PX = 0.05    # 200 px x 0.05 mm = 10 mm FOV (matches recon)
ASCI_NBINS_ANGULAR = 360
TOTAL_ASCI_BINS = FOV_NPIX[0] * FOV_NPIX[1] * ASCI_NBINS_ANGULAR

# Precompute pixel-coordinate arrays once (flattened to match PPDF/mask layout)
_PX_ROW = np.repeat(np.arange(FOV_NPIX[0]), FOV_NPIX[1]).astype(np.float64) * MM_PER_PX
_PX_COL = np.tile(np.arange(FOV_NPIX[1]), FOV_NPIX[0]).astype(np.float64) * MM_PER_PX


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


def compute_mpxi(work_dir: str):
    """
    Compute mean MPXI (multiplexing index) from beam mask files.

    For each detector, counts unique non-zero beam IDs in its mask row (= k).
    Returns mean(k) across all detectors and both layouts.
    Lower is better (less signal ambiguity).

    Masks are already T8-aggregated (PPDFs summed before mask extraction).
    """
    mask_files = sorted(glob.glob(os.path.join(work_dir, "beams_masks_configuration_*.hdf5")))
    if not mask_files:
        return np.nan

    all_k = []
    for mask_file in mask_files:
        try:
            with h5py.File(mask_file, "r") as f:
                masks = f["beam_mask"][:]  # (n_det, n_pix)
            # Per detector: count unique non-zero beam IDs
            for row in masks:
                unique_ids = np.unique(row)
                k = int(np.count_nonzero(unique_ids))  # exclude 0 (background)
                all_k.append(k)
        except Exception as e:
            print(f"  [warn] Failed to read {mask_file}: {e}")

    if not all_k:
        return np.nan

    return float(np.mean(all_k))


def _per_beam_radial_fwhm(mask_row, ppdf_row, beam_id, angle):
    """
    Measure the radial FWHM of a single beam in one PPDF row.

    The beam's "Angle (rad)" stored in beam_properties is the direction from
    the beam's weighted centre to the detector centre, i.e. the radial
    direction of the beam in the FOV. We:
      1. Take the pixels labelled with this beam_id in mask_row.
      2. Project their (x, y) positions onto the unit vector (cos a, sin a).
      3. Compute the PPDF-weighted variance of that 1D projection.
      4. Return 2.355 * sigma (FWHM under Gaussian approximation).

    Returns 0.0 if the beam is too small or its weighted variance is non-positive.
    """
    pix_idx = np.flatnonzero(mask_row == beam_id)
    if pix_idx.size < 3:
        return 0.0
    r = _PX_COL[pix_idx] * math.cos(angle) + _PX_ROW[pix_idx] * math.sin(angle)
    w = ppdf_row[pix_idx].astype(np.float64)
    wsum = w.sum()
    if wsum <= 0.0:
        return 0.0
    mean_r = np.dot(w, r) / wsum
    var_r = np.dot(w, (r - mean_r) ** 2) / wsum
    if var_r <= 0.0:
        return 0.0
    return 2.355 * math.sqrt(var_r)


def compute_ppds(work_dir: str) -> float:
    """
    Compute PPDS (Projection Probability Density Sensitivity), following the
    SPEBT project-strategy document (Eq. 6):

        PPDS_j = sum_i { PPDF_{i,j} / sum_b V_{i,b} }   for i where PPDF_{i,j} > 0

    V_{i,b} = FWHM_tangential * FWHM_radial (2D form from the strategy doc).
    FWHM_tangential is the per-beam FWHM stored in beam_properties (column 4).
    FWHM_radial is measured here from the beam mask, by projecting the beam's
    masked pixels onto the radial direction (cos(angle), sin(angle)) and
    fitting a Gaussian-equivalent FWHM (2.355 * weighted sigma) using the
    PPDF intensities as weights.

    Earlier implementations used (a) the beam-mask pixel area or (b) a
    circular-beam approximation V ≈ FWHM_tang^2; both saturated or
    under-penalised wide-beam configurations and did not correlate with CNR.
    This version respects the elongated geometry of pinhole projections and
    matches the strategy-doc formula exactly.

    PPDFs are aggregated per layout (across the 8 T8 sub-poses) and matched
    to that layout's beams_properties_*.hdf5 and beams_masks_*.hdf5 by the
    3-digit layout id embedded in each filename. PPDS is then averaged
    across layouts.

    Returns mean PPDS over the FOV. Returns NaN if any required file is missing.
    """
    ppdf_files = sorted(glob.glob(os.path.join(work_dir, "position_*_ppdfs_t8_*.hdf5")))
    prop_files = sorted(glob.glob(os.path.join(work_dir, "beams_properties_configuration_*.hdf5")))
    mask_files = sorted(glob.glob(os.path.join(work_dir, "beams_masks_configuration_*.hdf5")))
    if not ppdf_files or not prop_files or not mask_files:
        return float("nan")

    def _layout_id(path):
        for part in os.path.basename(path).replace(".hdf5", "").split("_"):
            if part.isdigit() and len(part) == 3:
                return part
        return None

    # Aggregate T8 PPDFs per layout
    layout_ppdfs = {}
    for f in ppdf_files:
        lid = _layout_id(f)
        if lid is None:
            continue
        try:
            with h5py.File(f, "r") as h:
                arr = h["ppdfs"][:].astype(np.float64)
        except Exception as e:
            print(f"  [warn] PPDS: failed to read {f}: {e}")
            continue
        if lid in layout_ppdfs:
            layout_ppdfs[lid] += arr
        else:
            layout_ppdfs[lid] = arr
    if not layout_ppdfs:
        return float("nan")

    # Index prop and mask files by layout id (so we can pair them with PPDFs)
    prop_by_lid = {}
    for f in prop_files:
        lid = _layout_id(f)
        if lid is not None:
            prop_by_lid[lid] = f
    mask_by_lid = {}
    for f in mask_files:
        lid = _layout_id(f)
        if lid is not None:
            mask_by_lid[lid] = f

    ppds_per_layout = []
    for lid, ppdfs in layout_ppdfs.items():
        prop_file = prop_by_lid.get(lid) or (
            next(iter(prop_by_lid.values())) if len(prop_by_lid) == 1 else None
        )
        mask_file = mask_by_lid.get(lid) or (
            next(iter(mask_by_lid.values())) if len(mask_by_lid) == 1 else None
        )
        if prop_file is None or mask_file is None:
            continue
        try:
            with h5py.File(prop_file, "r") as h:
                bp = h["beam_properties"][:]
            with h5py.File(mask_file, "r") as h:
                masks = h["beam_mask"][:]                          # (n_det, n_pix)
        except Exception as e:
            print(f"  [warn] PPDS: failed reading prop/mask for layout {lid}: {e}")
            continue

        n_det, n_pix = ppdfs.shape
        if masks.shape[0] != n_det or masks.shape[1] != n_pix:
            print(f"  [warn] PPDS: shape mismatch ppdfs={ppdfs.shape}, masks={masks.shape}")
            continue
        if bp.shape[0] == 0:
            continue

        det_ids   = bp[:, 1].astype(np.int64)
        beam_ids  = bp[:, 2].astype(np.int64)
        angles    = bp[:, 3].astype(np.float64)
        fwhms     = bp[:, 4].astype(np.float64)

        # Auto-detect 1-indexed detector ids
        if det_ids.size > 0 and det_ids.min() >= 1 and det_ids.max() <= n_det:
            det_ids = det_ids - 1

        # (det_id, beam_id) -> (fwhm_tang, angle); skip invalid entries
        info = {}
        for d, b, a, f_ in zip(det_ids, beam_ids, angles, fwhms):
            if not (0 <= d < n_det):
                continue
            if not (np.isfinite(a) and np.isfinite(f_) and f_ > 0.0):
                continue
            info[(int(d), int(b))] = (float(f_), float(a))

        # Per-detector: sum V_{i,b} = FWHM_tang * FWHM_rad over its beams
        sumV = np.zeros(n_det, dtype=np.float64)
        for det_i in range(n_det):
            mask_row = masks[det_i, :]
            unique_beams = np.unique(mask_row[mask_row > 0])
            if unique_beams.size == 0:
                continue
            ppdf_row = ppdfs[det_i, :]
            for b in unique_beams:
                key = (det_i, int(b))
                params = info.get(key)
                if params is None:
                    continue
                fwhm_t, angle = params
                fwhm_r = _per_beam_radial_fwhm(mask_row, ppdf_row, int(b), angle)
                if fwhm_r > 0.0:
                    sumV[det_i] += fwhm_t * fwhm_r

        valid = sumV > 0
        if not valid.any():
            continue
        denom = np.where(valid, sumV, 1.0)
        weighted = ppdfs / denom[:, None]                          # (n_det, n_pix)
        weighted[~valid, :] = 0.0
        ppds_j = weighted.sum(axis=0)                              # (n_pix,)
        ppds_per_layout.append(float(ppds_j.mean()))

    return float(np.mean(ppds_per_layout)) if ppds_per_layout else float("nan")


def compute_metrics(work_dir: str) -> dict:
    """
    Compute all metrics for a single configuration.
    Returns: fwhm_mean, sensitivity_total, sensitivity_mean, asci_pct,
             mpxi_mean, ppds_mean, n_ppdf_files.
    """
    sens_total, sens_mean, n_files = compute_sensitivity(work_dir)
    fwhm_mean, asci_pct = compute_fwhm_and_asci(work_dir)
    mpxi_mean = compute_mpxi(work_dir)
    ppds_mean = compute_ppds(work_dir)

    return {
        "fwhm_mean": fwhm_mean,
        "sensitivity_total": sens_total,
        "sensitivity_mean": sens_mean,
        "asci_pct": asci_pct,
        "n_ppdf_files": n_files,
        "mpxi_mean": mpxi_mean,
        "ppds_mean": ppds_mean,
    }


def main():
    parser = argparse.ArgumentParser(description="Compute metrics for SAI SC-SPECT config")
    parser.add_argument("--work_dir", type=str, required=True,
                        help="Directory containing PPDF HDF5 + beam analysis outputs")
    parser.add_argument("--out_csv", type=str, required=True,
                        help="Path to output CSV (appends if exists)")
    parser.add_argument("--config_name", type=str, default="config",
                        help="Config identifier for CSV row")
    # Design parameters for CSV tracking
    parser.add_argument("--aperture_diam_mm", type=float, default=None)
    parser.add_argument("--n_apertures", type=int, default=None)
    parser.add_argument("--n_det_ring1", type=int, default=None)
    parser.add_argument("--n_det_ring2", type=int, default=None)
    parser.add_argument("--force_zero", action="store_true",
                        help="Write NaN row (for infeasible configs)")
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
            "mpxi_mean": float("nan"),
            "ppds_mean": float("nan"),
        }
        print(f"[{args.config_name}] FORCE_ZERO: {args.reason}")
    else:
        results = compute_metrics(args.work_dir)
    results["config"] = args.config_name
    results["work_dir"] = args.work_dir

    if args.aperture_diam_mm is not None:
        results["aperture_diam_mm"] = args.aperture_diam_mm
    if args.n_apertures is not None:
        results["n_apertures"] = args.n_apertures
    if args.n_det_ring1 is not None:
        results["n_det_ring1"] = args.n_det_ring1
    if args.n_det_ring2 is not None:
        results["n_det_ring2"] = args.n_det_ring2

    # Append to CSV (create with header if new). We MUST align to the
    # existing header's column order — the CSV may have extra columns
    # (added by later backfills, e.g. cnr_mean) that this script doesn't
    # produce, and its own results-dict order may differ from the header.
    # Writing with mode="a" without alignment produces shifted rows that
    # crash pandas readers downstream.
    df_new = pd.DataFrame([results])
    if os.path.exists(args.out_csv):
        existing_cols = pd.read_csv(args.out_csv, nrows=0).columns.tolist()
        for col in existing_cols:
            if col not in df_new.columns:
                df_new[col] = float("nan")
        df_new = df_new.reindex(columns=existing_cols)
        df_new.to_csv(args.out_csv, mode="a", header=False, index=False)
    else:
        os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
        df_new.to_csv(args.out_csv, index=False)

    print(f"[{args.config_name}] FWHM={results['fwhm_mean']:.4f}  "
          f"ASCI={results['asci_pct']:.2f}%  "
          f"Sens={results['sensitivity_mean']:.4e}  "
          f"MPXI={results['mpxi_mean']:.4f}  "
          f"PPDS={results['ppds_mean']:.4e}  "
          f"({results['n_ppdf_files']} PPDF files)")


if __name__ == "__main__":
    main()
