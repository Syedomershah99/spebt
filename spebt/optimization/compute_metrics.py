#!/usr/bin/env python3
"""
Compute metrics for SAI SC-SPECT configurations.

Metrics: FWHM, ASCI, sensitivity_mean, sensitivity_total, MPXI, PPDS
  - 200×200 FOV (0.05 mm/px)
  - 16 HDF5 files per config (2 layouts × 8 T8 poses)
  - PPDS is available in compute_metrics but is NOT currently used as a MOBO
    objective (see mobo_agent.py docstring for the history). CNR is added by
    a separate step (compute_cnr.py) invoked after this script.

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


def check_layout_completeness(work_dir: str) -> list:
    """Return a list of complaints if any per-layout beam file is missing.

    Step 2 of run_sai_pipeline.sh runs the beam analysis for the two layouts in
    background subshells without checking exit codes, so a failure there leaves
    a partial set of outputs. Every aggregation below globs "whatever exists",
    which means metrics silently computed from ONE layout instead of two would be
    written to the CSV looking perfectly valid -- ASCI most of all, since
    combining one histogram rather than two directly lowers the coverage count.

    Metrics are still returned when incomplete (a partial number beats crashing
    the SLURM job), but the caller records the complaint so the row can be found
    and recomputed rather than silently trusted.
    """
    problems = []
    for label, pattern in (
        ("beam properties", "beams_properties_configuration_*.hdf5"),
        ("beam masks", "beams_masks_configuration_*.hdf5"),
        ("ASCI histograms", "asci_histogram_*.hdf5"),
    ):
        found = len(glob.glob(os.path.join(work_dir, pattern)))
        if found != N_LAYOUTS:
            problems.append(f"{label}: found {found}, expected {N_LAYOUTS}")
    n_ppdf = len(glob.glob(os.path.join(work_dir, "position_*_ppdfs_t8_*.hdf5")))
    if n_ppdf != N_TOTAL_FILES:
        problems.append(f"PPDF files: found {n_ppdf}, expected {N_TOTAL_FILES}")
    return problems


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
                    # 11-column schema from pymatana/scanner_modeling/beam_property_io.py:
                    #   0 position_id, 1 detector_id, 2 beam_id, 3 angle (rad),
                    #   4 FWHM (mm), 5 weighted_center_x, 6 weighted_center_y,
                    #   7 sensitivity, 8 relative_sensitivity, 9-10 (padding).
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


# Detector counts on rings 3 and 4 are fixed in the geometry generator
# (generate_mph_scanner_circularfov.py: DETS_PER_RING = [n1, n2, 40*12*2, 40*15*2]).
# Only rings 1 and 2 are design variables.
N_DET_RING3 = 40 * 12 * 2   # 960
N_DET_RING4 = 40 * 15 * 2   # 1200

# Ring weights for the weighted PPDS variant, innermost -> outermost.
# Rationale (RY, Jul 2026): outer rings sit further from the aperture, so their
# PPDFs are better collimated (narrower). Weighting them more strongly rewards
# designs whose counts come from the well-collimated rings, rather than
# rewarding raw sensitivity, which is maximised by wide, blurring PPDFs.
DEFAULT_RING_WEIGHTS = (1.0, 2.0, 3.0, 4.0)


def _ring_slices(n_det: int, n_det_ring1: int):
    """(start, stop) index pairs for rings 1-4, or None if the layout is unresolvable.

    Detectors are laid out ring-by-ring by build_sc_spect_detector_rings, so
    ring membership is just the cumulative counts [n1, n2, 960, 1200]. n2 is
    inferred from the total rather than passed in, since the PPDF row count is
    always the ground truth for how many detectors the simulation actually had.
    """
    if n_det_ring1 is None:
        return None
    n_det_ring2 = n_det - int(n_det_ring1) - N_DET_RING3 - N_DET_RING4
    if n_det_ring2 <= 0:
        print(f"  [warn] PPDS: ring layout does not add up "
              f"(n_det={n_det}, n_det_ring1={n_det_ring1} -> n_det_ring2={n_det_ring2})")
        return None
    sizes = [int(n_det_ring1), int(n_det_ring2), N_DET_RING3, N_DET_RING4]
    edges = np.cumsum([0] + sizes)
    return [(int(edges[i]), int(edges[i + 1])) for i in range(4)]


def _ring_weight_vector(n_det: int, n_det_ring1: int,
                        ring_weights=DEFAULT_RING_WEIGHTS) -> "np.ndarray":
    """Per-detector ring weights, or None if the ring layout cannot be resolved."""
    slices = _ring_slices(n_det, n_det_ring1)
    if slices is None:
        return None
    sizes = [hi - lo for lo, hi in slices]
    return np.repeat(np.asarray(ring_weights, dtype=np.float64), sizes)


def _ppds_components(work_dir: str, n_det_ring1: int = None):
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

    Returns a length-4 array of per-ring contributions when `n_det_ring1` is
    given, a length-1 array with the total otherwise, or None if any required
    file is missing. Callers should use compute_ppds / compute_ppds_per_ring.
    """
    ppdf_files = sorted(glob.glob(os.path.join(work_dir, "position_*_ppdfs_t8_*.hdf5")))
    prop_files = sorted(glob.glob(os.path.join(work_dir, "beams_properties_configuration_*.hdf5")))
    mask_files = sorted(glob.glob(os.path.join(work_dir, "beams_masks_configuration_*.hdf5")))
    if not ppdf_files or not prop_files or not mask_files:
        return None

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
        return None

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
        contrib = ppdfs / denom[:, None]                           # (n_det, n_pix)
        contrib[~valid, :] = 0.0

        # PPDS is a plain sum over detectors and ring weights are constant within
        # a ring, so the per-ring partial sums fully determine the weighted value
        # for ANY weighting. Returning them lets weightings be compared without
        # recomputing from the HDF5 files each time (~20 s per config).
        slices = _ring_slices(n_det, n_det_ring1)
        if slices is None:
            ppds_per_layout.append([float(contrib.sum(axis=0).mean())])
        else:
            ppds_per_layout.append(
                [float(contrib[lo:hi].sum(axis=0).mean()) for lo, hi in slices]
            )

    if not ppds_per_layout:
        return None
    return np.asarray(ppds_per_layout, dtype=np.float64).mean(axis=0)


def compute_ppds(work_dir: str, n_det_ring1: int = None,
                 ring_weights=DEFAULT_RING_WEIGHTS) -> float:
    """Mean PPDS over the FOV, optionally ring-weighted. NaN if files are missing.

    With `n_det_ring1` the per-ring contributions are combined using
    `ring_weights`; without it the plain unweighted total is returned.
    """
    comps = _ppds_components(work_dir, n_det_ring1)
    if comps is None:
        return float("nan")
    if comps.size == 1:
        return float(comps[0])
    return float(np.dot(np.asarray(ring_weights, dtype=np.float64), comps))


# FWHM window for the ASCI objective. Chosen by sweeping thresholds against CNR
# (Jul 2026): 0.45 mm peaks at rho +0.80 with no design saturated at 100%. It
# falls to +0.62 at 0.50 and +0.31 at 0.60 as saturation returns, and to +0.19 at
# 0.40 where so few beams pass that the metric goes sparse.
ASCI_FWHM_WINDOW_MM = 0.45
ASCI_WINDOW_COL = "asci_pct_fwhm0p45"


def compute_windowed_asci(work_dir: str, threshold_mm: float = ASCI_FWHM_WINDOW_MM) -> float:
    """ASCI counting only beams narrower than `threshold_mm`. NaN if unavailable."""
    try:
        from analyze_asci_window import windowed_asci
        return float(windowed_asci(work_dir, (threshold_mm,))[threshold_mm])
    except Exception as e:
        print(f"  [warn] windowed ASCI failed: {e}")
        return float("nan")


def compute_ppds_per_ring(work_dir: str, n_det_ring1: int):
    """Per-ring PPDS contributions (length 4), or None if unavailable.

    Storing these lets any ring weighting be evaluated as a dot product, instead
    of re-reading gigabytes of HDF5 per candidate weighting.
    """
    comps = _ppds_components(work_dir, n_det_ring1)
    if comps is None or comps.size != 4:
        return None
    return comps


def compute_metrics(work_dir: str, n_det_ring1: int = None) -> dict:
    """
    Compute all metrics for a single configuration.

    Includes the three metrics adopted after the Jul 2026 objective review:
      - ppds_ring1..4: per-ring PPDS. Ring 1 alone is the sensitivity
        replacement (rho +0.60 against CNR, where sensitivity was -0.92).
      - asci_pct_fwhm0p45: ASCI restricted to beams narrower than 0.45 mm.
        Unwindowed ASCI saturates (64% of designs at 100%) and correlates
        -0.75 with CNR; the window moves it to +0.80 with no saturation.
      - ppds_weighted_mean is kept only so earlier numbers stay reproducible.

    The per-ring values need n_det_ring1 to resolve ring membership; without it
    those columns come back NaN and the config will be dropped by the optimizer.
    """
    sens_total, sens_mean, n_files = compute_sensitivity(work_dir)
    fwhm_mean, asci_pct = compute_fwhm_and_asci(work_dir)
    mpxi_mean = compute_mpxi(work_dir)
    ppds_mean = compute_ppds(work_dir)

    results = {
        "fwhm_mean": fwhm_mean,
        "sensitivity_total": sens_total,
        "sensitivity_mean": sens_mean,
        "asci_pct": asci_pct,
        "n_ppdf_files": n_files,
        "mpxi_mean": mpxi_mean,
        "ppds_mean": ppds_mean,
        "ppds_weighted_mean": float("nan"),
    }

    for i in range(1, 5):
        results[f"ppds_ring{i}"] = float("nan")
    if n_det_ring1 is not None:
        comps = compute_ppds_per_ring(work_dir, n_det_ring1)
        if comps is not None:
            for i, v in enumerate(comps, start=1):
                results[f"ppds_ring{i}"] = float(v)
            results["ppds_weighted_mean"] = float(
                np.dot(DEFAULT_RING_WEIGHTS, comps))

    results[ASCI_WINDOW_COL] = compute_windowed_asci(work_dir)

    # Flag incomplete input so a silently-partial row can be found later
    problems = check_layout_completeness(work_dir)
    results["inputs_complete"] = int(not problems)
    if problems:
        print("  [ERROR] INCOMPLETE INPUTS -- metrics below are computed from a "
              "partial file set and must not be trusted:")
        for p in problems:
            print(f"    - {p}")

    return results


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
    parser.add_argument("--d2_inner_mm", type=float, default=None,
                        help="Inner diameter of detector ring 2 (D2/D3 design variable)")
    parser.add_argument("--d3_inner_mm", type=float, default=None,
                        help="Inner diameter of detector ring 3 (D2/D3 design variable)")
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
            "ppds_weighted_mean": float("nan"),
            ASCI_WINDOW_COL: float("nan"),
            **{f"ppds_ring{i}": float("nan") for i in range(1, 5)},
        }
        print(f"[{args.config_name}] FORCE_ZERO: {args.reason}")
    else:
        results = compute_metrics(args.work_dir, n_det_ring1=args.n_det_ring1)
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
    if args.d2_inner_mm is not None:
        results["d2_inner_mm"] = args.d2_inner_mm
    if args.d3_inner_mm is not None:
        results["d3_inner_mm"] = args.d3_inner_mm

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
          f"ASCIw={results[ASCI_WINDOW_COL]:.2f}%  "
          f"MPXI={results['mpxi_mean']:.4f}  "
          f"PPDSr1={results['ppds_ring1']:.4e}  "
          f"({results['n_ppdf_files']} PPDF files)")


if __name__ == "__main__":
    main()
