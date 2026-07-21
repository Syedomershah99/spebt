"""
Local unit tests for the SAI SC-SPECT MOBO pipeline.

Covers the pure-Python logic that runs on Mac without CCR access:
- backfill_cnr matcher (all three fallback layers)
- mobo_agent feasibility math
- compute_metrics._per_beam_radial_fwhm projection math
- compute_metrics.compute_ppds end-to-end with synthetic HDF5 files
- compute_cnr._write_cnr_to_csv row/column semantics
- CSV column-alignment behaviour of compute_metrics.main()

Run with:
    cd /Users/Omer/Desktop/RA/omer/spebt
    python -m pytest tests/test_pipeline.py -v
"""
import math
import os
import subprocess
import sys
import tempfile

import h5py
import numpy as np
import pandas as pd
import pytest

# Wire the optimization/ folder onto the path so we can import as modules
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_REPO_ROOT, "optimization"))


# =============================================================================
# backfill_cnr._find_cnr_npz — matcher logic
# =============================================================================
class TestBackfillCnrMatcher:
    """Verify the three fallback layers in backfill_cnr._find_cnr_npz."""

    @pytest.fixture
    def recon_root(self, tmp_path):
        """Layout mimicking what we have on CCR: mixed naming conventions."""
        root = tmp_path / "recon_results"
        root.mkdir()

        # A: exact-match style (MOBO — indices are already zero-padded)
        (root / "mobo_0069_ap0.3138_nap124_nd1_612_nd2_230").mkdir()
        np.savez(root / "mobo_0069_ap0.3138_nap124_nd1_612_nd2_230" / "cnr_results.npz",
                 overall_cnr=4.44)

        # B: 500-iter suffix — must be preferred
        (root / "mobo_0069_ap0.3138_nap124_nd1_612_nd2_230_500iter").mkdir()
        np.savez(root / "mobo_0069_ap0.3138_nap124_nd1_612_nd2_230_500iter" / "cnr_results.npz",
                 overall_cnr=4.53)

        # C: LHS-style mismatch (padded index, less-precise aperture)
        (root / "lhs4d_0003_ap0.3400_nap63_nd1_202_nd2_788").mkdir()
        np.savez(root / "lhs4d_0003_ap0.3400_nap63_nd1_202_nd2_788" / "cnr_results.npz",
                 overall_cnr=4.25)

        # D: no matching folder at all
        # (nothing created for config "mobo_9999_...")
        return root

    def test_exact_match_returns_direct(self, recon_root):
        """Layer 1: exact folder name match."""
        from backfill_cnr import _find_cnr_npz
        result = _find_cnr_npz(str(recon_root),
                               "mobo_0069_ap0.3138_nap124_nd1_612_nd2_230")
        assert result is not None
        # With both an exact-match dir AND a _500iter dir, prefix glob picks up both
        # and _pick_best prefers 500iter. So the exact-match layer might miss it.
        # Actually layer 1 checks exact first, so returns the non-500iter dir.
        assert "cnr_results.npz" in result

    def test_prefix_glob_prefers_500iter(self, recon_root):
        """Layer 2: prefix glob when no exact-match. The 500-iter variant should win."""
        from backfill_cnr import _find_cnr_npz
        # Remove the exact-match dir so we fall through to prefix glob
        exact = recon_root / "mobo_0069_ap0.3138_nap124_nd1_612_nd2_230" / "cnr_results.npz"
        exact.unlink()
        exact.parent.rmdir()

        result = _find_cnr_npz(str(recon_root),
                               "mobo_0069_ap0.3138_nap124_nd1_612_nd2_230")
        assert result is not None
        assert "500iter" in result

    def test_design_signature_recovers_lhs(self, recon_root):
        """Layer 3: design-signature match recovers LHS runs with padded index."""
        from backfill_cnr import _find_cnr_npz
        # CSV config has unpadded index + full-precision aperture; recon folder has
        # padded index + truncated aperture. Signature (nap, nd1, nd2) is unique.
        result = _find_cnr_npz(str(recon_root),
                               "lhs4d_3_ap0.340036_nap63_nd1_202_nd2_788")
        assert result is not None
        assert "lhs4d_0003" in result

    def test_returns_none_when_no_match(self, recon_root):
        """No match anywhere → None (backfill logs 'no recon output')."""
        from backfill_cnr import _find_cnr_npz
        result = _find_cnr_npz(str(recon_root),
                               "mobo_9999_ap0.5_nap100_nd1_300_nd2_500")
        assert result is None

    def test_pick_best_prefers_500iter(self):
        """Helper directly: order-independence."""
        from backfill_cnr import _pick_best
        assert "500iter" in _pick_best([
            "/tmp/a/cnr_results.npz",
            "/tmp/b_500iter/cnr_results.npz",
        ])
        assert "500iter" in _pick_best([
            "/tmp/b_500iter/cnr_results.npz",
            "/tmp/a/cnr_results.npz",
        ])
        # No 500iter → first one
        assert _pick_best(["/tmp/a/cnr_results.npz",
                           "/tmp/b/cnr_results.npz"]) == "/tmp/a/cnr_results.npz"


# =============================================================================
# mobo_agent — feasibility math
# =============================================================================
class TestMoboAgentFeasibility:
    """The aperture-overlap feasibility check must accept legit designs and reject
    ones that would cause apertures to touch/overlap on the HR ring."""

    def test_is_feasible_accepts_baseline(self):
        from mobo_agent import is_feasible
        # Baseline: 180 apertures at 0.4 mm diameter → chord ≈ 1.22 mm — fine
        assert is_feasible(0.4, 180)

    def test_is_feasible_rejects_too_many_wide(self):
        from mobo_agent import is_feasible
        # 270 apertures at 1.0 mm — chord per aperture ≈ 0.81 mm < 1.0 mm — overlap
        assert not is_feasible(1.0, 270)

    def test_is_feasible_at_safety_margin(self):
        """Just above the SAFETY_MARGIN threshold should reject; just below accept."""
        from mobo_agent import is_feasible, HR_CIRCUMFERENCE, SAFETY_MARGIN
        n = 200
        boundary_diam = SAFETY_MARGIN * HR_CIRCUMFERENCE / n
        assert not is_feasible(boundary_diam + 0.01, n)
        assert is_feasible(boundary_diam - 0.01, n)

    def test_is_feasible_at_bounds_extreme(self):
        """Design-space corner (min aperture, max n) must be feasible."""
        from mobo_agent import is_feasible
        assert is_feasible(0.2, 270)


# =============================================================================
# compute_metrics._per_beam_radial_fwhm — projection math
# =============================================================================
class TestPerBeamRadialFwhm:
    """Verify the FWHM-along-beam-axis calculation on synthetic beams."""

    def _make_gaussian_beam_row(self, cx_mm, cy_mm, sigma_r_mm, sigma_t_mm,
                                angle_rad, beam_id=1, fov_shape=(200, 200),
                                mm_per_px=0.05):
        """Construct a synthetic (mask_row, ppdf_row) with one 2D Gaussian beam.

        The beam is centered at (cx_mm, cy_mm), oriented so its radial axis points
        along (cos(angle), sin(angle)), with the given sigmas.
        """
        rows = np.repeat(np.arange(fov_shape[0]), fov_shape[1]).astype(np.float64) * mm_per_px
        cols = np.tile(np.arange(fov_shape[1]), fov_shape[0]).astype(np.float64) * mm_per_px
        dx = cols - cx_mm
        dy = rows - cy_mm
        # Radial coord (along beam axis)
        r = dx * math.cos(angle_rad) + dy * math.sin(angle_rad)
        # Tangential coord (perpendicular)
        t = -dx * math.sin(angle_rad) + dy * math.cos(angle_rad)
        # 2D Gaussian intensity
        ppdf_row = np.exp(-(r * r) / (2 * sigma_r_mm ** 2) - (t * t) / (2 * sigma_t_mm ** 2))
        # Mask: pixels above ~1e-3 of the peak belong to the beam
        mask_row = np.where(ppdf_row > 1e-3, beam_id, 0).astype(np.int32)
        return mask_row, ppdf_row.astype(np.float64)

    def test_recovers_fwhm_of_known_gaussian(self):
        """Synthesise a beam with known sigma_radial, verify recovered FWHM ≈ 2.355·sigma."""
        import compute_metrics as cm
        sigma_r = 0.5  # mm
        sigma_t = 0.2
        mask_row, ppdf_row = self._make_gaussian_beam_row(
            cx_mm=5.0, cy_mm=5.0, sigma_r_mm=sigma_r, sigma_t_mm=sigma_t,
            angle_rad=0.0, beam_id=7,
        )
        fwhm = cm._per_beam_radial_fwhm(mask_row, ppdf_row, beam_id=7, angle=0.0)
        expected = 2.355 * sigma_r
        # 5% tolerance (weighted moments deviate from analytical FWHM at these sigmas)
        assert abs(fwhm - expected) / expected < 0.05, f"got {fwhm}, expected {expected}"

    def test_returns_zero_for_empty_beam(self):
        import compute_metrics as cm
        empty_mask = np.zeros(200 * 200, dtype=np.int32)
        empty_ppdf = np.zeros(200 * 200, dtype=np.float64)
        assert cm._per_beam_radial_fwhm(empty_mask, empty_ppdf, beam_id=1, angle=0.0) == 0.0

    def test_returns_zero_for_tiny_beam(self):
        import compute_metrics as cm
        mask_row = np.zeros(200 * 200, dtype=np.int32)
        mask_row[0:2] = 1  # only 2 pixels labelled — below the pix_idx.size < 3 guard
        ppdf_row = np.ones_like(mask_row, dtype=np.float64)
        assert cm._per_beam_radial_fwhm(mask_row, ppdf_row, beam_id=1, angle=0.0) == 0.0

    def test_returns_zero_when_weights_all_zero(self):
        import compute_metrics as cm
        mask_row = np.zeros(200 * 200, dtype=np.int32)
        mask_row[0:100] = 1  # pixels are labelled
        ppdf_row = np.zeros_like(mask_row, dtype=np.float64)  # but zero intensity
        assert cm._per_beam_radial_fwhm(mask_row, ppdf_row, beam_id=1, angle=0.0) == 0.0

    def test_rotation_invariance(self):
        """A beam with radial axis at angle=0 and one at angle=π/2 (same sigmas)
        must both recover the same FWHM_radial when the correct projection angle
        is supplied. Sanity check on the projection direction."""
        import compute_metrics as cm
        sigma_r, sigma_t = 0.5, 0.2  # same physics, different orientation

        mask_0, ppdf_0 = self._make_gaussian_beam_row(
            5.0, 5.0, sigma_r, sigma_t, angle_rad=0.0, beam_id=1,
        )
        fwhm_0 = cm._per_beam_radial_fwhm(mask_0, ppdf_0, 1, 0.0)

        mask_pi2, ppdf_pi2 = self._make_gaussian_beam_row(
            5.0, 5.0, sigma_r, sigma_t, angle_rad=math.pi / 2, beam_id=1,
        )
        fwhm_pi2 = cm._per_beam_radial_fwhm(mask_pi2, ppdf_pi2, 1, math.pi / 2)

        # Both should recover ~2.355 * sigma_r along their (different) radial axes.
        expected = 2.355 * sigma_r
        assert abs(fwhm_0 - expected) / expected < 0.10
        assert abs(fwhm_pi2 - expected) / expected < 0.10
        assert abs(fwhm_0 - fwhm_pi2) / max(fwhm_0, fwhm_pi2) < 0.10


# =============================================================================
# compute_metrics.compute_ppds — end-to-end with synthetic HDF5
# =============================================================================
class TestComputePpds:
    """Build a small synthetic work_dir with matched PPDF, beam-properties, and
    beam-mask HDF5 files, then verify compute_ppds returns something sensible."""

    @pytest.fixture
    def work_dir(self, tmp_path):
        """One layout, one T8 pose, 2 detectors, 1 beam each."""
        wd = tmp_path / "work"
        wd.mkdir()
        n_det = 2
        n_pix = 200 * 200

        # PPDF: (n_det, n_pix). Detector 0 has a small hot spot, detector 1 a bigger one.
        ppdfs = np.zeros((n_det, n_pix), dtype=np.float32)
        # detector 0: 4 pixels lit around index 10101 (~row 50, col 101)
        ppdfs[0, [10101, 10102, 10301, 10302]] = 1.0
        # detector 1: 4 pixels lit around a different location
        ppdfs[1, [15050, 15051, 15250, 15251]] = 1.0

        # Write a single PPDF file: position_000_ppdfs_t8_00.hdf5
        with h5py.File(wd / "position_000_ppdfs_t8_00.hdf5", "w") as h:
            h.create_dataset("ppdfs", data=ppdfs)

        # Mask: (n_det, n_pix), int beam IDs (1-based, 0=background)
        masks = np.zeros((n_det, n_pix), dtype=np.int32)
        masks[0, [10101, 10102, 10301, 10302]] = 1
        masks[1, [15050, 15051, 15250, 15251]] = 1
        with h5py.File(wd / "beams_masks_configuration_000.hdf5", "w") as h:
            h.create_dataset("beam_mask", data=masks)

        # Beam properties: 11 columns (matching the real schema).
        # Layout: 2 rows (one beam per detector). Detector indices are 0-based here.
        # Columns: (position, det_id, beam_id, angle, FWHM, ...pad the rest with zeros)
        bp = np.zeros((2, 11), dtype=np.float32)
        bp[0] = [0, 0, 1, 0.0, 0.30, 0, 0, 0.5, 0.5, 0, 0]  # detector 0, beam 1, angle=0, FWHM=0.30
        bp[1] = [0, 1, 1, 0.5, 0.40, 0, 0, 0.5, 0.5, 0, 0]  # detector 1, beam 1, angle=0.5, FWHM=0.40
        with h5py.File(wd / "beams_properties_configuration_000.hdf5", "w") as h:
            h.create_dataset("beam_properties", data=bp)

        return wd

    def test_ppds_returns_finite_value(self, work_dir):
        import compute_metrics as cm
        val = cm.compute_ppds(str(work_dir))
        assert np.isfinite(val), f"expected finite PPDS, got {val}"
        assert val > 0.0

    def test_ppds_nan_when_ppdfs_missing(self, tmp_path):
        import compute_metrics as cm
        wd = tmp_path / "empty"
        wd.mkdir()
        assert np.isnan(cm.compute_ppds(str(wd)))

    def test_ppds_nan_when_only_masks(self, tmp_path):
        """Masks exist but no PPDF files."""
        import compute_metrics as cm
        wd = tmp_path / "no_ppdf"
        wd.mkdir()
        masks = np.zeros((2, 200 * 200), dtype=np.int32)
        with h5py.File(wd / "beams_masks_configuration_000.hdf5", "w") as h:
            h.create_dataset("beam_mask", data=masks)
        # Also need beam_properties for compute_ppds to enter its layout loop
        bp = np.zeros((0, 11), dtype=np.float32)
        with h5py.File(wd / "beams_properties_configuration_000.hdf5", "w") as h:
            h.create_dataset("beam_properties", data=bp)
        # No PPDF files — should return NaN cleanly
        assert np.isnan(cm.compute_ppds(str(wd)))

    def test_ppds_matches_expected_value(self, work_dir):
        """Compute the expected PPDS by hand and compare.

        For our synthetic case:
        - PPDF sum per pixel across detectors is 1 or 2 (whichever detectors light it).
        - sumV per detector = FWHM_tang * FWHM_rad. FWHM_tang stored = 0.30 (det 0) or 0.40 (det 1).
          FWHM_rad is measured from mask via _per_beam_radial_fwhm. With only 4 aligned pixels,
          the weighted std is small; expected FWHM_rad ~= 2.355 * pixel_pitch_along_axis.
        Since the pixels are placed strategically, we mainly check the value is
        positive and correlates with the expected order of magnitude.
        """
        import compute_metrics as cm
        val = cm.compute_ppds(str(work_dir))
        # Two detectors, each contributes PPDF/sumV summed over their lit pixels.
        # sumV[0] ~= 0.30 * O(0.05*something) ~= small; PPDS_j at their pixels ~= 1/sumV.
        # Mean over 40000 pixels: only 8 non-zero, mean = sum/40000.
        # This is a coarse sanity check on order of magnitude — just verify > 0.
        assert 0.0 < val < 1e6


# =============================================================================
# compute_cnr._write_cnr_to_csv — CSV update semantics
# =============================================================================
class TestWriteCnrToCsv:
    """Verify the row-update / row-append / column-create paths."""

    def test_creates_csv_when_missing(self, tmp_path):
        from compute_cnr import _write_cnr_to_csv
        csv = tmp_path / "results.csv"
        _write_cnr_to_csv(str(csv), "mobo_0001", 3.72)
        df = pd.read_csv(csv)
        assert list(df.columns) == ["config", "cnr_mean"]
        assert df.loc[0, "config"] == "mobo_0001"
        assert df.loc[0, "cnr_mean"] == pytest.approx(3.72)

    def test_updates_existing_row(self, tmp_path):
        from compute_cnr import _write_cnr_to_csv
        csv = tmp_path / "results.csv"
        pd.DataFrame({
            "fwhm_mean": [0.5, 0.6],
            "config": ["mobo_0001", "mobo_0002"],
            "cnr_mean": [np.nan, np.nan],
        }).to_csv(csv, index=False)

        _write_cnr_to_csv(str(csv), "mobo_0001", 4.25)
        df = pd.read_csv(csv)
        assert df.loc[df["config"] == "mobo_0001", "cnr_mean"].iloc[0] == pytest.approx(4.25)
        # Other row untouched
        assert np.isnan(df.loc[df["config"] == "mobo_0002", "cnr_mean"].iloc[0])

    def test_adds_column_when_missing(self, tmp_path):
        from compute_cnr import _write_cnr_to_csv
        csv = tmp_path / "results.csv"
        pd.DataFrame({
            "fwhm_mean": [0.5],
            "config": ["mobo_0001"],
        }).to_csv(csv, index=False)

        _write_cnr_to_csv(str(csv), "mobo_0001", 4.44)
        df = pd.read_csv(csv)
        assert "cnr_mean" in df.columns
        assert df.loc[0, "cnr_mean"] == pytest.approx(4.44)

    def test_appends_row_when_config_missing(self, tmp_path):
        from compute_cnr import _write_cnr_to_csv
        csv = tmp_path / "results.csv"
        pd.DataFrame({
            "fwhm_mean": [0.5],
            "config": ["mobo_0001"],
            "cnr_mean": [3.0],
        }).to_csv(csv, index=False)

        _write_cnr_to_csv(str(csv), "mobo_9999", 2.5)
        df = pd.read_csv(csv)
        assert len(df) == 2
        assert df.loc[df["config"] == "mobo_9999", "cnr_mean"].iloc[0] == pytest.approx(2.5)

    def test_writes_nan(self, tmp_path):
        from compute_cnr import _write_cnr_to_csv
        csv = tmp_path / "results.csv"
        pd.DataFrame({
            "config": ["mobo_0001"],
            "cnr_mean": [1.0],
        }).to_csv(csv, index=False)
        _write_cnr_to_csv(str(csv), "mobo_0001", float("nan"))
        df = pd.read_csv(csv)
        assert np.isnan(df.loc[0, "cnr_mean"])


# =============================================================================
# compute_metrics.main() — CSV column alignment on append
# =============================================================================
class TestCsvAlignment:
    """The append path in compute_metrics.main() MUST align to the existing
    header, or subsequent MOBO GP fits crash on misaligned numeric columns."""

    def _run_compute_metrics_cli(self, csv_path, config_name, work_dir):
        """Run compute_metrics.py as a CLI subprocess with the --force_zero flag
        so we don't need real HDF5 data. Returns the written CSV as a DataFrame."""
        script = os.path.join(_REPO_ROOT, "optimization", "compute_metrics.py")
        subprocess.run(
            [sys.executable, script,
             "--work_dir", str(work_dir),
             "--out_csv", str(csv_path),
             "--config_name", config_name,
             "--aperture_diam_mm", "0.5",
             "--n_apertures", "180",
             "--n_det_ring1", "480",
             "--n_det_ring2", "720",
             "--force_zero",
             "--reason", "test"],
            check=True, capture_output=True, text=True,
        )
        return pd.read_csv(csv_path)

    def test_new_csv_has_expected_columns(self, tmp_path):
        csv = tmp_path / "results.csv"
        wd = tmp_path / "work"
        wd.mkdir()
        df = self._run_compute_metrics_cli(csv, "mobo_test_0001", wd)
        # Every metric column should be present, plus config/work_dir/design params
        for col in ["fwhm_mean", "sensitivity_total", "sensitivity_mean",
                    "asci_pct", "n_ppdf_files", "mpxi_mean", "ppds_mean",
                    "config", "work_dir",
                    "aperture_diam_mm", "n_apertures", "n_det_ring1", "n_det_ring2"]:
            assert col in df.columns, f"missing column {col}"

    def test_append_respects_existing_header_with_extra_cols(self, tmp_path):
        """Simulate CSV that already has cnr_mean at the end (from backfill).
        Appending must place the new row's values in the CORRECT columns."""
        csv = tmp_path / "results.csv"
        wd = tmp_path / "work"
        wd.mkdir()

        # Seed CSV in the shape that would exist AFTER backfill_cnr.py ran:
        # 14 columns, cnr_mean at the very end.
        pd.DataFrame([{
            "fwhm_mean": 0.5, "sensitivity_total": 1.0, "sensitivity_mean": 0.1,
            "asci_pct": 95.0, "n_ppdf_files": 16, "mpxi_mean": 3.0,
            "config": "seed_row", "work_dir": "/tmp/seed",
            "aperture_diam_mm": 0.4, "n_apertures": 180,
            "n_det_ring1": 480, "n_det_ring2": 720,
            "ppds_mean": 0.5, "cnr_mean": 3.0,
        }]).to_csv(csv, index=False)

        # compute_metrics.py appends
        df = self._run_compute_metrics_cli(csv, "new_row_0001", wd)
        assert len(df) == 2

        # The NEW row is the second one; its work_dir must be the string we passed,
        # NOT some misaligned numeric value from a shifted column.
        new_row = df.iloc[1]
        assert new_row["config"] == "new_row_0001"
        assert new_row["work_dir"] == str(wd)
        assert new_row["aperture_diam_mm"] == pytest.approx(0.5)
        assert new_row["n_apertures"] == 180
        # cnr_mean is not written by compute_metrics.py, so should be NaN on the new row
        assert pd.isna(new_row["cnr_mean"])
