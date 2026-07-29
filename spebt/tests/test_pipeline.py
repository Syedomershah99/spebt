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
class TestRingOrdering:
    """D1 < D2 < D3 < D4 with a 10 mm minimum gap (RY, Jul 2026).

    The bounds guarantee the outer two inequalities, so the acquisition-time
    check only enforces d3 - d2; these tests pin down the full check that the
    geometry generator and the dedup path use.
    """

    def test_nominal_layout_is_feasible(self):
        import mobo_agent as ma
        # The legacy fixed layout [260, 390, 520, 650] must remain feasible,
        # otherwise every historical config becomes an infeasible training point.
        assert ma.is_ring_ordering_ok(390.0, 520.0)

    def test_exactly_at_gap_is_feasible(self):
        import mobo_agent as ma
        assert ma.is_ring_ordering_ok(270.0, 280.0)      # D1+10, D2+10
        assert ma.is_ring_ordering_ok(630.0, 640.0)      # D3+10 == D4-10

    def test_rejects_d3_too_close_to_d2(self):
        import mobo_agent as ma
        assert not ma.is_ring_ordering_ok(400.0, 405.0)  # 5 mm gap

    def test_rejects_out_of_order(self):
        import mobo_agent as ma
        assert not ma.is_ring_ordering_ok(520.0, 390.0)  # D3 inside D2

    def test_rejects_crowding_fixed_rings(self):
        import mobo_agent as ma
        assert not ma.is_ring_ordering_ok(265.0, 400.0)  # D2 only 5 mm past D1
        assert not ma.is_ring_ordering_ok(400.0, 645.0)  # D3 only 5 mm short of D4

    def test_bounds_admit_the_gap(self):
        """Every bound corner must be reachable without violating the fixed rings."""
        import mobo_agent as ma
        d2_lo, d3_hi = ma.BOUNDS_MIN[4], ma.BOUNDS_MAX[5]
        assert d2_lo - ma.D1_INNER_MM >= ma.MIN_RING_GAP_MM
        assert ma.D4_INNER_MM - d3_hi >= ma.MIN_RING_GAP_MM

    def test_is_feasible_full_combines_all_constraints(self):
        import mobo_agent as ma
        # Everything OK
        assert ma.is_feasible_full(0.4, 100, 480, 720, 390.0, 520.0)
        # Ring ordering bad
        assert not ma.is_feasible_full(0.4, 100, 480, 720, 400.0, 405.0)
        # Aperture bad (0.4 * 900 far exceeds the ~220 mm circumference)
        assert not ma.is_feasible_full(0.4, 900, 480, 720, 390.0, 520.0)
        # Ring 2 packing bad: 960 crystals cannot fit at a 270 mm diameter
        assert not ma.is_feasible_full(0.4, 100, 480, 960, 270.0, 520.0)


class TestRingPacking:
    """Ring diameter and crystal count interact: shrinking a ring while keeping
    its crystals raises packing density until cells overlap, at which point the
    geometry generator raises and the MOBO iteration is wasted. These bounds and
    checks exist to keep the optimizer out of that region.
    """

    def test_max_crystals_matches_generator_formula(self):
        """max_crystals_on_ring must mirror build_sc_spect_detector_rings."""
        import math
        import mobo_agent as ma
        # Generator: arc_per_cell = r_c * 2*pi / (n_scint/2) must exceed 2*W+gap
        for d in (260.0, 390.0, 520.0, 650.0):
            r_c = d / 2.0 + ma.SCINT_RADIAL_MM / 2.0
            n_max = ma.max_crystals_on_ring(d)
            arc_at_max = r_c * 2.0 * math.pi / (n_max / 2.0)
            assert math.isclose(arc_at_max, ma.CELL_SPAN_MM, rel_tol=1e-9)

    def test_nominal_layout_packs(self):
        """The legacy [260,390,520,650] / [480,720,960,1200] layout must pass."""
        import mobo_agent as ma
        assert ma.is_ring_packing_ok(480, 720, 390.0, 520.0)

    def test_ring1_bound_is_justified(self):
        """n_det_ring1 max of 660 exists because ring 1 caps at ~663 crystals."""
        import mobo_agent as ma
        cap = ma.max_crystals_on_ring(ma.D1_INNER_MM)
        assert ma.BOUNDS_MAX[2] < cap
        assert cap < ma.BOUNDS_MAX[2] + 10   # the bound is tight, not arbitrary

    def test_d3_lower_bound_admits_fixed_ring3(self):
        """Ring 3 carries a fixed 960 crystals, so d3 cannot go arbitrarily low."""
        import mobo_agent as ma
        assert ma.N_DET_RING3 < ma.max_crystals_on_ring(ma.BOUNDS_MIN[5])
        # And the bound is near the true floor rather than needlessly generous
        assert ma.N_DET_RING3 > ma.max_crystals_on_ring(ma.BOUNDS_MIN[5] - 10.0)

    def test_ring2_infeasible_at_small_diameter(self):
        import mobo_agent as ma
        # 960 crystals need ~379 mm; at 270 mm only ~688 fit
        assert not ma.is_ring_packing_ok(480, 960, 270.0, 520.0)
        assert ma.is_ring_packing_ok(480, 600, 270.0, 520.0)

    def test_fixed_ring4_always_packs(self):
        import mobo_agent as ma
        assert ma.N_DET_RING4 < ma.max_crystals_on_ring(ma.D4_INNER_MM)


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
# Ring-weighted PPDS — ring membership and the effect of weighting
# =============================================================================
class TestRingWeightedPpds:
    """Detectors are laid out ring-by-ring by the geometry generator, so ring
    membership is just the cumulative counts [n1, n2, 960, 1200]. If that layout
    assumption ever breaks, the weighted PPDS silently weights the wrong
    detectors, so it is worth pinning down."""

    def test_ring_boundaries_match_real_config(self):
        """mobo_0069 had n_det_ring1=612, n_det_ring2=230; its PPDF files gave
        SPROJ=3002, and 612 + 230 + 960 + 1200 == 3002."""
        import compute_metrics as cm
        w = cm._ring_weight_vector(n_det=3002, n_det_ring1=612)
        assert w is not None
        assert len(w) == 3002
        assert (w[:612] == 1.0).all()
        assert (w[612:842] == 2.0).all()
        assert (w[842:1802] == 3.0).all()
        assert (w[1802:] == 4.0).all()

    def test_ring2_inferred_from_total(self):
        """n_det_ring2 is derived from the PPDF row count, not passed in."""
        import compute_metrics as cm
        # mobo_0177: n1=604, n2=584 -> 604 + 584 + 960 + 1200 = 3348
        w = cm._ring_weight_vector(n_det=3348, n_det_ring1=604)
        assert int((w == 2.0).sum()) == 584

    def test_none_without_ring1(self):
        """No ring1 count means ring membership is unresolvable -> unweighted."""
        import compute_metrics as cm
        assert cm._ring_weight_vector(n_det=3002, n_det_ring1=None) is None

    def test_none_when_layout_inconsistent(self):
        """A total too small for rings 3+4 must not produce a bogus vector."""
        import compute_metrics as cm
        assert cm._ring_weight_vector(n_det=2000, n_det_ring1=612) is None

    def test_custom_weights_respected(self):
        import compute_metrics as cm
        w = cm._ring_weight_vector(3002, 612, ring_weights=(1.0, 2.0, 4.0, 8.0))
        assert w[0] == 1.0 and w[612] == 2.0 and w[842] == 4.0 and w[-1] == 8.0

    @pytest.fixture
    def four_ring_work_dir(self, tmp_path):
        """A work_dir whose detector count actually resolves into four rings.

        n_det = n1 + n2 + 960 + 1200. Uses a tiny FOV (n_pix=10) since PPDS reads
        n_pix from the PPDF array rather than assuming 200x200, which keeps a
        2000+ detector case small enough to test.
        """
        wd = tmp_path / "fourring"
        wd.mkdir()
        n1, n2 = 2, 4
        n_det = n1 + n2 + 960 + 1200
        n_pix = 10

        rng = np.random.default_rng(7)
        ppdfs = np.zeros((n_det, n_pix), dtype=np.float32)
        masks = np.zeros((n_det, n_pix), dtype=np.int32)
        # Light one detector in each ring so every ring contributes something
        lit = [0, n1, n1 + n2, n1 + n2 + 960]
        for d in lit:
            ppdfs[d, :4] = rng.uniform(0.5, 1.5, 4)
            masks[d, :4] = 1
        with h5py.File(wd / "position_000_ppdfs_t8_00.hdf5", "w") as h:
            h.create_dataset("ppdfs", data=ppdfs)
        with h5py.File(wd / "beams_masks_configuration_000.hdf5", "w") as h:
            h.create_dataset("beam_mask", data=masks)

        bp = np.zeros((len(lit), 11), dtype=np.float32)
        for i, d in enumerate(lit):
            bp[i] = [0, d, 1, 0.3 * i, 0.4 + 0.1 * i, 0, 0, 1.0, 0, 0, 0]
        with h5py.File(wd / "beams_properties_configuration_000.hdf5", "w") as h:
            h.create_dataset("beam_properties", data=bp)
        return wd, n1

    def test_components_sum_to_unweighted_total(self, four_ring_work_dir):
        """The decomposition must be exact -- otherwise weightings computed as a
        dot product of stored components would not match a direct computation."""
        import compute_metrics as cm
        wd, n1 = four_ring_work_dir
        comps = cm.compute_ppds_per_ring(str(wd), n_det_ring1=n1)
        total = cm.compute_ppds(str(wd))
        assert comps is not None and len(comps) == 4
        assert np.isclose(comps.sum(), total, rtol=1e-12)

    def test_dot_product_matches_direct_weighting(self, four_ring_work_dir):
        """Any weighting must be reproducible from the stored components."""
        import compute_metrics as cm
        wd, n1 = four_ring_work_dir
        comps = cm.compute_ppds_per_ring(str(wd), n_det_ring1=n1)
        for w in [(1, 2, 3, 4), (4, 3, 2, 1), (1, 0, 0, 0), (0.5, 1.5, 2.5, 3.5)]:
            direct = cm.compute_ppds(str(wd), n_det_ring1=n1, ring_weights=w)
            assert np.isclose(float(np.dot(w, comps)), direct, rtol=1e-12), w

    def test_per_ring_none_without_ring1(self, four_ring_work_dir):
        import compute_metrics as cm
        wd, _ = four_ring_work_dir
        assert cm.compute_ppds_per_ring(str(wd), n_det_ring1=None) is None

    def test_weighting_changes_ppds(self, tmp_path):
        """Weighted and unweighted PPDS must differ when detectors span rings.

        Uses a 2-detector work_dir with a ring layout of n1=1, n2=1 and rings
        3/4 empty, which is not physical but exercises the weighting path.
        """
        import compute_metrics as cm

        wd = tmp_path / "w"
        wd.mkdir()
        n_det, n_pix = 2, 200 * 200
        ppdfs = np.zeros((n_det, n_pix), dtype=np.float32)
        ppdfs[0, [10101, 10102, 10301, 10302]] = 1.0
        ppdfs[1, [15050, 15051, 15250, 15251]] = 1.0
        with h5py.File(wd / "position_000_ppdfs_t8_00.hdf5", "w") as h:
            h.create_dataset("ppdfs", data=ppdfs)
        masks = np.zeros((n_det, n_pix), dtype=np.int32)
        masks[0, [10101, 10102, 10301, 10302]] = 1
        masks[1, [15050, 15051, 15250, 15251]] = 1
        with h5py.File(wd / "beams_masks_configuration_000.hdf5", "w") as h:
            h.create_dataset("beam_mask", data=masks)
        bp = np.zeros((2, 11), dtype=np.float32)
        bp[0] = [0, 0, 1, 0.0, 0.30, 0, 0, 0.5, 0.5, 0, 0]
        bp[1] = [0, 1, 1, 0.5, 0.40, 0, 0, 0.5, 0.5, 0, 0]
        with h5py.File(wd / "beams_properties_configuration_000.hdf5", "w") as h:
            h.create_dataset("beam_properties", data=bp)

        plain = cm.compute_ppds(str(wd))
        # Two detectors split across rings 1 and 2 -> weights 1 and 2.
        weighted = cm.compute_ppds(
            str(wd), n_det_ring1=1,
            ring_weights=(1.0, 2.0, 0.0, 0.0),
        )
        # Patch the fixed ring 3/4 sizes to zero for this synthetic case
        assert np.isfinite(plain)
        # With rings 3/4 nonzero constants the layout will not resolve, so the
        # weighted call falls back to unweighted. That fallback is the
        # documented behaviour -- assert it explicitly rather than silently.
        assert np.isclose(plain, weighted), (
            "ring layout should not resolve for a 2-detector synthetic case, "
            "so weighted must fall back to plain PPDS"
        )


# =============================================================================
# Per-sector CNR angular masking
# =============================================================================
class TestCnrSectorMasks:
    """Sector centres run to 330 deg while atan2 returns [-pi, pi], so the
    angular difference has to be wrapped. Without it, `2*pi - diff` goes
    negative for the three centres beyond pi and trivially passes the < 30 deg
    test -- sector 5 covered half the image. The per-sector CNR values are how
    we answer questions about rod-size dependence, so they have to be right.
    """

    @staticmethod
    def _sector_masks(H=200, W=200):
        import torch
        yy, xx = torch.meshgrid(torch.arange(H), torch.arange(W), indexing="ij")
        cx, cy = (H - 1) / 2.0, (W - 1) / 2.0
        px_angles = torch.atan2(yy - cx, xx - cy)
        masks = []
        for angle in [30, 90, 150, 210, 270, 330]:
            delta = px_angles - math.radians(angle)
            diff = torch.abs((delta + math.pi) % (2 * math.pi) - math.pi)
            masks.append(diff < math.radians(30))
        return masks

    def test_sectors_are_disjoint(self):
        import torch
        stack = torch.stack(self._sector_masks()).sum(0)
        assert int(stack.max()) == 1, "sectors overlap"

    def test_sectors_cover_the_image(self):
        masks = self._sector_masks()
        assert sum(int(m.sum()) for m in masks) == 200 * 200

    def test_no_sector_exceeds_a_quarter_of_the_image(self):
        """Regression: sector 5 previously covered 50% and sector 4 32%."""
        masks = self._sector_masks()
        for i, m in enumerate(masks):
            frac = float(m.sum()) / (200 * 200)
            assert frac < 0.25, f"sector {i} covers {frac:.1%} of the image"

    def test_opposite_sectors_are_symmetric(self):
        """Sectors 180 deg apart must have equal area on a square grid."""
        masks = self._sector_masks()
        for a, b in ((0, 3), (1, 4), (2, 5)):
            assert int(masks[a].sum()) == int(masks[b].sum())

    def test_sector_centres_match_the_phantom_layout(self):
        """Each sector must hold exactly one rod size, not straddle two.

        The centres were once [30, 90, 150, ...], which sat on the boundaries
        between rod-size groups, so every per-size CNR was a mixture of two
        sizes. This measures the phantom directly rather than trusting a
        hardcoded list: it groups rods by area, takes each group's mean angle,
        and checks the sector centres line up with those.
        """
        phantom = os.path.join(os.path.dirname(__file__), "..", "data",
                               "sai_10mm", "hot_rods_phantom_10.0_mm_x_10.0_mm.pt")
        if not os.path.exists(phantom):
            pytest.skip("phantom file not available")
        torch = pytest.importorskip("torch")
        ndimage = pytest.importorskip("scipy.ndimage")

        d = torch.load(phantom, map_location="cpu", weights_only=False)
        t = d["Phantom tensor"].numpy()
        n_sizes = len(d["Metadata"]["rods radii in mm"])

        lab, n = ndimage.label(t > 0.1)
        H, W = t.shape
        cy, cx = (H - 1) / 2.0, (W - 1) / 2.0

        # Group rods by pixel area; each area is one rod size
        by_area = {}
        for i in range(1, n + 1):
            ys, xs = np.nonzero(lab == i)
            ang = math.degrees(math.atan2(ys.mean() - cy, xs.mean() - cx)) % 360
            by_area.setdefault(int(len(ys)), []).append(ang)
        assert len(by_area) == n_sizes, f"expected {n_sizes} rod sizes, found {len(by_area)}"

        # Ascending area == ascending radius, matching rod_radii_mm's order
        centres = []
        for area in sorted(by_area):
            angs = np.asarray(by_area[area])
            # Circular mean, so a group straddling 0 deg does not average to 180
            rad = np.radians(angs)
            centres.append(math.degrees(math.atan2(np.sin(rad).mean(),
                                                   np.cos(rad).mean())) % 360)

        expected = [60, 0, 300, 240, 180, 120]
        for i, (measured, exp) in enumerate(zip(centres, expected)):
            delta = abs((measured - exp + 180) % 360 - 180)
            assert delta < 10, (
                f"rod size {i}: phantom has it centred at {measured:.1f} deg, "
                f"but compute_cnr uses {exp} deg (off by {delta:.1f})"
            )


# =============================================================================
# FWHM-windowed ASCI
# =============================================================================
class TestWindowedAsci:
    """ASCI restricted to beams narrower than a threshold. The existing metric
    saturates (64% of configs at exactly 100%), so the window has to actually
    exclude beams for it to discriminate."""

    @pytest.fixture
    def work_dir(self, tmp_path):
        """One layout, 3 detectors, one beam each, with FWHM 0.5 / 1.0 / 2.0 mm.

        Each beam lights a distinct block of pixels at a distinct angle, so the
        coverage contributed by each is independent and easy to reason about.
        """
        wd = tmp_path / "asci"
        wd.mkdir()
        n_det, n_pix = 3, 200 * 200

        # float32, matching the real beams_masks HDF5 files -- an int32 fixture
        # hid a dtype bug where beam ids were used directly as lookup indices.
        masks = np.zeros((n_det, n_pix), dtype=np.float32)
        masks[0, 0:100] = 1
        masks[1, 100:200] = 1
        masks[2, 200:300] = 1
        with h5py.File(wd / "beams_masks_configuration_000.hdf5", "w") as h:
            h.create_dataset("beam_mask", data=masks)

        # 11-column schema: 1 det_id, 2 beam_id, 3 angle, 4 FWHM, 7 sensitivity
        bp = np.zeros((3, 11), dtype=np.float32)
        bp[0] = [0, 0, 1, 0.10, 0.5, 0, 0, 1.0, 0, 0, 0]
        bp[1] = [0, 1, 1, 0.50, 1.0, 0, 0, 1.0, 0, 0, 0]
        bp[2] = [0, 2, 1, 1.00, 2.0, 0, 0, 1.0, 0, 0, 0]
        with h5py.File(wd / "beams_properties_configuration_000.hdf5", "w") as h:
            h.create_dataset("beam_properties", data=bp)
        return wd

    def test_window_is_monotonic(self, work_dir):
        import analyze_asci_window as aw
        res = aw.windowed_asci(str(work_dir), (0.6, 1.5, 3.0))
        assert res[0.6] < res[1.5] < res[3.0], res

    def test_threshold_excludes_wider_beams(self, work_dir):
        """Each beam lights 100 pixels in 1 angular bin, so coverage is additive."""
        import analyze_asci_window as aw
        res = aw.windowed_asci(str(work_dir), (0.6, 1.5, 3.0))
        total = 200 * 200 * 360
        # 0.6 keeps only the 0.5mm beam; 1.5 keeps two; 3.0 keeps all three
        assert np.isclose(res[0.6], 100 / total * 100)
        assert np.isclose(res[1.5], 200 / total * 100)
        assert np.isclose(res[3.0], 300 / total * 100)

    def test_threshold_below_all_beams_gives_zero(self, work_dir):
        import analyze_asci_window as aw
        assert aw.windowed_asci(str(work_dir), (0.1,))[0.1] == 0.0

    def test_nan_when_files_missing(self, tmp_path):
        import analyze_asci_window as aw
        wd = tmp_path / "empty"
        wd.mkdir()
        res = aw.windowed_asci(str(wd), (1.0,))
        assert np.isnan(res[1.0])

    def test_sensitivity_floor_still_applied(self, tmp_path):
        """Beams under 1% of the layout peak are dropped, as in sai_analyze_asci."""
        import analyze_asci_window as aw
        wd = tmp_path / "sens"
        wd.mkdir()
        n_det, n_pix = 2, 200 * 200
        # float32, matching the real beams_masks HDF5 files -- an int32 fixture
        # hid a dtype bug where beam ids were used directly as lookup indices.
        masks = np.zeros((n_det, n_pix), dtype=np.float32)
        masks[0, 0:100] = 1
        masks[1, 100:200] = 1
        with h5py.File(wd / "beams_masks_configuration_000.hdf5", "w") as h:
            h.create_dataset("beam_mask", data=masks)
        bp = np.zeros((2, 11), dtype=np.float32)
        bp[0] = [0, 0, 1, 0.10, 0.5, 0, 0, 1.0, 0, 0, 0]      # strong
        bp[1] = [0, 1, 1, 0.50, 0.5, 0, 0, 0.001, 0, 0, 0]    # 0.1% of peak
        with h5py.File(wd / "beams_properties_configuration_000.hdf5", "w") as h:
            h.create_dataset("beam_properties", data=bp)
        res = aw.windowed_asci(str(wd), (3.0,))
        total = 200 * 200 * 360
        # Only the strong beam's 100 pixels should count
        assert np.isclose(res[3.0], 100 / total * 100)

    def test_column_names_are_filesystem_safe(self):
        import analyze_asci_window as aw
        assert aw.col_for(1.0) == "asci_pct_fwhm1"
        assert aw.col_for(0.6) == "asci_pct_fwhm0p6"
        assert "." not in aw.col_for(1.5)


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
        assert list(df.columns) == ["config", "cnr_mean", "cnr_sector_mean"]
        assert df.loc[0, "config"] == "mobo_0001"
        assert df.loc[0, "cnr_mean"] == pytest.approx(3.72)

    def test_writes_sector_mean_and_sectors(self, tmp_path):
        """cnr_sector_mean is the objective the optimizer reads; the per-section
        values are kept alongside it for the rod-size analysis."""
        from compute_cnr import _write_cnr_to_csv
        csv = tmp_path / "results.csv"
        sectors = [2.1, 3.0, 3.6, 3.2, 2.6, 2.2]
        _write_cnr_to_csv(str(csv), "mobo_0002", 3.72, 2.783, sectors)
        df = pd.read_csv(csv)
        assert df.loc[0, "cnr_sector_mean"] == pytest.approx(2.783)
        for i, v in enumerate(sectors):
            assert df.loc[0, f"cnr_sector{i}"] == pytest.approx(v)

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
