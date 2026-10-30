"""A strip lying along x (angle 0) has its radial FWHM along its length."""
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "optimization"))
from compute_metrics import FOV_NPIX, MM_PER_PX, _per_beam_radial_fwhm


def test_strip_along_x_is_measured_along_its_length():
    n0, n1 = FOV_NPIX
    mask = np.zeros(n0 * n1)
    ppdf = np.zeros(n0 * n1)
    pix = np.arange(n0) * n1 + n1 // 2           # every x index, one y index: a line along x
    mask[pix], ppdf[pix] = 1, 1.0
    length = _per_beam_radial_fwhm(mask, ppdf, 1, 0.0)
    expected = 2.355 * np.std(np.arange(n0) * MM_PER_PX)
    assert math.isclose(length, expected, rel_tol=1e-6)
    assert _per_beam_radial_fwhm(mask, ppdf, 1, math.pi / 2) < 1e-6   # across the line: no width
