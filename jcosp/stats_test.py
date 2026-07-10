"""Tests for stats.py against loop transliterations of cosp_stats.F90."""

import unittest

import numpy as np

from jcosp import stats
from jcosp.constants import R_GROUND, R_UNDEF


def change_vertical_grid_reference(zfull, zhalf, y, newgrid_bot, newgrid_top,
                                   log_units=False):
    """Loop transliteration of COSP_CHANGE_VERTICAL_GRID (single point).

    Inputs here are SFC-first, as the Fortran requires; the caller flips.
    """
    nlev, ncol = y.shape
    nglev = len(newgrid_bot)
    r = np.zeros((nglev, ncol))

    oldgrid_bot = zhalf.copy()
    oldgrid_top = np.empty(nlev)
    oldgrid_top[:-1] = oldgrid_bot[1:]
    oldgrid_top[-1] = zfull[-1] + zfull[-1] - zhalf[-1]

    for k in range(nglev):
        wt = 0.0
        for lev in range(nlev):
            if oldgrid_bot[lev] - newgrid_top[k] >= 0:
                continue
            if oldgrid_top[lev] - newgrid_bot[k] > 0:
                w = (min(oldgrid_top[lev], newgrid_top[k])
                     - max(oldgrid_bot[lev], newgrid_bot[k]))
                if w != 0:
                    wt += w
                    for j in range(ncol):
                        yp = y[lev, j]
                        if log_units:
                            yp = 0.0 if y[lev, j] == R_UNDEF else 10 ** (y[lev, j] / 10)
                        r[k, j] += w * yp
        if wt > 0:
            r[k, :] /= wt
    for k in range(nglev):
        for j in range(ncol):
            if newgrid_top[k] > zhalf[0]:
                if log_units:
                    r[k, j] = R_UNDEF if r[k, j] <= 0 else 10 * np.log10(r[k, j])
            else:
                r[k, j] = R_GROUND
    return r


class ChangeVerticalGridTest(unittest.TestCase):
    def _model_grid(self, nlev, surfelev=0.0):
        # Stretched model levels, SFC-first here for the reference.
        zhalf = surfelev + np.concatenate([[0], np.cumsum(
            200 + 150 * np.arange(nlev - 1))])
        zfull = zhalf + 0.5 * np.diff(np.append(zhalf, zhalf[-1] + 4000))
        return zfull, zhalf

    def test_matches_fortran_transliteration(self):
        rng = np.random.default_rng(3)
        nlev, ncol, nglev = 18, 5, 40
        zstep = 480.0
        new_bot_sfc = zstep * np.arange(nglev)
        new_top_sfc = new_bot_sfc + zstep
        for log_units in (False, True):
            for surfelev in (0.0, 700.0):
                zfull_s, zhalf_s = self._model_grid(nlev, surfelev)
                if log_units:
                    y_s = rng.uniform(-40, 20, (nlev, ncol))
                    y_s[rng.uniform(size=y_s.shape) < 0.3] = R_UNDEF
                else:
                    y_s = rng.uniform(0, 1, (nlev, ncol))
                ref = change_vertical_grid_reference(
                    zfull_s, zhalf_s, y_s, new_bot_sfc, new_top_sfc, log_units)
                # JAX version is TOA-first: flip inputs and outputs.
                out = np.asarray(stats.change_vertical_grid(
                    zfull_s[::-1].copy(), zhalf_s[::-1].copy(),
                    y_s[::-1].copy(), new_bot_sfc[::-1].copy(),
                    new_top_sfc[::-1].copy(), log_units=log_units))
                np.testing.assert_allclose(
                    out[::-1], ref, rtol=1e-6, atol=1e-6,
                    err_msg=f"log_units={log_units} surfelev={surfelev}")

    def test_conserves_mean_on_matching_grids(self):
        # Regridding onto the same layering returns the same values.
        nlev = 10
        zhalf_s = 480.0 * np.arange(nlev)
        zfull_s = zhalf_s + 240.0
        y = np.linspace(0, 1, nlev)[:, None] * np.ones((1, 3))
        out = np.asarray(stats.change_vertical_grid(
            zfull_s[::-1].copy(), zhalf_s[::-1].copy(), y[::-1].copy(),
            (zhalf_s[::-1]).copy(), (zhalf_s[::-1] + 480.0).copy()))
        np.testing.assert_allclose(out, y[::-1], atol=1e-12)


def hist2d_reference(var1, var2, bin1, bin2):
    """Loop transliteration of cosp_stats.F90:hist2D (single batch element)."""
    nbin1, nbin2 = len(bin1) - 1, len(bin2) - 1
    joint = np.zeros((nbin1, nbin2))
    for ij in range(nbin1):
        for ik in range(nbin2):
            joint[ij, ik] = np.count_nonzero(
                (var1 >= bin1[ij]) & (var1 < bin1[ij + 1])
                & (var2 >= bin2[ik]) & (var2 < bin2[ik + 1]))
    return joint


class Hist2dTest(unittest.TestCase):
    def test_matches_fortran_transliteration(self):
        rng = np.random.default_rng(7)
        n, nb = 200, 3
        bin1 = np.array([-30.0, -20.0, -10.0, 0.0, 10.0, 20.0])
        bin2 = np.array([0.0, 2.0, 4.0, 8.0, 60.0])
        var1 = rng.uniform(-40, 30, (n, nb))
        var2 = rng.uniform(-1, 70, (n, nb))
        # Sprinkle sentinel fills, which must simply not be counted.
        var1[rng.uniform(size=var1.shape) < 0.1] = R_UNDEF
        var2[rng.uniform(size=var2.shape) < 0.1] = -999.0
        out = np.asarray(stats.hist2d(var1, var2, bin1, bin2))
        for b in range(nb):
            np.testing.assert_array_equal(
                out[:, :, b], hist2d_reference(var1[:, b], var2[:, b],
                                               bin1, bin2))

    def test_total_count(self):
        # Every in-range sample lands in exactly one joint bin.
        rng = np.random.default_rng(8)
        var1 = rng.uniform(0.0, 1.0, (50,))
        var2 = rng.uniform(0.0, 1.0, (50,))
        out = np.asarray(stats.hist2d(var1, var2,
                                      np.linspace(0, 1, 5),
                                      np.linspace(0, 1, 4)))
        self.assertEqual(out.sum(), 50)


class Hist1dTest(unittest.TestCase):
    def test_counts(self):
        var = np.array([[-60.0, -12.0, 3.0, 100.0]]).T  # (ncol=4, batch=1)
        bins = np.array([-100.0, -15.0, 0.0, 80.0])
        out = np.asarray(stats.hist1d(var, bins))
        np.testing.assert_array_equal(out[:, 0], [1, 1, 1])  # 100 is out of range

    def test_ground_poisons(self):
        var = np.array([[-60.0, R_GROUND, 3.0]]).T
        bins = np.array([-100.0, 0.0, 80.0])
        out = np.asarray(stats.hist1d(var, bins))
        self.assertTrue((out == R_UNDEF).all())


if __name__ == "__main__":
    unittest.main()
