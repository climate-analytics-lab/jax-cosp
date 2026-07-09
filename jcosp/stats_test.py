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
        for l in range(nlev):
            if oldgrid_bot[l] - newgrid_top[k] >= 0:
                continue
            if oldgrid_top[l] - newgrid_bot[k] > 0:
                w = (min(oldgrid_top[l], newgrid_top[k])
                     - max(oldgrid_bot[l], newgrid_bot[k]))
                if w != 0:
                    wt += w
                    for j in range(ncol):
                        yp = y[l, j]
                        if log_units:
                            yp = 0.0 if y[l, j] == R_UNDEF else 10 ** (y[l, j] / 10)
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
