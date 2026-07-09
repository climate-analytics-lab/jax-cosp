"""Tests for the SCOPS port.

The vectorized JAX implementation is checked against a direct, loop-for-loop
NumPy transliteration of the Fortran (same uniforms in, same subcolumns out),
plus statistical checks that sampled subcolumns reproduce the input cloud
fractions under each overlap assumption.
"""

import unittest

import jax
import numpy as np

from jcosp.subcol.scops import generate_uniforms, scops, scops_from_uniforms


def scops_reference(cc, conv, u_init, u_levels, overlap):
    """Loop transliteration of scops.F90 (npoints=1 batch handled by caller)."""
    nlev = cc.shape[0]
    ncol = u_levels.shape[1]
    frac_out = np.zeros((nlev, ncol), dtype=int)
    boxpos = (np.arange(ncol) + 0.5) / ncol
    threshold = np.empty(ncol)

    for ilev in range(nlev):
        if ilev == 0:
            if overlap == 1:
                threshold[:] = boxpos
            else:
                threshold[:] = conv[0] + (1 - conv[0]) * u_init

        for ibox in range(ncol):
            maxocc = 1 if boxpos[ibox] <= conv[ilev] else 0
            if overlap == 1:
                threshold_min, maxosc = conv[ilev], 1
            elif overlap == 2:
                threshold_min, maxosc = conv[ilev], 0
            else:
                tca_above = 0.0 if ilev == 0 else cc[ilev - 1]
                ovl = min(tca_above, cc[ilev])
                threshold_min = max(conv[ilev], ovl)
                maxosc = 1 if (threshold[ibox] < ovl
                               and threshold[ibox] > conv[ilev]) else 0
            ran = u_levels[ilev, ibox]
            threshold[ibox] = (
                maxocc * boxpos[ibox]
                + (1 - maxocc) * (maxosc * threshold[ibox]
                                  + (1 - maxosc) * (threshold_min
                                                    + (1 - threshold_min) * ran)))
            frac_out[ilev, ibox] = 1 if cc[ilev] > threshold[ibox] else 0
            if threshold[ibox] <= conv[ilev] and conv[ilev] > 0:
                frac_out[ilev, ibox] = 2
    return frac_out


class ScopsTest(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(1234)
        self.nlev, self.ncol = 12, 20

    def _random_profile(self):
        cc = np.clip(self.rng.uniform(-0.2, 0.9, self.nlev), 0, 1)
        conv = np.clip(self.rng.uniform(-0.1, 0.3, self.nlev), 0, 1)
        conv = np.minimum(conv, cc)  # tca includes the convective part
        return cc, conv

    def test_matches_fortran_transliteration(self):
        for overlap in (1, 2, 3):
            for trial in range(5):
                cc, conv = self._random_profile()
                u_init = self.rng.uniform(size=self.ncol)
                u_levels = self.rng.uniform(size=(self.nlev, self.ncol))
                ref = scops_reference(cc, conv, u_init, u_levels, overlap)
                out = scops_from_uniforms(cc, conv, u_init, u_levels, overlap)
                np.testing.assert_array_equal(
                    np.asarray(out), ref,
                    err_msg=f"overlap={overlap} trial={trial}")

    def test_batch_agrees_with_single_column(self):
        # A batched call with tiled uniforms must reproduce each column.
        cc, conv = self._random_profile()
        u_init = self.rng.uniform(size=self.ncol)
        u_levels = self.rng.uniform(size=(self.nlev, self.ncol))
        single = scops_from_uniforms(cc, conv, u_init, u_levels, 3)
        nb = 3
        out = scops_from_uniforms(
            np.repeat(cc[:, None], nb, 1), np.repeat(conv[:, None], nb, 1),
            np.repeat(u_init[:, None], nb, 1),
            np.repeat(u_levels[:, :, None], nb, 2), 3)
        for b in range(nb):
            np.testing.assert_array_equal(np.asarray(out[:, :, b]),
                                          np.asarray(single))

    def test_cloud_fraction_reproduced(self):
        # With many subcolumns the per-level cloudy fraction approaches the
        # input cloud fraction, and the convective part approaches conv.
        cc = np.array([0.0, 0.3, 0.7, 0.5, 0.9])
        conv = np.array([0.0, 0.1, 0.2, 0.0, 0.4])
        ncol = 4000
        for overlap in (1, 2, 3):
            out = np.asarray(scops(jax.random.PRNGKey(0), cc, conv, ncol, overlap))
            frac_cloudy = (out > 0).mean(axis=1)
            frac_conv = (out == 2).mean(axis=1)
            np.testing.assert_allclose(frac_cloudy, cc, atol=0.03,
                                       err_msg=f"overlap={overlap}")
            np.testing.assert_allclose(frac_conv, conv, atol=0.03,
                                       err_msg=f"overlap={overlap}")

    def test_maximum_overlap_nests_clouds(self):
        # Under maximum overlap, the cloudy subcolumns of a less-cloudy level
        # are a subset of those of any cloudier level.
        cc = np.array([0.2, 0.6, 0.4])
        conv = np.zeros(3)
        out = np.asarray(scops(jax.random.PRNGKey(1), cc, conv, 1000, overlap=1))
        cloudy = out > 0
        for k in range(3):
            for m in range(3):
                if cc[k] <= cc[m]:
                    self.assertTrue(np.all(~cloudy[k] | cloudy[m]))

    def test_invalid_overlap_raises(self):
        cc, conv = self._random_profile()
        with self.assertRaises(ValueError):
            scops(jax.random.PRNGKey(0), cc, conv, 10, overlap=4)


if __name__ == "__main__":
    unittest.main()
