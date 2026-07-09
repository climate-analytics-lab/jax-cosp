"""Tests for the PREC_SCOPS port, against a loop transliteration of the Fortran."""

import unittest

import jax
import numpy as np

from jcosp.constants import SCOPS_CCFRAC
from jcosp.subcol.prec_scops import prec_scops
from jcosp.subcol.scops import scops


def prec_scops_reference(frac_out, ls_p_rate, cv_p_rate):
    """Loop transliteration of prec_scops.F90 for a single gridpoint.

    frac_out is (nlev, ncol); level 0 is the top, matching the Fortran's
    TOA-to-surface ordering.
    """
    nlev, ncol = frac_out.shape
    prec = np.zeros((nlev, ncol), dtype=int)
    cv_col = int(SCOPS_CCFRAC * ncol) or 1

    col_has_ls = (frac_out == 1).any(axis=0)
    col_has_cv = (frac_out == 2).any(axis=0)

    for ilev in range(nlev):
        # Large-scale block.
        if ls_p_rate[ilev] > 0:
            flag = False
            for ibox in range(ncol):  # possibility ONE (&TWO below top)
                if frac_out[ilev, ibox] == 1 or (
                        ilev > 0 and prec[ilev - 1, ibox] in (1, 3)):
                    prec[ilev, ibox] = 1
                    flag = True
            if not flag and ilev < nlev - 1:  # possibility THREE
                for ibox in range(ncol):
                    if frac_out[ilev + 1, ibox] == 1:
                        prec[ilev, ibox] = 1
                        flag = True
            if not flag:  # possibility FOUR
                for ibox in range(ncol):
                    if col_has_ls[ibox]:
                        prec[ilev, ibox] = 1
                        flag = True
            if not flag:  # possibility FIVE
                prec[ilev, :] = 1
        # Convective block.
        if cv_p_rate[ilev] > 0:
            def _set_cv(ibox):
                prec[ilev, ibox] = 2 if prec[ilev, ibox] == 0 else 3

            flag = False
            for ibox in range(ncol):
                if frac_out[ilev, ibox] == 2 or (
                        ilev > 0 and prec[ilev - 1, ibox] in (2, 3)):
                    _set_cv(ibox)
                    flag = True
            if not flag and ilev < nlev - 1:
                for ibox in range(ncol):
                    if frac_out[ilev + 1, ibox] == 2:
                        _set_cv(ibox)
                        flag = True
            if not flag:
                for ibox in range(ncol):
                    if col_has_cv[ibox]:
                        _set_cv(ibox)
                        flag = True
            if not flag:
                for ibox in range(cv_col):
                    _set_cv(ibox)
    return prec


class PrecScopsTest(unittest.TestCase):
    def test_matches_fortran_transliteration(self):
        rng = np.random.default_rng(42)
        nlev, ncol = 10, 25
        for trial in range(10):
            cc = np.clip(rng.uniform(-0.3, 0.8, nlev), 0, 1)
            conv = np.minimum(np.clip(rng.uniform(-0.1, 0.3, nlev), 0, 1), cc)
            frac_out = np.asarray(
                scops(jax.random.PRNGKey(trial), cc, conv, ncol))
            # Sparse random precip fluxes, including all-zero levels.
            ls = np.where(rng.uniform(size=nlev) > 0.5,
                          rng.uniform(size=nlev), 0.0)
            cv = np.where(rng.uniform(size=nlev) > 0.6,
                          rng.uniform(size=nlev), 0.0)
            ref = prec_scops_reference(frac_out, ls, cv)
            out = np.asarray(prec_scops(frac_out, ls, cv))
            np.testing.assert_array_equal(out, ref, err_msg=f"trial={trial}")

    def test_batched(self):
        rng = np.random.default_rng(7)
        nlev, ncol, nb = 8, 12, 4
        frac_out = rng.integers(0, 3, size=(nlev, ncol, nb))
        ls = rng.uniform(size=(nlev, nb)) * (rng.uniform(size=(nlev, nb)) > 0.4)
        cv = rng.uniform(size=(nlev, nb)) * (rng.uniform(size=(nlev, nb)) > 0.6)
        out = np.asarray(prec_scops(frac_out, ls, cv))
        for b in range(nb):
            ref = prec_scops_reference(frac_out[:, :, b], ls[:, b], cv[:, b])
            np.testing.assert_array_equal(out[:, :, b], ref, err_msg=f"b={b}")

    def test_no_precip_no_flags(self):
        frac_out = np.ones((5, 8), dtype=int)
        out = np.asarray(prec_scops(frac_out, np.zeros(5), np.zeros(5)))
        self.assertTrue((out == 0).all())


if __name__ == "__main__":
    unittest.main()
