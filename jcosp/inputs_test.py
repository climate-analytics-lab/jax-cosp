"""Tests for the host-input mapping (cosp_precip_mxratio + distribution)."""

import unittest

import numpy as np

from jcosp import config, inputs
from jcosp.constants import RD_PRECIP
from jcosp.subcol import prec_scops


def precip_mxratio_reference(p, t, prec_frac, prec_type, n_ax, n_bx, alpha_x,
                             c_x, d_x, g_x, a_x, b_x, g1, g2, g3, g4, flux):
    """Loop transliteration of COSP_PRECIP_MXRATIO (single point)."""
    nlev, ncol = prec_frac.shape
    mxratio = np.zeros((nlev, ncol))
    reff = np.zeros((nlev, ncol))
    if n_ax < 0:
        return mxratio, reff
    xi = d_x / (alpha_x + b_x - n_bx + 1.0)
    rho0 = 1.29
    sigma = (g2 / (g1 * c_x)) * (n_ax * a_x * g2) ** xi
    one_over_xip1 = 1.0 / (xi + 1.0)
    gamma_4_3_2 = 0.5 * g4 / g3
    delta = alpha_x + b_x + d_x - n_bx + 1.0
    for k in range(nlev):
        for j in range(ncol):
            if prec_frac[k, j] in (prec_type, 3):
                rho = p[k] / (RD_PRECIP * t[k])
                mxratio[k, j] = (flux[k] * (rho / rho0) ** g_x
                                 * sigma) ** one_over_xip1 / rho
                if flux[k] != 0:
                    lam = (a_x * c_x * (rho0 / rho) ** g_x * n_ax * g1
                           / flux[k]) ** (1.0 / delta)
                    reff[k, j] = gamma_4_3_2 / lam
    return mxratio, reff


class PrecipMxratioTest(unittest.TestCase):
    def test_matches_fortran_transliteration(self):
        rng = np.random.default_rng(5)
        nlev, ncol = 8, 6
        p = np.linspace(2e4, 1e5, nlev)
        t = np.linspace(220, 290, nlev)
        prec_frac = rng.integers(0, 4, (nlev, ncol))
        flux = np.where(rng.uniform(size=nlev) > 0.3,
                        rng.uniform(0, 3e-4, nlev), 0.0)
        for i in (config.I_LSRAIN, config.I_LSSNOW, config.I_CVRAIN,
                  config.I_CVSNOW, config.I_LSGRPL):
            args = [float(config.PRECIP_N_AX[i]), float(config.PRECIP_N_BX[i]),
                    float(config.PRECIP_ALPHA_X[i]), float(config.PRECIP_C_X[i]),
                    float(config.PRECIP_D_X[i]), float(config.PRECIP_G_X[i]),
                    float(config.PRECIP_A_X[i]), float(config.PRECIP_B_X[i]),
                    float(config.PRECIP_GAMMA_1[i]), float(config.PRECIP_GAMMA_2[i]),
                    float(config.PRECIP_GAMMA_3[i]), float(config.PRECIP_GAMMA_4[i])]
            ptype = float(config.PRECIP_TYPE[i])
            ref_mx, ref_re = precip_mxratio_reference(
                p, t, prec_frac, ptype, *args, flux)
            mx, re = inputs.precip_mxratio(
                p, t, prec_frac, ptype, flux, None,
                n_ax=args[0], n_bx=args[1], alpha_x=args[2], c_x=args[3],
                d_x=args[4], g_x=args[5], a_x=args[6], b_x=args[7],
                gamma1=args[8], gamma2=args[9], gamma3=args[10], gamma4=args[11])
            np.testing.assert_allclose(np.asarray(mx), ref_mx, rtol=1e-6,
                                       atol=1e-30, err_msg=config.HYDRO_NAMES[i])
            np.testing.assert_allclose(np.asarray(re), ref_re, rtol=1e-6,
                                       atol=1e-30, err_msg=config.HYDRO_NAMES[i])

    def test_rain_mixing_ratio_magnitude(self):
        # 1 mm/hr of rain at the surface is a few 1e-5 kg/kg of rain water.
        i = config.I_LSRAIN
        flux = np.array([2.8e-4])  # kg m-2 s-1 ~ 1 mm/hr
        mx, _ = inputs.precip_mxratio(
            np.array([1e5]), np.array([285.0]),
            np.ones((1, 1), dtype=int), 1.0, flux, None,
            n_ax=float(config.PRECIP_N_AX[i]), n_bx=float(config.PRECIP_N_BX[i]),
            alpha_x=float(config.PRECIP_ALPHA_X[i]), c_x=float(config.PRECIP_C_X[i]),
            d_x=float(config.PRECIP_D_X[i]), g_x=float(config.PRECIP_G_X[i]),
            a_x=float(config.PRECIP_A_X[i]), b_x=float(config.PRECIP_B_X[i]),
            gamma1=float(config.PRECIP_GAMMA_1[i]), gamma2=float(config.PRECIP_GAMMA_2[i]),
            gamma3=float(config.PRECIP_GAMMA_3[i]), gamma4=float(config.PRECIP_GAMMA_4[i]))
        self.assertGreater(float(mx[0, 0]), 1e-6)
        self.assertLess(float(mx[0, 0]), 1e-3)


class DistributeHydrometeorsTest(unittest.TestCase):
    def test_conserves_gridbox_mean_condensate(self):
        # mean over subcolumns of in-cloud values re-creates the gridbox mean.
        rng = np.random.default_rng(9)
        nlev, ncol = 6, 200
        frac_out = rng.integers(0, 3, (nlev, ncol))
        prec_frac = np.zeros((nlev, ncol), dtype=int)
        p = np.linspace(3e4, 1e5, nlev)
        t = np.linspace(230, 290, nlev)
        mr_lsliq = rng.uniform(0, 1e-4, nlev)
        mr_lsice = rng.uniform(0, 1e-4, nlev)
        zeros = np.zeros(nlev)
        hyd = inputs.distribute_hydrometeors(
            frac_out, prec_frac, p, t, mr_lsliq, mr_lsice, zeros, zeros,
            zeros, zeros, zeros, zeros, zeros)
        got = np.asarray(hyd.mr_hydro[..., config.I_LSCLIQ]).mean(axis=1)
        has_ls = (frac_out == 1).any(axis=1)
        np.testing.assert_allclose(got[has_ls], mr_lsliq[has_ls], rtol=1e-6)
        self.assertTrue((got[~has_ls] == 0).all())

    def test_frac_precip_ice_bounds(self):
        rng = np.random.default_rng(19)
        nlev, ncol = 6, 30
        frac_out = rng.integers(0, 3, (nlev, ncol))
        ls = rng.uniform(0, 3e-4, nlev)
        prec_frac = np.asarray(prec_scops(frac_out, ls, np.zeros(nlev)))
        p = np.linspace(3e4, 1e5, nlev)
        t = np.linspace(230, 290, nlev)
        zeros = np.zeros(nlev)
        hyd = inputs.distribute_hydrometeors(
            frac_out, prec_frac, p, t, zeros, zeros, zeros, zeros,
            ls, ls * 0.5, zeros, zeros, zeros)
        fpi = np.asarray(hyd.frac_precip_ice)
        self.assertTrue(((fpi >= 0) & (fpi <= 1)).all())


if __name__ == "__main__":
    unittest.main()
