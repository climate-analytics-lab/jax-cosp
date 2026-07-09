"""Tests for the lidar subcolumn stage.

The JAX port is checked against loop transliterations of the Fortran
``cmp_backsignal`` / ``cmp_beta`` level recursions and the perpendicular
channel assembly of ``lidar_subcolumn`` (lidar_simulator.F90), plus physical
sanity checks (clear-sky SR = 1, cloud detection and attenuation) and a
gradient check through the full optics + ATB chain.
"""

import unittest

import jax
import jax.numpy as jnp
import numpy as np

from jcosp.constants import R_UNDEF
from jcosp.lidar.diagnostics import scattering_ratio
from jcosp.lidar.optics import lidar_optics
from jcosp.lidar.simulator import (attenuated_backscatter,
                                   backscatter_coefficient, lidar_subcolumn)
from jcosp.lidar.optics_test import make_atmosphere


def ref_cmp_backsignal(beta, tau):
    """Transliteration of cmp_backsignal; Fortran layout (npoints, nlev)."""
    pnorm = np.zeros_like(beta)
    pnorm[:, 0] = beta[:, 0] / (2.0 * tau[:, 0]) * (1.0 - np.exp(-2.0 * tau[:, 0]))
    for k in range(1, beta.shape[1]):
        dtau = tau[:, k] - tau[:, k - 1]
        pnorm[:, k] = np.where(
            dtau > 0.0,
            beta[:, k] * np.exp(-2.0 * tau[:, k - 1]) / (2.0 * np.where(
                dtau > 0.0, dtau, 1.0)) * (1.0 - np.exp(-2.0 * dtau)),
            beta[:, k] * np.exp(-2.0 * tau[:, k - 1]))
    return pnorm


def ref_cmp_beta(pnorm, tau):
    """Transliteration of cmp_beta; Fortran layout (npoints, nlev)."""
    eps = np.finfo(np.float64).eps
    beta = np.zeros_like(pnorm)
    beta[:, 0] = pnorm[:, 0] * (2.0 * tau[:, 0]) / (1.0 - np.exp(-2.0 * tau[:, 0]))
    for k in range(1, pnorm.shape[1]):
        dtau = tau[:, k] - tau[:, k - 1]
        trans = np.exp(-2.0 * tau[:, k - 1])
        trans_s = np.where(trans > eps, trans, 1.0)  # guarded branch below
        inner = np.where(
            dtau > 0.0,
            pnorm[:, k] / trans_s * (2.0 * np.where(dtau > 0.0, dtau, 1.0))
            / (1.0 - np.exp(-2.0 * np.where(dtau > 0.0, dtau, 1.0))),
            pnorm[:, k] / trans_s)
        beta[:, k] = np.where(trans > eps, inner, pnorm[:, k] / eps)
    return beta


def ref_perp_total(pnorm, beta_perp_ice, beta_perp_liq, beta_mol, tautot):
    """Transliteration of the pnorm_perp_tot block of lidar_subcolumn.

    Fortran layout: pnorm/tautot/beta_perp_* (npoints, ncol, nlev),
    beta_mol (npoints, nlev).
    """
    npoints, ncol, nlev = pnorm.shape
    out = np.zeros_like(pnorm)
    depol = 1.0 + 1.0 / 0.0284
    for icol in range(ncol):
        out[:, icol, 0] = np.where(
            tautot[:, icol, 0] > 0.0,
            (beta_perp_ice[:, icol, 0] + beta_perp_liq[:, icol, 0]
             - beta_mol[:, 0] / depol) / (2.0 * tautot[:, icol, 0])
            * (1.0 - np.exp(-2.0 * tautot[:, icol, 0])), 0.0)
        for k in range(1, nlev):
            dtau = tautot[:, icol, k] - tautot[:, icol, k - 1]
            s = (beta_perp_ice[:, icol, k] + beta_perp_liq[:, icol, k]
                 - beta_mol[:, k] / depol)
            attn = np.exp(-2.0 * tautot[:, icol, k - 1])
            val = np.where(
                dtau > 0.0,
                s * attn / (2.0 * np.where(dtau > 0.0, dtau, 1.0))
                * (1.0 - np.exp(-2.0 * dtau)),
                s * attn)
            out[:, icol, k] = np.where(pnorm[:, icol, k] == 0.0, 0.0, val)
    return out


def cloudy_scene(nlev=20, ncol=4, npoints=2, seed=7, q_scale=1e-3):
    """Build a random cloudy atmosphere through the full lidar optics."""
    rng = np.random.default_rng(seed)
    pres, presf, temp = make_atmosphere(rng, nlev, npoints)
    q = np.where(rng.uniform(size=(npoints, ncol, nlev, 4)) < 0.35,
                 rng.uniform(0.1, 1.0, (npoints, ncol, nlev, 4)) * q_scale, 0.0)
    rad = rng.uniform(5e-6, 40e-6, (npoints, nlev, 4))
    opt = lidar_optics(
        *(jnp.asarray(np.transpose(q[..., i], (2, 1, 0))) for i in range(4)),
        *(jnp.asarray(rad[..., i].T) for i in range(4)),
        jnp.asarray(pres.T), jnp.asarray(presf.T), jnp.asarray(temp.T))
    return opt


class BacksignalTest(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(99)

    def _beta_tau(self, npoints=6, nlev=15):
        beta = self.rng.uniform(1e-7, 1e-2, (npoints, nlev))
        dtau = self.rng.uniform(0.0, 0.8, (npoints, nlev))
        dtau[:, 0] = np.maximum(dtau[:, 0], 1e-6)  # Fortran needs tau(1) > 0
        # Exercise the dtau == 0 fallback branch as well.
        dtau[:, 5] = 0.0
        return beta, np.cumsum(dtau, axis=1)

    def test_backsignal_matches_transliteration(self):
        beta, tau = self._beta_tau()
        ref = ref_cmp_backsignal(beta, tau)
        out = attenuated_backscatter(jnp.asarray(beta.T), jnp.asarray(tau.T))
        np.testing.assert_allclose(np.asarray(out), ref.T, rtol=1e-13)

    def test_cmp_beta_matches_transliteration(self):
        beta, tau = self._beta_tau()
        pnorm = ref_cmp_backsignal(beta, tau)
        ref = ref_cmp_beta(pnorm, tau)
        out = backscatter_coefficient(jnp.asarray(pnorm.T), jnp.asarray(tau.T))
        np.testing.assert_allclose(np.asarray(out), ref.T, rtol=1e-13)
        # Saturated column: transmission underflow path (pnorm/eps).
        tau_big = tau.copy()
        tau_big[:, -3:] += 30.0
        ref = ref_cmp_beta(pnorm, tau_big)
        out = backscatter_coefficient(jnp.asarray(pnorm.T), jnp.asarray(tau_big.T))
        np.testing.assert_allclose(np.asarray(out), ref.T, rtol=1e-13)

    def test_roundtrip_recovers_beta(self):
        beta, tau = self._beta_tau()
        atb = attenuated_backscatter(jnp.asarray(beta.T), jnp.asarray(tau.T))
        back = backscatter_coefficient(atb, jnp.asarray(tau.T))
        np.testing.assert_allclose(np.asarray(back), beta.T, rtol=1e-10)


class LidarSubcolumnTest(unittest.TestCase):
    def test_perpendicular_matches_transliteration(self):
        opt = cloudy_scene()
        sub = lidar_subcolumn(opt.beta_mol, opt.tau_mol, opt.betatot,
                              opt.tautot, opt.betatot_ice, opt.tautot_ice,
                              opt.betatot_liq, opt.tautot_liq)
        # Reference in Fortran layout.
        to_f = lambda a: np.transpose(np.asarray(a), (2, 1, 0))
        beta_mol_f = np.asarray(opt.beta_mol).T
        pnorm_ice = ref_cmp_backsignal(
            *(x.reshape(-1, x.shape[-1]) for x in
              (to_f(opt.betatot_ice), to_f(opt.tautot_ice))))
        pnorm_liq = ref_cmp_backsignal(
            *(x.reshape(-1, x.shape[-1]) for x in
              (to_f(opt.betatot_liq), to_f(opt.tautot_liq))))
        shape_f = to_f(opt.betatot).shape
        perp_ice = 0.2904 * pnorm_ice
        perp_liq = 1000.0 * 0.4099 * pnorm_liq**2 + 0.009 * pnorm_liq
        bpi = ref_cmp_beta(perp_ice,
                           to_f(opt.tautot_ice).reshape(-1, shape_f[-1]))
        bpl = ref_cmp_beta(perp_liq,
                           to_f(opt.tautot_liq).reshape(-1, shape_f[-1]))
        ref = ref_perp_total(to_f(sub.pnorm), bpi.reshape(shape_f),
                             bpl.reshape(shape_f), beta_mol_f, to_f(opt.tautot))
        np.testing.assert_allclose(to_f(sub.pnorm_perp), ref, rtol=1e-12,
                                   atol=1e-300)

    def test_clear_sky_sr_is_one(self):
        # No condensate: pnorm == pmol per subcolumn, so SR == 1 exactly
        # wherever it is defined (away from the R_UNDEF/below-ground flags).
        rng = np.random.default_rng(3)
        nlev, ncol, npoints = 16, 3, 2
        pres, presf, temp = make_atmosphere(rng, nlev, npoints)
        zeros = jnp.zeros((nlev, ncol, npoints))
        rad = jnp.full((nlev, npoints), 10e-6)
        opt = lidar_optics(zeros, zeros, zeros, zeros, rad, rad, rad, rad,
                           jnp.asarray(pres.T), jnp.asarray(presf.T),
                           jnp.asarray(temp.T))
        sub = lidar_subcolumn(opt.beta_mol, opt.tau_mol, opt.betatot,
                              opt.tautot, opt.betatot_ice, opt.tautot_ice,
                              opt.betatot_liq, opt.tautot_liq)
        sr = np.asarray(scattering_ratio(sub.pnorm, sub.pmol))
        np.testing.assert_allclose(sr, 1.0, rtol=1e-12)
        # The molecular ATB is attenuated: less than the unattenuated
        # backscatter, monotonically more so toward the surface.
        ratio = np.asarray(sub.pmol) / np.asarray(opt.beta_mol)
        self.assertTrue(np.all(ratio < 1.0))
        self.assertTrue(np.all(np.diff(ratio, axis=0) < 0.0))

    def test_thick_cloud_detected_then_attenuated(self):
        # A very thick liquid cloud at levels 8-9: SR >> 5 at cloud top,
        # then near-total attenuation below (SR < S_att = 0.01).
        rng = np.random.default_rng(11)
        nlev, ncol, npoints = 20, 2, 1
        pres, presf, temp = make_atmosphere(rng, nlev, npoints)
        q = np.zeros((nlev, ncol, npoints))
        q[8:10] = 5e-3  # kg/kg, all subcolumns
        rad = jnp.full((nlev, npoints), 10e-6)
        zeros = jnp.zeros_like(jnp.asarray(q))
        opt = lidar_optics(jnp.asarray(q), zeros, zeros, zeros,
                           rad, rad, rad, rad, jnp.asarray(pres.T),
                           jnp.asarray(presf.T), jnp.asarray(temp.T))
        sub = lidar_subcolumn(opt.beta_mol, opt.tau_mol, opt.betatot,
                              opt.tautot, opt.betatot_ice, opt.tautot_ice,
                              opt.betatot_liq, opt.tautot_liq)
        sr = np.asarray(scattering_ratio(sub.pnorm, sub.pmol))
        self.assertTrue(np.all(sr[8] > 5.0))          # cloud detected
        np.testing.assert_allclose(sr[:8], 1.0, rtol=1e-12)  # clear above
        self.assertTrue(np.all(sr[12:] < 0.01))       # fully attenuated below
        # Perpendicular channel is positive in the (liquid) cloud.
        self.assertTrue(np.all(np.asarray(sub.pnorm_perp)[8:10] > 0.0))

    def test_no_phase_optics_skips_perpendicular(self):
        opt = cloudy_scene()
        sub = lidar_subcolumn(opt.beta_mol, opt.tau_mol, opt.betatot,
                              opt.tautot)
        self.assertIsNone(sub.pnorm_perp)

    def test_gradient_through_atb_finite_nonzero(self):
        rng = np.random.default_rng(21)
        nlev, ncol, npoints = 12, 2, 1
        pres, presf, temp = make_atmosphere(rng, nlev, npoints)
        rad = jnp.full((nlev, npoints), 12e-6)
        q0 = jnp.zeros((nlev, ncol, npoints)).at[5].set(2e-4)

        def total_atb(qliq):
            zeros = jnp.zeros_like(qliq)
            opt = lidar_optics(qliq, zeros, zeros, zeros, rad, rad, rad, rad,
                               jnp.asarray(pres.T), jnp.asarray(presf.T),
                               jnp.asarray(temp.T))
            sub = lidar_subcolumn(opt.beta_mol, opt.tau_mol, opt.betatot,
                                  opt.tautot, opt.betatot_ice, opt.tautot_ice,
                                  opt.betatot_liq, opt.tautot_liq)
            return jnp.sum(sub.pnorm) + jnp.sum(sub.pnorm_perp)

        g = np.asarray(jax.grad(total_atb)(q0))
        self.assertTrue(np.all(np.isfinite(g)))
        self.assertGreater(np.abs(g).max(), 0.0)
        # Finite-difference cross-check on the cloud cell.
        eps = 1e-9
        f0 = float(total_atb(q0))
        f1 = float(total_atb(q0.at[5, 0, 0].add(eps)))
        np.testing.assert_allclose(g[5, 0, 0], (f1 - f0) / eps, rtol=1e-4)


class ScatteringRatioFlagTest(unittest.TestCase):
    def test_undefined_where_molecular_missing(self):
        pnorm = jnp.full((3, 1), 1e-5)          # (nlev, ncol)
        betamol = jnp.array([1e-5, 0.0, 1e30])  # (nlev,)
        sr = np.asarray(scattering_ratio(pnorm, betamol))
        self.assertEqual(sr[0, 0], 1.0)
        self.assertEqual(sr[1, 0], R_UNDEF)  # betamol not positive
        self.assertEqual(sr[2, 0], R_UNDEF)  # betamol >= xmax


if __name__ == "__main__":
    unittest.main()
