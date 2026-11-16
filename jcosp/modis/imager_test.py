"""Tests for the imager water-path retrieval (``jcosp.modis.imager``)."""

import unittest

import jax
import jax.numpy as jnp
import numpy as np

from jcosp import config
from jcosp.modis import imager, optics
from jcosp.modis import simulator as msim
from jcosp.modis.simulator_test import _random_scene

_CCI = imager.CLOUD_CCI


def _single_phase(liquid, re_um, tau_layer=2.0, nlev=4, ncol=5):
    """Subcolumn retrievals of a uniform single-phase cloud of radius re_um."""
    phase = config.MODIS_PHASE_LIQUID if liquid else config.MODIS_PHASE_ICE
    re = jnp.full((nlev, ncol), re_um)
    tau = np.full((nlev, ncol), tau_layer)
    liq = np.full((nlev, ncol), 1.0 if liquid else 0.0)
    phalf = np.linspace(30000.0, 90000.0, nlev + 1)
    return msim.modis_subcolumn(
        phalf, tau, liq, np.asarray(optics.get_g_nir(phase, re)),
        np.asarray(optics.get_ssa_nir(phase, re)))


class ImagerWaterPathsTest(unittest.TestCase):
    def test_liquid_cloud_matches_modis_path(self):
        # Cloud_cci's liquid conversion (4/3 rho / Q_e, Q_e = 2) is MODIS's
        # (2/3 rho), so an all-liquid overcast scene gives the same path.
        sub = _single_phase(True, 12.0)
        lwp, iwp = imager.imager_water_paths(sub)
        col = msim.modis_column(sub.phase, sub.cloud_top_pressure, sub.tau,
                                sub.size)
        np.testing.assert_allclose(lwp, col.lwp, rtol=1e-12)
        np.testing.assert_allclose(lwp, 2 / 3 * 1000.0 * 8.0 * 12e-6,
                                   rtol=0.02)
        self.assertEqual(float(iwp), 0.0)

    def test_ice_cloud_uses_the_products_constants(self):
        sub = _single_phase(False, 40.0)
        lwp, iwp = imager.imager_water_paths(sub)
        np.testing.assert_allclose(
            iwp, 4 / 3 * 8.0 * 40e-6 * 916.7 / 2.1, rtol=0.02)
        self.assertEqual(float(lwp), 0.0)

    def test_liquid_under_an_ice_top_is_reported_as_ice(self):
        nlev, ncol = 4, 3
        liq = np.zeros((nlev, ncol))
        liq[1:] = 1.0                      # ice layer over three liquid ones
        tau = np.full((nlev, ncol), 2.0)   # top layer alone exceeds tau = 1
        _, g, w0 = optics.modis_optics(tau * liq, np.full_like(tau, 10.0),
                                       tau * (1 - liq), np.full_like(tau, 40.0))
        sub = msim.modis_subcolumn(np.linspace(3e4, 9e4, nlev + 1), tau, liq,
                                   np.asarray(g), np.asarray(w0))
        lwp, iwp = imager.imager_water_paths(sub)
        self.assertEqual(float(lwp), 0.0)
        # The whole column's optical depth, at the ice-table size.
        np.testing.assert_allclose(
            iwp, np.mean(4 / 3 * 8.0 * np.asarray(sub.size_if_ice)
                         * 916.7 / 2.1), rtol=1e-12)

    def test_every_detected_subcolumn_has_a_phase(self):
        # MODIS leaves cloud-top liquid shares in (0.3, 0.7) out of both
        # paths ("undetermined"); the two-phase rule books them all.
        rng = np.random.default_rng(3)
        tau, liq, g, w0, phalf = _random_scene(rng, ncol=200)
        sub = msim.modis_subcolumn(phalf, tau, liq, g, w0)
        self.assertTrue(
            (np.asarray(sub.phase) == config.MODIS_PHASE_UNDETERMINED).any())
        lwp, iwp = imager.imager_water_paths(sub)
        det = np.asarray(sub.phase) != config.MODIS_PHASE_NONE
        is_liq = np.asarray(sub.liquid_fraction) > 0.5
        t = np.minimum(np.where(det, np.asarray(sub.tau), 0.0), 256.0)
        s_l, s_i = np.asarray(sub.size_if_liquid), np.asarray(sub.size_if_ice)
        ref_l = np.where(det & is_liq & (s_l > 0),
                         4 / 3 * t * s_l * 1000.0 / 2.0, 0.0).mean()
        ref_i = np.where(det & ~is_liq & (s_i > 0),
                         4 / 3 * t * s_i * 916.7 / 2.1, 0.0).mean()
        np.testing.assert_allclose(lwp, ref_l, rtol=1e-12)
        np.testing.assert_allclose(iwp, ref_i, rtol=1e-12)
        self.assertGreater(float(lwp), 0.0)
        self.assertGreater(float(iwp), 0.0)

    def test_optical_depth_is_capped_and_clear_sky_is_zero(self):
        thick = _single_phase(True, 12.0, tau_layer=100.0)   # column tau 400
        capped = _single_phase(True, 12.0, tau_layer=64.0)   # column tau 256
        self.assertEqual(float(jnp.sum(thick.tau[0])) > 256.0, True)
        np.testing.assert_allclose(
            imager.imager_water_paths(thick)[0]
            / np.mean(np.asarray(thick.size_if_liquid)),
            imager.imager_water_paths(capped)[0]
            / np.mean(np.asarray(capped.size_if_liquid)), rtol=1e-12)
        clear = _single_phase(True, 12.0, tau_layer=0.01)
        lwp, iwp = imager.imager_water_paths(clear)
        self.assertEqual((float(lwp), float(iwp)), (0.0, 0.0))

    def test_a_product_is_its_constants(self):
        sub = _single_phase(False, 40.0)
        doubled = _CCI._replace(density_ice=2 * _CCI.density_ice)
        np.testing.assert_allclose(
            imager.imager_water_paths(sub, doubled)[1],
            2 * imager.imager_water_paths(sub)[1], rtol=1e-12)

    def test_gradients_are_finite(self):
        # Masked-out subcolumns carry tau = R_UNDEF and failed sizes; none
        # may leak into the derivative.
        rng = np.random.default_rng(5)
        tau, liq, g, w0, phalf = _random_scene(rng)

        def total(tau_in):
            sub = msim.modis_subcolumn(phalf, tau_in, liq, g, w0)
            lwp, iwp = imager.imager_water_paths(sub)
            return lwp + iwp

        grad = np.asarray(jax.grad(total)(jnp.asarray(tau)))
        self.assertTrue(np.isfinite(grad).all())
        self.assertGreater(np.abs(grad).max(), 0.0)


if __name__ == "__main__":
    unittest.main()
