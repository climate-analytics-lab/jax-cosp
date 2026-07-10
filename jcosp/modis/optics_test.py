"""Tests for the MODIS input optics against Fortran loop transliterations.

References transliterate ``cosp_optics.F90`` (cosp_simulator_optics,
MODIS_OPTICS_PARTITION, modis_optics) and the ``modis_simulator.F90``
Collection-6 near-IR fits (get_g_nir / get_ssa_nir).
"""

import unittest

import jax
import jax.numpy as jnp
import numpy as np

from jcosp import config
from jcosp.modis import optics


# ---------------------------------------------------------------------------
# Fortran transliterations
# ---------------------------------------------------------------------------
def _quad(x, c):
    return c[0] + x * (c[1] + x * c[2])


def _cubic(x, c):
    return c[0] + x * (c[1] + x * (c[2] + x * c[3]))


def get_g_nir_reference(phase, re):
    """Scalar transliteration of modis_simulator.F90:get_g_nir (C6 fits)."""
    ice = (0.7490, 6.5153e-3, -5.4136e-5)
    small_water = (1.0364, -8.8800e-2, 7.0000e-3)
    big_water = (0.6035, 2.8993e-2, -1.1051e-3, 1.5134e-5)
    if phase == config.MODIS_PHASE_LIQUID:
        if re < 7.0:
            g = _quad(re, small_water)
            if re < config.MODIS_RE_WATER_MIN:
                g = _quad(config.MODIS_RE_WATER_MIN, small_water)
        else:
            g = _cubic(re, big_water)
            if re > config.MODIS_RE_WATER_MAX:
                g = _cubic(config.MODIS_RE_WATER_MAX, big_water)
        return g
    g = _quad(re, ice)
    if re < config.MODIS_RE_ICE_MIN:
        g = _quad(config.MODIS_RE_ICE_MIN, ice)
    if re > config.MODIS_RE_ICE_MAX:
        g = _quad(config.MODIS_RE_ICE_MAX, ice)
    return g


def get_ssa_nir_reference(phase, re):
    """Scalar transliteration of modis_simulator.F90:get_ssa_nir (C6 fits)."""
    ice = (0.9625, -1.8069e-2, 3.3281e-4, -2.2865e-6)
    water = (1.0044, -1.1397e-2, 1.3300e-4)
    if phase == config.MODIS_PHASE_LIQUID:
        w = _quad(re, water)
        if re < config.MODIS_RE_WATER_MIN:
            w = _quad(config.MODIS_RE_WATER_MIN, water)
        if re > config.MODIS_RE_WATER_MAX:
            w = _quad(config.MODIS_RE_WATER_MAX, water)
        return w
    w = _cubic(re, ice)
    if re < config.MODIS_RE_ICE_MIN:
        w = _cubic(config.MODIS_RE_ICE_MIN, ice)
    if re > config.MODIS_RE_ICE_MAX:
        w = _cubic(config.MODIS_RE_ICE_MAX, ice)
    return w


def optics_partition_reference(cloud_water, cloud_ice, water_size, ice_size,
                               tau):
    """Loop transliteration of MODIS_OPTICS_PARTITION (level x column)."""
    frac_l = np.empty_like(tau)
    nlev, ncol = tau.shape
    for k in range(nlev):
        for c in range(ncol):
            if cloud_ice[k, c] <= 0.0:
                frac_l[k, c] = 1.0
            elif cloud_water[k, c] <= 0.0:
                frac_l[k, c] = 0.0
            else:
                wl = cloud_water[k, c] / water_size[k, c]
                wi = cloud_ice[k, c] / (0.93 * ice_size[k, c])
                frac_l[k, c] = wl / (wl + wi)
    tau_l = frac_l * tau
    return tau_l, tau - tau_l


def modis_optics_reference(tau_liq, size_liq, tau_ice, size_ice):
    """Loop transliteration of cosp_optics.F90:modis_optics."""
    g = np.zeros_like(tau_liq)
    w0 = np.zeros_like(tau_liq)
    frac_liq = np.zeros_like(tau_liq)
    nlev, ncol = tau_liq.shape
    for k in range(nlev):
        for c in range(ncol):
            wg = get_g_nir_reference(config.MODIS_PHASE_LIQUID, size_liq[k, c])
            ww = get_ssa_nir_reference(config.MODIS_PHASE_LIQUID, size_liq[k, c])
            ig = get_g_nir_reference(config.MODIS_PHASE_ICE, size_ice[k, c])
            iw = get_ssa_nir_reference(config.MODIS_PHASE_ICE, size_ice[k, c])
            tau = tau_liq[k, c] + tau_ice[k, c]
            if tau > 0:
                w0[k, c] = (tau_liq[k, c] * ww + tau_ice[k, c] * iw) / tau
                g[k, c] = (tau_liq[k, c] * wg * ww
                           + tau_ice[k, c] * ig * iw) / (w0[k, c] * tau)
                frac_liq[k, c] = tau_liq[k, c] / tau
    return frac_liq, g, w0


class NirFitsTest(unittest.TestCase):
    def test_match_transliteration(self):
        res = np.concatenate([np.linspace(0.5, 100.0, 61), [4.0, 5.0, 7.0,
                                                            30.0, 90.0]])
        for phase in (config.MODIS_PHASE_LIQUID, config.MODIS_PHASE_ICE):
            g = np.asarray(optics.get_g_nir(phase, jnp.asarray(res)))
            w = np.asarray(optics.get_ssa_nir(phase, jnp.asarray(res)))
            g_ref = [get_g_nir_reference(phase, r) for r in res]
            w_ref = [get_ssa_nir_reference(phase, r) for r in res]
            np.testing.assert_allclose(g, g_ref, rtol=1e-14,
                                       err_msg=f"g, phase={phase}")
            np.testing.assert_allclose(w, w_ref, rtol=1e-14,
                                       err_msg=f"w0, phase={phase}")

    def test_physical_range(self):
        res = jnp.linspace(1.0, 120.0, 200)
        for phase in (config.MODIS_PHASE_LIQUID, config.MODIS_PHASE_ICE):
            g = np.asarray(optics.get_g_nir(phase, res))
            w = np.asarray(optics.get_ssa_nir(phase, res))
            self.assertTrue(((g > 0) & (g < 1)).all())
            self.assertTrue(((w > 0) & (w <= 1)).all())


class SimulatorOpticsTest(unittest.TestCase):
    def test_merge_2d_and_3d(self):
        rng = np.random.default_rng(0)
        nlev, ncol = 5, 7
        frac = rng.integers(0, 3, (nlev, ncol))
        strat = rng.uniform(1, 2, nlev)
        conv = rng.uniform(3, 4, nlev)
        out = np.asarray(optics.simulator_optics(frac, strat, conv))
        # Loop transliteration of cosp_simulator_optics2D.
        ref = np.zeros((nlev, ncol))
        for c in range(ncol):
            ref[frac[:, c] == 1, c] = strat[frac[:, c] == 1]
            ref[frac[:, c] == 2, c] = conv[frac[:, c] == 2]
        np.testing.assert_array_equal(out, ref)
        # 3D variant: subcolumn-resolved inputs pass through the same merge.
        strat3 = rng.uniform(1, 2, (nlev, ncol))
        conv3 = rng.uniform(3, 4, (nlev, ncol))
        out3 = np.asarray(optics.simulator_optics(frac, strat3, conv3))
        ref3 = np.where(frac == 1, strat3, np.where(frac == 2, conv3, 0.0))
        np.testing.assert_array_equal(out3, ref3)


class OpticsPartitionTest(unittest.TestCase):
    def _fields(self, rng, nlev=6, ncol=8):
        cw = np.where(rng.uniform(size=(nlev, ncol)) < 0.6,
                      rng.uniform(1e-5, 1e-3, (nlev, ncol)), 0.0)
        ci = np.where(rng.uniform(size=(nlev, ncol)) < 0.6,
                      rng.uniform(1e-5, 1e-3, (nlev, ncol)), 0.0)
        ws = rng.uniform(5e-6, 20e-6, (nlev, ncol))
        s = rng.uniform(15e-6, 60e-6, (nlev, ncol))
        tau = rng.uniform(0.0, 5.0, (nlev, ncol))
        return cw, ci, ws, s, tau

    def test_matches_transliteration(self):
        rng = np.random.default_rng(1)
        cw, ci, ws, s, tau = self._fields(rng)
        tl, ti = optics.optics_partition(cw, ci, ws, s, tau)
        tl_ref, ti_ref = optics_partition_reference(cw, ci, ws, s, tau)
        np.testing.assert_allclose(np.asarray(tl), tl_ref, rtol=1e-14)
        np.testing.assert_allclose(np.asarray(ti), ti_ref, rtol=1e-14)

    def test_partition_conserves_tau(self):
        rng = np.random.default_rng(2)
        cw, ci, ws, s, tau = self._fields(rng)
        tl, ti = optics.optics_partition(cw, ci, ws, s, tau)
        np.testing.assert_allclose(np.asarray(tl) + np.asarray(ti), tau,
                                   rtol=1e-14)
        self.assertTrue((np.asarray(tl) >= 0).all())
        self.assertTrue((np.asarray(ti) >= 0).all())

    def test_gradient_finite(self):
        rng = np.random.default_rng(3)
        cw, ci, ws, s, tau = self._fields(rng)

        def liq_tau_sum(cw_):
            tl, _ = optics.optics_partition(cw_, ci, ws, s, tau)
            return jnp.sum(tl)

        g = np.asarray(jax.grad(liq_tau_sum)(jnp.asarray(cw)))
        self.assertTrue(np.isfinite(g).all())
        self.assertGreater(np.abs(g).max(), 0.0)


class ModisOpticsTest(unittest.TestCase):
    def test_matches_transliteration(self):
        rng = np.random.default_rng(4)
        nlev, ncol = 6, 8
        tl = np.where(rng.uniform(size=(nlev, ncol)) < 0.7,
                      rng.uniform(0, 3, (nlev, ncol)), 0.0)
        ti = np.where(rng.uniform(size=(nlev, ncol)) < 0.7,
                      rng.uniform(0, 3, (nlev, ncol)), 0.0)
        sl = rng.uniform(2.0, 40.0, (nlev, ncol))    # microns
        si = rng.uniform(4.0, 100.0, (nlev, ncol))
        frac_liq, g, w0 = optics.modis_optics(tl, sl, ti, si)
        f_ref, g_ref, w_ref = modis_optics_reference(tl, sl, ti, si)
        np.testing.assert_allclose(np.asarray(frac_liq), f_ref, rtol=1e-13)
        np.testing.assert_allclose(np.asarray(g), g_ref, rtol=1e-13)
        np.testing.assert_allclose(np.asarray(w0), w_ref, rtol=1e-13)

    def test_pure_phases(self):
        # Pure liquid: w0/g equal the liquid fits; fracLIQ = 1.
        tau = np.full((3, 2), 2.0)
        size = np.full((3, 2), 10.0)
        frac_liq, g, w0 = optics.modis_optics(tau, size, 0 * tau, size)
        np.testing.assert_allclose(np.asarray(frac_liq), 1.0)
        np.testing.assert_allclose(
            np.asarray(g),
            np.asarray(optics.get_g_nir(config.MODIS_PHASE_LIQUID,
                                        jnp.asarray(size))), rtol=1e-14)
        np.testing.assert_allclose(
            np.asarray(w0),
            np.asarray(optics.get_ssa_nir(config.MODIS_PHASE_LIQUID,
                                          jnp.asarray(size))), rtol=1e-14)


class BuildModisOpticsTest(unittest.TestCase):
    def test_driver_block_assembly(self):
        rng = np.random.default_rng(5)
        nlev, ncol = 6, 10
        frac_out = rng.integers(0, 3, (nlev, ncol))
        mr = np.zeros((nlev, ncol, config.N_HYDRO))
        reff = np.zeros((nlev, ncol, config.N_HYDRO))
        for i in (config.I_LSCLIQ, config.I_LSCICE):
            mr[..., i] = np.where(frac_out == 1, rng.uniform(
                1e-5, 1e-3, (nlev, ncol)), 0.0)
            reff[..., i] = np.where(frac_out == 1, rng.uniform(
                5e-6, 3e-5, (nlev, ncol)), 0.0)
        for i in (config.I_CVCLIQ, config.I_CVCICE):
            mr[..., i] = np.where(frac_out == 2, rng.uniform(
                1e-5, 1e-3, (nlev, ncol)), 0.0)
            reff[..., i] = np.where(frac_out == 2, rng.uniform(
                5e-6, 3e-5, (nlev, ncol)), 0.0)
        dtau_s = rng.uniform(0, 2, nlev)
        dtau_c = rng.uniform(0, 4, nlev)

        out = optics.build_modis_optics(frac_out, mr, reff, dtau_s, dtau_c)
        # tau painted through the SCOPS map...
        np.testing.assert_array_equal(
            np.asarray(out.tau_067),
            np.where(frac_out == 1, dtau_s[:, None],
                     np.where(frac_out == 2, dtau_c[:, None], 0.0)))
        # ... partition conserves it, and only cloud classes are read.
        np.testing.assert_allclose(
            np.asarray(out.tau_liq) + np.asarray(out.tau_ice),
            np.asarray(out.tau_067), rtol=1e-14)
        np.testing.assert_array_equal(
            np.asarray(out.cloud_water),
            mr[..., config.I_LSCLIQ] + mr[..., config.I_CVCLIQ])
        # Clear cells carry no optics.
        clear = frac_out == 0
        for field in (out.tau_067, out.frac_liq, out.g, out.w0):
            self.assertTrue((np.asarray(field)[clear] == 0).all())


if __name__ == "__main__":
    unittest.main()
