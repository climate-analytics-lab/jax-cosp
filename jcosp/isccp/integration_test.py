"""Integration test: ISCCP inside simulate_cloudsat_modis (CTP substitution).

Verifies the cosp.F90 wiring: with ``run_isccp=True`` the ISCCP simulator
runs on the same SCOPS realization and its ``boxptop`` replaces the MODIS
CO2-slicing cloud-top pressure of clouds below 700 hPa, changing the MODIS
CTP statistics. Requires the generated Z-scale LUT (as the other end-to-end
tests do).
"""

import importlib.resources
import unittest

import jax
import jax.numpy as jnp
import numpy as np

from jcosp import config
from jcosp.constants import R_UNDEF
from jcosp.simulator import CloudsatInputs, simulate_cloudsat_modis

_LUT_PATH = importlib.resources.files("jcosp") / "data" / "quickbeam_lut_94GHz_1M.npz"


def low_cloud_state(nlev=30, nbatch=2):
    """Build a warm liquid cloud entirely below 700 hPa (0.4-1.8 km)."""
    zhalf_1d = np.linspace(20000.0, 0.0, nlev + 1)
    zhalf = np.repeat(zhalf_1d[1:, None], nbatch, axis=1)  # layer bottoms
    zfull = zhalf + 0.5 * (zhalf_1d[:-1] - zhalf_1d[1:])[:, None]
    pressure = 101325.0 * np.exp(-zfull / 7500.0)
    temperature = np.clip(288.0 - 6.5e-3 * zfull, 210.0, None)
    q = 0.005 * (pressure / 101325.0) ** 2

    cloudy = (zfull > 400.0) & (zfull < 1800.0)
    cloud_frac = np.where(cloudy, 0.7, 0.0)
    mr_lsliq = np.where(cloudy, 3e-4, 0.0)
    zeros = np.zeros((nlev, nbatch))
    reff = np.zeros((nlev, nbatch, config.N_HYDRO))
    reff[..., config.I_LSCLIQ] = np.where(mr_lsliq > 0, 12e-6, 0.0)
    state = CloudsatInputs(
        pressure=pressure, temperature=temperature, specific_humidity=q,
        zfull=zfull, zhalf=zhalf, cloud_frac=cloud_frac, conv_frac=zeros,
        mr_lsliq=mr_lsliq, mr_lsice=zeros, mr_ccliq=zeros, mr_ccice=zeros,
        fl_lsrain=zeros, fl_lssnow=zeros, fl_lsgrpl=zeros,
        fl_ccrain=zeros, fl_ccsnow=zeros,
        land=np.zeros(nbatch), surfelev=np.zeros(nbatch),
        t2m=np.full(nbatch, 288.0), reff=reff)
    dtau_s = np.where(mr_lsliq > 0, 6.0, 0.0)
    dem_s = np.where(mr_lsliq > 0, 1.0 - np.exp(-6.0 / 2.13), 0.0)
    phalf = np.repeat((101325.0 * np.exp(-zhalf_1d / 7500.0))[:, None],
                      nbatch, axis=1)
    return state, dtau_s, dem_s, phalf


@unittest.skipUnless(_LUT_PATH.is_file(),
                     "Z-scale LUT not generated (python -m jcosp.radar.lut.generate)")
class IsccpModisIntegrationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from jcosp.radar.optics import load_lut
        cls.lut = load_lut(_LUT_PATH)

    def test_isccp_ctp_substitution_changes_modis_ctp(self):
        state, dtau_s, dem_s, phalf = low_cloud_state()
        zeros = np.zeros_like(dtau_s)
        common = dict(key=jax.random.PRNGKey(11), ncolumns=40)
        base = simulate_cloudsat_modis(
            state, self.lut, jnp.asarray(dtau_s), jnp.asarray(zeros),
            jnp.asarray(phalf), **common)
        wisc = simulate_cloudsat_modis(
            state, self.lut, jnp.asarray(dtau_s), jnp.asarray(zeros),
            jnp.asarray(phalf), run_isccp=True,
            dem_s=dem_s, dem_c=zeros, skt=state.t2m, emsfc_lw=0.98,
            **common)

        # Same SCOPS realization on both runs.
        np.testing.assert_array_equal(np.asarray(base.cloudsat.frac_out),
                                      np.asarray(wisc.cloudsat.frac_out))
        self.assertIsNone(base.isccp)
        self.assertIsNotNone(wisc.isccp)

        # The ISCCP column sees the low cloud.
        tca = np.asarray(wisc.isccp.column.totalcldarea)
        self.assertTrue((tca > 30.0).all() and (tca <= 100.0).all())
        boxptop = np.asarray(wisc.isccp.subcolumn.boxptop)
        defined = boxptop != R_UNDEF
        self.assertTrue(defined.any())
        # Cloud tops of a 0.4-1.8 km cloud: roughly 780-1000 hPa.
        self.assertTrue((boxptop[defined] > 70000.0).all())
        self.assertTrue((boxptop[defined] < 101325.0).all())

        # MODIS low-cloud CTP substitution: cloudy subcolumns whose
        # CO2-slicing CTP is below 700 hPa now carry the ISCCP boxptop.
        ctp0 = np.asarray(base.modis.subcolumn.cloud_top_pressure)
        ctp1 = np.asarray(wisc.modis.subcolumn.cloud_top_pressure)
        sub = (ctp0 != R_UNDEF) & (ctp0 > config.MODIS_CO2SLICING_PRESSURE_LIMIT)
        self.assertTrue(sub.any(), "no low-cloud subcolumns to substitute")
        np.testing.assert_array_equal(ctp1[sub], boxptop[sub])
        np.testing.assert_array_equal(ctp1[~sub], ctp0[~sub])
        self.assertTrue((ctp1[sub] != ctp0[sub]).any(),
                        "substitution did not change any CTP")

        # ...and the column-level CTP statistic shifts.
        m0 = np.asarray(base.modis.column.ctp_mean)
        m1 = np.asarray(wisc.modis.column.ctp_mean)
        self.assertTrue((m0 != m1).any())

        # The ISCCP tau/CTP histogram puts the cloud in low-pressure bins:
        # surface-first flipped axis -> low clouds at pressure index 0.
        fq = np.asarray(wisc.isccp.column.fq_isccp)
        self.assertTrue((fq[:, 0].sum(axis=0) > 0).all())
        np.testing.assert_allclose(fq[1:].sum(axis=(0, 1)), tca, rtol=1e-9)

    def test_run_isccp_requires_inputs(self):
        state, dtau_s, dem_s, phalf = low_cloud_state()
        with self.assertRaises(ValueError):
            simulate_cloudsat_modis(
                state, self.lut, jnp.asarray(dtau_s),
                jnp.asarray(np.zeros_like(dtau_s)), jnp.asarray(phalf),
                key=jax.random.PRNGKey(0), ncolumns=10, run_isccp=True)


if __name__ == "__main__":
    unittest.main()
