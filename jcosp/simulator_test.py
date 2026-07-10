"""End-to-end CloudSat simulator tests (require the generated Z-scale LUT)."""

import importlib.resources
import unittest

import jax
import jax.numpy as jnp
import numpy as np

from jcosp import config
from jcosp.constants import R_UNDEF
from jcosp.simulator import (CloudsatInputs, simulate_cloudsat,
                             simulate_cloudsat_modis)

_LUT_PATH = importlib.resources.files("jcosp") / "data" / "quickbeam_lut_94GHz_1M.npz"


def idealized_state(nlev=30, nbatch=4, seed=0):
    """Build a moist mid-latitude-ish profile with a raining warm cloud."""
    # TOA-first heights: nlev layers spanning ~20 km down to the surface.
    zhalf_1d = np.linspace(20000.0, 0.0, nlev + 1)
    zhalf = np.repeat(zhalf_1d[1:, None], nbatch, axis=1)  # layer bottoms
    zfull = zhalf + 0.5 * (zhalf_1d[:-1] - zhalf_1d[1:])[:, None]
    # Hydrostatic-ish pressure/temperature and a simple humidity profile.
    pressure = 101325.0 * np.exp(-zfull / 7500.0)
    temperature = np.clip(288.0 - 6.5e-3 * zfull, 210.0, None)
    q = 0.01 * (pressure / 101325.0) ** 2

    cloud_frac = np.zeros((nlev, nbatch))
    conv_frac = np.zeros((nlev, nbatch))
    mr_lsliq = np.zeros((nlev, nbatch))
    mr_lsice = np.zeros((nlev, nbatch))
    # A liquid cloud between ~1 and 3 km, fraction 0.6.
    cloudy = (zfull > 1000.0) & (zfull < 3000.0)
    cloud_frac[cloudy] = 0.6
    mr_lsliq[cloudy] = 3e-4
    # An ice cloud aloft in half the batch elements (for cold-rain cases).
    icy = (zfull > 6000.0) & (zfull < 8000.0)
    ice_batch = np.zeros(nbatch, dtype=bool)
    ice_batch[: nbatch // 2] = True
    cloud_frac[icy & ice_batch[None, :]] = np.maximum(
        cloud_frac[icy & ice_batch[None, :]], 0.5)
    mr_lsice[icy & ice_batch[None, :]] = 1e-4

    fl_lsrain = np.zeros((nlev, nbatch))
    fl_lsrain[zfull < 3000.0] = 3e-4  # ~1 mm/hr below cloud base
    zeros = np.zeros((nlev, nbatch))
    return CloudsatInputs(
        pressure=pressure, temperature=temperature, specific_humidity=q,
        zfull=zfull, zhalf=zhalf, cloud_frac=cloud_frac, conv_frac=conv_frac,
        mr_lsliq=mr_lsliq, mr_lsice=mr_lsice, mr_ccliq=zeros, mr_ccice=zeros,
        fl_lsrain=fl_lsrain, fl_lssnow=zeros, fl_lsgrpl=zeros,
        fl_ccrain=zeros, fl_ccsnow=zeros,
        land=np.zeros(nbatch), surfelev=np.zeros(nbatch),
        t2m=np.full(nbatch, 288.0))


@unittest.skipUnless(_LUT_PATH.is_file(),
                     "Z-scale LUT not generated (python -m jcosp.radar.lut.generate)")
class SimulatorTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from jcosp.radar.optics import load_lut
        cls.lut = load_lut(_LUT_PATH)

    def test_end_to_end(self):
        state = idealized_state()
        out = simulate_cloudsat(state, self.lut, key=jax.random.PRNGKey(0),
                                ncolumns=50)
        nlev, nbatch = state.pressure.shape
        self.assertEqual(out.dbze.shape, (nlev, 50, nbatch))
        dbze = np.asarray(out.dbze)
        valid = dbze != R_UNDEF
        # The raining cloud must produce echoes in a plausible dBZe range.
        self.assertTrue(valid.any())
        self.assertGreater(dbze[valid].max(), -30.0)
        self.assertLess(dbze[valid].max(), 40.0)
        # CFAD is a normalized histogram over subcolumns.
        cfad = np.asarray(out.cfad_ze)
        ok = cfad != R_UNDEF
        self.assertTrue((cfad[ok] >= 0).all() and (cfad[ok] <= 1).all())
        # Attenuation only reduces reflectivity.
        ze_non = np.asarray(out.ze_non)
        self.assertTrue((dbze[valid] <= ze_non[valid] + 1e-6).all())

    def test_warm_and_cold_rain_split(self):
        # Batch elements with the icy layer aloft are the only candidates for
        # cold rain; purely liquid columns must classify warm.
        state = idealized_state(nbatch=4)
        out = simulate_cloudsat(state, self.lut, key=jax.random.PRNGKey(1),
                                ncolumns=100)
        warm = np.asarray(out.warm_rain.warm_rain
                          + out.warm_rain.warm_drizzle)
        self.assertTrue((warm[2:] > 0).all(),
                        msg=f"liquid-only columns produced no warm rain: {warm}")
        cold = np.asarray(out.warm_rain.cold_rain
                          + out.warm_rain.cold_drizzle)
        self.assertTrue((cold[2:] == 0).all())

    def test_jit_and_grad(self):
        # The pipeline downstream of the subcolumn sampling must be jittable
        # and differentiable w.r.t. cloud water.
        state = idealized_state(nbatch=2)
        from jcosp.subcol import scops
        frac_out = scops(jax.random.PRNGKey(2), state.cloud_frac,
                         state.conv_frac, 20)

        def mean_max_dbze(mr_lsliq):
            out = simulate_cloudsat(state._replace(mr_lsliq=mr_lsliq),
                                    self.lut, frac_out=frac_out)
            dbze = out.dbze
            # Smooth proxy of max dBZe over valid cells.
            valid = dbze > -100.0
            lin = jnp.where(valid, jnp.power(10.0, dbze / 10.0), 0.0)
            return jnp.log(jnp.sum(lin) + 1e-9)

        f = jax.jit(mean_max_dbze)
        val = f(jnp.asarray(state.mr_lsliq))
        self.assertTrue(np.isfinite(float(val)))
        g = jax.grad(mean_max_dbze)(jnp.asarray(state.mr_lsliq))
        g = np.asarray(g)
        self.assertTrue(np.isfinite(g).all(), "gradient contains NaN/inf")
        self.assertGreater(np.abs(g).max(), 0.0, "gradient identically zero")

    def test_cloudsat_modis_joint_products(self):
        # Combined driver: MODIS retrievals + CFODD/warm-rain occurrence.
        nlev, nbatch = 30, 4
        state = idealized_state(nlev=nlev, nbatch=nbatch)
        # Rebuild the liquid cloud lower (0.4-1.8 km) so its top is warmer
        # than freezing — the CFODD SLWC check requires a warm cloud top.
        zfull = state.zfull
        cloudy = (zfull > 400.0) & (zfull < 1800.0)
        cloud_frac = np.where(cloudy, 0.6, 0.0)
        cloud_frac[np.asarray(state.mr_lsice) > 0] = 0.5  # keep ice aloft
        mr_lsliq = np.where(cloudy, 3e-4, 0.0)
        fl_lsrain = np.where(zfull < 1800.0, 3e-4, 0.0)
        state = state._replace(cloud_frac=cloud_frac, mr_lsliq=mr_lsliq,
                               fl_lsrain=fl_lsrain)
        # Cloud effective radii (m): 13-um droplets (CFODD class 2), 30-um ice.
        reff = np.zeros((nlev, nbatch, config.N_HYDRO))
        reff[..., config.I_LSCLIQ] = np.where(state.mr_lsliq > 0, 13e-6, 0.0)
        reff[..., config.I_LSCICE] = np.where(state.mr_lsice > 0, 30e-6, 0.0)
        state = state._replace(reff=reff)
        # Host 0.67-um optical depths and edge pressures.
        dtau_s = np.where(state.mr_lsliq > 0, 8.0,
                          np.where(state.mr_lsice > 0, 2.0, 0.0))
        zhalf_edges = np.linspace(20000.0, 0.0, nlev + 1)
        phalf = np.repeat((101325.0 * np.exp(-zhalf_edges / 7500.0))[:, None],
                          nbatch, axis=1)
        sunlit = np.array([1.0, 0.0, 1.0, 1.0])  # element 1 (icy) is dark

        out = simulate_cloudsat_modis(
            state, self.lut, jnp.asarray(dtau_s),
            jnp.asarray(np.zeros_like(dtau_s)), jnp.asarray(phalf),
            key=jax.random.PRNGKey(4), ncolumns=60, sunlit=sunlit)

        col = out.modis.column
        cf_tot = np.asarray(col.cf_total)
        # Sunlit points see the cloud; the dark point is undefined.
        self.assertTrue((cf_tot[[0, 2, 3]] > 0).all())
        self.assertEqual(cf_tot[1], R_UNDEF)
        # Liquid retrievals: Reff near the 13-um input, LWP consistent.
        size_liq = np.asarray(col.size_liquid)
        self.assertTrue((np.abs(size_liq[[2, 3]] - 13e-6) < 2e-6).all())
        self.assertTrue((np.asarray(col.lwp)[[2, 3]] > 0).all())
        # Ice contamination is retrieved in the icy sunlit element.
        self.assertGreater(np.asarray(col.iwp)[0], 0.0)

        wr = np.asarray(out.warmrain.wr_occfreq_ntotal)
        cfodd = np.asarray(out.warmrain.cfodd_ntotal)
        # Dark point: undefined. Icy sunlit point: screened out (zeros).
        self.assertTrue((wr[:, 1] == R_UNDEF).all())
        self.assertTrue((cfodd[..., 1] == R_UNDEF).all())
        self.assertEqual(wr[:, 0].sum(), 0.0)
        # Warm liquid-only points: qualifying subcolumns land in exactly one
        # regime partition and CFODD class 2 (12 <= Reff < 18 um).
        self.assertTrue((wr[:, [2, 3]].sum(axis=0) > 0).all())
        self.assertTrue((cfodd[:, :, 1, [2, 3]].sum(axis=(0, 1)) > 0).all())
        self.assertEqual(cfodd[:, :, [0, 2], :][..., [2, 3]].sum(), 0.0)

    def test_reproducible_with_fixed_key(self):
        state = idealized_state(nbatch=2)
        out1 = simulate_cloudsat(state, self.lut, key=jax.random.PRNGKey(3),
                                 ncolumns=30)
        out2 = simulate_cloudsat(state, self.lut, key=jax.random.PRNGKey(3),
                                 ncolumns=30)
        np.testing.assert_array_equal(np.asarray(out1.dbze),
                                      np.asarray(out2.dbze))


if __name__ == "__main__":
    unittest.main()
