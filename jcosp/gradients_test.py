"""Every main observable has a defined gradient, and the values are untouched.

COSP's diagnostics are counts of subcolumns passing hard tests, so their
reference derivatives are exactly zero. ``jcosp.surrogate`` gives the tests
smooth derivatives and ``jcosp.subcol.scops.subcolumn_weights`` gives the
SCOPS sampling one. This file checks the result where a calibration meets
it, at the drivers' outputs:

* no output value depends on the smoothing;
* every observable has finite derivatives, in forward and reverse mode,
  with respect to the model state it is simulated from, and the two modes
  are adjoint;
* every observable responds to the cloud fraction and to the in-cloud
  quantities its instrument sees;
* the headline derivative, MODIS cloud fraction with respect to model cloud
  fraction, is the derivative of the expectation over SCOPS draws;
* all of it holds in float32, the precision a host model runs in.

Requires the generated Z-scale LUT, as the other end-to-end tests do.
"""

import importlib.resources
import os
import pathlib
import subprocess
import sys
import textwrap
import unittest

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jcosp import config, inputs, modis, surrogate
from jcosp.constants import R_UNDEF
from jcosp.lidar import CalipsoInputs, lidar_only_cloud, simulate_calipso
from jcosp.radar.quickbeam import cloudsat_precip_occurrence
from jcosp.simulator import CloudsatInputs, simulate_cloudsat_modis
from jcosp.subcol import prec_scops, scops_with_weights

_LUT_PATH = importlib.resources.files("jcosp") / "data" / "quickbeam_lut_94GHz_1M.npz"


def mixed_scene(nlev=30, nbatch=6, seed=0):
    """Build a scene that exercises every simulator.

    Low liquid cloud everywhere, ice cloud aloft and mixed-phase cloud in
    some columns, a convective column, rain and snow, land and ocean points
    on either side of freezing, one night point.
    """
    rng = np.random.default_rng(seed)
    col = np.arange(nbatch)[None, :]
    zhalf_1d = np.linspace(20000.0, 0.0, nlev + 1)
    zhalf = np.repeat(zhalf_1d[1:, None], nbatch, axis=1)
    zfull = zhalf + 0.5 * (zhalf_1d[:-1] - zhalf_1d[1:])[:, None]
    pressure = 101325.0 * np.exp(-zfull / 7500.0)
    phalf = np.repeat((101325.0 * np.exp(-zhalf_1d / 7500.0))[:, None],
                      nbatch, axis=1)
    temperature = (np.clip(288.0 - 6.5e-3 * zfull, 210.0, None)
                   + rng.normal(0.0, 1.0, (1, nbatch)))
    q = 0.01 * (pressure / 101325.0) ** 2

    zeros = np.zeros((nlev, nbatch))
    cf, conv, liq, ice, ccl, cci = (zeros.copy() for _ in range(6))
    low = (zfull > 600.0) & (zfull < 3000.0)
    cf[low] = rng.uniform(0.2, 0.9, low.sum())
    liq[low] = rng.uniform(5e-5, 5e-4, low.sum())
    high = (zfull > 6000.0) & (zfull < 10000.0) & (col % 2 == 0)
    cf[high] = rng.uniform(0.1, 0.7, high.sum())
    ice[high] = rng.uniform(1e-6, 1e-4, high.sum())
    mid = (zfull > 3000.0) & (zfull < 6000.0) & (col % 3 == 0)
    cf[mid] = rng.uniform(0.05, 0.5, mid.sum())
    liq[mid] = rng.uniform(1e-5, 2e-4, mid.sum())
    ice[mid] = rng.uniform(1e-6, 5e-5, mid.sum())
    cv = (zfull > 800.0) & (zfull < 7000.0) & (col == 1)
    conv[cv] = 0.08
    cf[cv] = np.maximum(cf[cv], 0.12)
    ccl[cv] = 2e-4
    cci[cv] = np.where(zfull[cv] > 4000.0, 5e-5, 0.0)

    rain = np.where(zfull < 3000.0, rng.uniform(0.0, 4e-4, (1, nbatch)), 0.0)
    rain[:, 2] *= 1e-3
    snow = np.where((zfull > 3000.0) & (zfull < 9000.0) & (col % 2 == 0),
                    5e-5, 0.0)
    ccrain = np.where(cv | ((zfull < 800.0) & (col == 1)), 3e-4, 0.0)

    reff = np.zeros((nlev, nbatch, config.N_HYDRO))
    reff[..., config.I_LSCLIQ] = np.where(
        liq > 0, rng.uniform(6e-6, 20e-6, liq.shape), 0.0)
    reff[..., config.I_LSCICE] = np.where(
        ice > 0, rng.uniform(15e-6, 60e-6, ice.shape), 0.0)
    reff[..., config.I_CVCLIQ] = np.where(ccl > 0, 10e-6, 0.0)
    reff[..., config.I_CVCICE] = np.where(cci > 0, 40e-6, 0.0)

    state = CloudsatInputs(
        pressure=pressure, temperature=temperature, specific_humidity=q,
        zfull=zfull, zhalf=zhalf, cloud_frac=cf, conv_frac=conv,
        mr_lsliq=liq, mr_lsice=ice, mr_ccliq=ccl, mr_ccice=cci,
        fl_lsrain=rain, fl_lssnow=snow, fl_lsgrpl=zeros,
        fl_ccrain=ccrain, fl_ccsnow=zeros,
        land=(np.arange(nbatch) % 2).astype(float),
        surfelev=np.array([0.0, 300.0, 0.0, 1200.0, 0.0, 50.0])[:nbatch],
        t2m=np.array([288.0, 270.0, 274.0, 280.0, 268.0, 290.0])[:nbatch],
        reff=reff)
    state = CloudsatInputs(*[jnp.asarray(v) for v in state])

    dp = (phalf[1:] - phalf[:-1]) / 9.81

    def tau(water, radius, density):
        return 1.5 * water * dp / (density * np.where(radius > 0, radius, 1.0))

    dtau_s = (tau(liq, reff[..., config.I_LSCLIQ], 1000.0)
              + tau(ice, reff[..., config.I_LSCICE], 917.0))
    dtau_c = tau(ccl, 10e-6, 1000.0)
    extra = dict(
        dtau_s=dtau_s, dtau_c=dtau_c,
        dem_s=1.0 - np.exp(-dtau_s / 2.13), dem_c=1.0 - np.exp(-dtau_c / 2.13),
        phalf=phalf, sunlit=np.array([1.0, 1.0, 1.0, 0.0, 1.0, 1.0])[:nbatch])
    return state, {k: jnp.asarray(v) for k, v in extra.items()}


# The model fields a calibration reaches the simulators through.
PARAMETERS = ("cloud_frac", "mr_lsliq", "mr_lsice", "fl_lsrain",
              "temperature", "dtau_s", "dem_s")

# For each observable, an in-cloud quantity its instrument responds to in
# ``mixed_scene`` (the cloud fraction is required of all of them). The
# three total-cloud-cover products of the joint radar/lidar diagnostic are
# occurrence fractions of echoes that are far from the -30 dBZe limit in
# this scene, so only the sampling moves them; their thresholds are
# exercised in ``ThresholdResponseTest``.
RESPONDS_TO = {
    "cloudsat.": "fl_lsrain",
    "modis.": "dtau_s",
    "imager.": "dtau_s",
    "cfodd.": "fl_lsrain",
    "isccp.": "dtau_s",
    "isccp.meantb": "dem_s",
    "calipso.": "mr_lsice",
    "calipso.radar_lidar_tcc": None,
    "calipso.cloudsat_tcc": None,
    "calipso.cloudsat_tcc2": None,
}

# Observables that are means of continuous retrievals, which have a
# reference derivative; every other one is a count and has none.
CONTINUOUS = {
    "cloudsat.pia", "imager.lwp", "imager.iwp", "isccp.meanalbedocld",
    "isccp.meanptop", "isccp.meantaucld", "isccp.meantb", "modis.ctp_mean",
    "modis.lwp", "modis.iwp", "modis.size_liquid", "modis.size_ice",
    "modis.tau_total", "modis.tau_liquid", "modis.tau_ice",
    "modis.tau_total_log", "modis.tau_liquid_log", "modis.tau_ice_log",
}


def _expected_response(name):
    best = max((p for p in RESPONDS_TO if name.startswith(p)), key=len)
    return RESPONDS_TO[best]


def observables(lut, state, extra, params, key, ncolumns=40,
                smoothing=surrogate.DEFAULT_SMOOTHING):
    """Every column-stage diagnostic of the package, as a flat dict."""
    s = state._replace(**{k: params[k] for k in PARAMETERS
                          if k in CloudsatInputs._fields})
    joint = simulate_cloudsat_modis(
        s, lut, params["dtau_s"], extra["dtau_c"], extra["phalf"], key=key,
        ncolumns=ncolumns, sunlit=extra["sunlit"], run_isccp=True,
        dem_s=params["dem_s"], dem_c=extra["dem_c"], skt=state.t2m,
        emsfc_lw=0.98, smoothing=smoothing)
    cs = joint.cloudsat
    weights = cs.subcolumn_weight

    # CALIPSO on the same SCOPS realization, as a host wires it.
    hyd = inputs.distribute_hydrometeors(
        cs.frac_out, cs.prec_frac, s.pressure, s.temperature,
        s.mr_lsliq, s.mr_lsice, s.mr_ccliq, s.mr_ccice,
        s.fl_lsrain, s.fl_lssnow, s.fl_lsgrpl, s.fl_ccrain, s.fl_ccsnow,
        reff_in=s.reff, weights=weights)
    lidar = simulate_calipso(
        CalipsoInputs(
            pressure=s.pressure, pressure_half=extra["phalf"],
            temperature=s.temperature, zfull=s.zfull, zhalf=s.zhalf,
            q_lsliq=hyd.mr_hydro[..., config.I_LSCLIQ],
            q_lsice=hyd.mr_hydro[..., config.I_LSCICE],
            q_cvliq=hyd.mr_hydro[..., config.I_CVCLIQ],
            q_cvice=hyd.mr_hydro[..., config.I_CVCICE],
            reff_lsliq=s.reff[..., config.I_LSCLIQ],
            reff_lsice=s.reff[..., config.I_LSCICE],
            reff_cvliq=s.reff[..., config.I_CVCLIQ],
            reff_cvice=s.reff[..., config.I_CVCICE]),
        dbze=cs.dbze, weights=weights, smoothing=smoothing)
    lwp, iwp = modis.imager_water_paths(
        joint.modis.subcolumn, weights=weights, smoothing=smoothing)

    out = {"cloudsat.cfad_ze": cs.cfad_ze,
           "cloudsat.precip_cover": cs.precip_cover, "cloudsat.pia": cs.pia}
    out.update({f"cloudsat.{f}": getattr(cs.warm_rain, f)
                for f in cs.warm_rain._fields})
    out.update({f"modis.{f}": getattr(joint.modis.column, f)
                for f in joint.modis.column._fields})
    out.update({"imager.lwp": lwp, "imager.iwp": iwp,
                "cfodd.cfodd_ntotal": joint.warmrain.cfodd_ntotal,
                "cfodd.wr_occfreq_ntotal": joint.warmrain.wr_occfreq_ntotal})
    out.update({f"isccp.{f}": getattr(joint.isccp.column, f)
                for f in joint.isccp.column._fields})
    out.update({f"calipso.{f}": getattr(lidar, f)
                for f in ("cfad_sr", "lidarcld", "cldlayer", "lidarcldphase",
                          "cldlayerphase", "lidarcldtmp")})
    out.update({f"calipso.{f}": getattr(lidar.lidar_only, f)
                for f in lidar.lidar_only._fields})
    return out


@pytest.mark.slow
@unittest.skipUnless(_LUT_PATH.is_file(),
                     "Z-scale LUT not generated (python -m jcosp.radar.lut.generate)")
class ObservableGradientTest(unittest.TestCase):
    """One forward run and its derivatives, shared by the checks below.

    The checks are sub-tests of a single test so that the shared run is
    made once however the test session is distributed over workers.
    """

    @classmethod
    def setUpClass(cls):
        from jcosp.radar.optics import load_lut
        lut = load_lut(_LUT_PATH)
        state, extra = mixed_scene()
        key = jax.random.PRNGKey(0)
        cls.params = {
            **{k: getattr(state, k) for k in PARAMETERS
               if k in CloudsatInputs._fields},
            "dtau_s": extra["dtau_s"], "dem_s": extra["dem_s"]}

        def run(params, smoothing=surrogate.DEFAULT_SMOOTHING):
            return observables(lut, state, extra, params, key,
                               smoothing=smoothing)

        cls.simulate = staticmethod(run)
        cls.values = run(cls.params)

        # Forward mode: one direction per parameter, a relative change of
        # each field wherever it is nonzero.
        rng = np.random.default_rng(0)
        cls.directions = {
            k: v * jnp.asarray(rng.uniform(0.5, 1.5, v.shape))
            for k, v in cls.params.items()}
        cls.tangents = {}
        for name in PARAMETERS:
            direction = {k: jnp.zeros_like(v) for k, v in cls.params.items()}
            direction[name] = cls.directions[name]
            cls.tangents[name] = jax.jvp(run, (cls.params,), (direction,))[1]

        # Reverse mode: one cotangent over all the observables at once.
        cls.cotangent = {k: jnp.asarray(rng.normal(size=v.shape), v.dtype)
                         for k, v in cls.values.items()}
        cls.gradient = jax.vjp(run, cls.params)[1](cls.cotangent)[0]

    def test_observable_gradients(self):
        checks = [getattr(self, name) for name in sorted(dir(self))
                  if name.startswith("check_")]
        self.assertEqual(len(checks), 8)
        for check in checks:
            with self.subTest(check=check.__name__):
                check()

    def check_values_do_not_depend_on_the_smoothing(self):
        reference = self.simulate(self.params, surrogate.NO_SMOOTHING)
        self.assertEqual(set(reference), set(self.values))
        for name, value in self.values.items():
            np.testing.assert_array_equal(
                np.asarray(value), np.asarray(reference[name]), err_msg=name)
            self.assertEqual(value.dtype, reference[name].dtype, msg=name)

    def check_scene_exercises_every_observable(self):
        # A diagnostic that is undefined or empty everywhere would pass the
        # derivative checks without testing anything.
        for name, value in self.values.items():
            value = np.asarray(value)
            defined = value[value != R_UNDEF]
            self.assertGreater(defined.size, 0, msg=name)
            self.assertGreater(np.abs(defined).max(), 0.0, msg=name)

    def check_forward_mode_is_finite(self):
        for param, tangent in self.tangents.items():
            for name, t in tangent.items():
                self.assertTrue(np.isfinite(np.asarray(t)).all(),
                                msg=f"d {name} / d {param} is not finite")

    def check_every_observable_responds_to_cloud_fraction(self):
        for name, t in self.tangents["cloud_frac"].items():
            self.assertGreater(float(jnp.abs(t).max()), 0.0,
                               msg=f"{name} has no derivative in cloud_frac")

    def check_every_observable_responds_to_what_its_instrument_sees(self):
        for name in self.values:
            param = _expected_response(name)
            if param is None:
                continue
            self.assertGreater(
                float(jnp.abs(self.tangents[param][name]).max()), 0.0,
                msg=f"{name} has no derivative in {param}")

    def check_reverse_mode_is_finite_and_reaches_every_parameter(self):
        for param, g in self.gradient.items():
            g = np.asarray(g)
            self.assertTrue(np.isfinite(g).all(),
                            msg=f"gradient in {param} is not finite")
            self.assertGreater(np.abs(g).max(), 0.0,
                               msg=f"gradient in {param} is identically zero")

    def check_forward_and_reverse_modes_are_adjoint(self):
        # <J t, c> == <t, J^T c>, parameter by parameter.
        for param in PARAMETERS:
            forward = sum(float(jnp.vdot(self.tangents[param][k],
                                         self.cotangent[k]))
                          for k in self.values)
            reverse = float(jnp.vdot(self.directions[param],
                                     self.gradient[param]))
            self.assertAlmostEqual(
                forward / reverse, 1.0, places=8,
                msg=f"{param}: forward {forward}, reverse {reverse}")

    def check_reference_derivative_of_the_counts_is_zero(self):
        # What the surrogates are for: without them a calibration target
        # that is a count gets no gradient at all.
        tangent = jax.jvp(
            lambda p: self.simulate(p, surrogate.NO_SMOOTHING),
            (self.params,), (self.directions,))[1]
        for name, t in tangent.items():
            if name in CONTINUOUS:
                continue
            np.testing.assert_array_equal(np.asarray(t), 0.0, err_msg=name)
        counts = set(self.values) - CONTINUOUS
        self.assertGreater(len(counts), 25)


_FLOAT32_CHECK = """
    import importlib.resources
    import jax
    import jax.numpy as jnp
    import numpy as np
    import jcosp.gradients_test as g
    from jcosp.radar.optics import load_lut
    from jcosp.simulator import CloudsatInputs

    assert not jax.config.jax_enable_x64
    lut = load_lut(importlib.resources.files("jcosp") / "data"
                   / "quickbeam_lut_94GHz_1M.npz")
    state, extra = g.mixed_scene()
    params = {**{k: getattr(state, k) for k in g.PARAMETERS
                 if k in CloudsatInputs._fields},
              "dtau_s": extra["dtau_s"], "dem_s": extra["dem_s"]}
    run = lambda p: g.observables(lut, state, extra, p, jax.random.PRNGKey(0))

    values, tangents = jax.jvp(run, (params,), (params,))
    for name, v in values.items():
        assert v.dtype == jnp.float32, (name, v.dtype)
        assert np.isfinite(np.asarray(tangents[name])).all(), name
    rng = np.random.default_rng(0)
    cotangent = {k: jnp.asarray(rng.normal(size=v.shape), v.dtype)
                 for k, v in values.items()}
    gradient = jax.vjp(run, params)[1](cotangent)[0]
    for name, v in gradient.items():
        v = np.asarray(v)
        assert np.isfinite(v).all(), "gradient in " + name + " is not finite"
        assert np.abs(v).max() > 0.0, "gradient in " + name + " is zero"
    print("float32 gradients finite")
"""


@pytest.mark.slow
@unittest.skipUnless(_LUT_PATH.is_file(),
                     "Z-scale LUT not generated (python -m jcosp.radar.lut.generate)")
class Float32GradientTest(unittest.TestCase):
    """The same scene in float32, the precision JAX-GCM runs in.

    The suite runs in float64 (conftest), where a guard such as
    ``max(x, 1e-30)`` keeps a masked-out branch finite. In float32 the same
    guard can overflow one operation later, and the branch then returns NaN
    to reverse mode (0 * inf) for every output, selected or not. That only
    shows in a process without x64, hence the subprocess.
    """

    def test_forward_and_reverse_modes_are_finite(self):
        root = pathlib.Path(__file__).resolve().parent.parent
        env = {**os.environ, "JAX_ENABLE_X64": "0", "JAX_PLATFORMS": "cpu"}
        result = subprocess.run(
            [sys.executable, "-c", textwrap.dedent(_FLOAT32_CHECK)],
            cwd=root, env=env, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, msg=result.stderr[-3000:])
        self.assertIn("float32 gradients finite", result.stdout)


def _modis_scene(nlev=12):
    """One gridpoint: a thin high deck over a thicker low one."""
    z = np.linspace(0.0, 1.0, nlev)
    cloud_frac = (0.45 * np.exp(-((z - 0.25) / 0.1) ** 2)
                  + 0.6 * np.exp(-((z - 0.75) / 0.12) ** 2))
    # Levels with less cloud than the sampling derivative's probability
    # floor are deliberately under-weighted by it; keep them out of a test
    # of its accuracy.
    cloud_frac = np.where(cloud_frac > 0.02, cloud_frac, 0.0)
    phalf = np.linspace(20000.0, 101325.0, nlev + 1)
    pressure = 0.5 * (phalf[1:] + phalf[:-1])
    temperature = 220.0 + 70.0 * z
    liquid = z > 0.5
    # In-cloud layer optical depths small enough that a subcolumn cloudy at
    # one or two thin levels stays under the detection limit.
    dtau = np.where(liquid, 0.9, 0.12)
    reff = np.zeros((nlev, config.N_HYDRO))
    reff[:, config.I_LSCLIQ] = np.where(liquid, 10e-6, 0.0)
    reff[:, config.I_LSCICE] = np.where(liquid, 0.0, 30e-6)
    return dict(
        cloud_frac=jnp.asarray(cloud_frac), phalf=jnp.asarray(phalf),
        pressure=jnp.asarray(pressure), temperature=jnp.asarray(temperature),
        mr_liq=jnp.asarray(np.where(liquid, 2e-4, 0.0)),
        mr_ice=jnp.asarray(np.where(liquid, 0.0, 2e-5)),
        dtau=jnp.asarray(dtau), reff=jnp.asarray(reff))


def _modis_column(scene, cloud_frac, dtau, key, ncolumns,
                  smoothing=surrogate.DEFAULT_SMOOTHING):
    """SCOPS -> hydrometeors -> MODIS for one gridpoint."""
    zeros = jnp.zeros_like(cloud_frac)
    frac_out, weights = scops_with_weights(
        key, cloud_frac, zeros, ncolumns, floor=smoothing.cloud_fraction)
    prec_frac = prec_scops(frac_out, zeros, zeros)
    hyd = inputs.distribute_hydrometeors(
        frac_out, prec_frac, scene["pressure"], scene["temperature"],
        scene["mr_liq"], scene["mr_ice"], zeros, zeros,
        zeros, zeros, zeros, zeros, zeros, reff_in=scene["reff"],
        weights=weights)
    return modis.simulate_modis(
        frac_out, hyd.mr_hydro, hyd.reff, dtau, zeros, scene["phalf"],
        weights=weights, smoothing=smoothing).column


class ModisCloudFractionTest(unittest.TestCase):
    """The use case: MODIS cloud fraction against model cloud fraction."""

    def test_derivative_is_that_of_the_expectation(self):
        # With enough subcolumns the sampled mean is smooth enough to
        # difference; the derivative the weights give has to match it.
        scene = _modis_scene()
        ncolumns, eps = 50_000, 0.04

        @jax.jit
        def cover(scale, key):
            col = _modis_column(scene, scale * scene["cloud_frac"],
                                scene["dtau"], key, ncolumns)
            return jnp.stack([col.cf_total, col.cf_low])

        fd, ad = [], []
        for key in jax.random.split(jax.random.PRNGKey(3), 2):
            value = np.asarray(cover(1.0, key))
            # Partly cloudy, and not all of the cloud is detected.
            self.assertTrue((value > 20.0).all() and (value < 80.0).all())
            fd.append((cover(1.0 + eps, key) - cover(1.0 - eps, key))
                      / (2 * eps))
            ad.append(jax.jacfwd(lambda s: cover(s, key))(1.0))
        fd, ad = np.mean(fd, axis=0), np.mean(ad, axis=0)
        self.assertTrue((fd > 20.0).all())
        np.testing.assert_allclose(ad, fd, rtol=0.04)

    def test_detection_limit_responds_to_optical_depth(self):
        # Thicker cloud, same cloud fraction: more subcolumns detected.
        scene = _modis_scene()
        key = jax.random.PRNGKey(4)

        def cover(scale, smoothing=surrogate.DEFAULT_SMOOTHING):
            return _modis_column(scene, scene["cloud_frac"],
                                 scale * scene["dtau"], key, 400,
                                 smoothing).cf_total

        self.assertGreater(float(jax.grad(cover)(1.0)), 0.0)
        self.assertEqual(
            float(jax.grad(lambda s: cover(s, surrogate.NO_SMOOTHING))(1.0)),
            0.0)

    def test_jit_grad_and_vmap(self):
        scene = _modis_scene()
        keys = jax.random.split(jax.random.PRNGKey(5), 3)

        @jax.jit
        def objective(cloud_frac, dtau):
            def one(key):
                col = _modis_column(scene, cloud_frac, dtau, key, 30)
                return col.cf_total + 1e3 * col.lwp
            return jnp.mean(jax.vmap(one)(keys))

        g_cf, g_tau = jax.grad(objective, argnums=(0, 1))(
            scene["cloud_frac"], scene["dtau"])
        for g in (g_cf, g_tau):
            g = np.asarray(g)
            self.assertTrue(np.isfinite(g).all())
            self.assertGreater(np.abs(g).max(), 0.0)


class ThresholdResponseTest(unittest.TestCase):
    """Thresholds the mixed scene does not sit near."""

    def test_lidar_only_cloud(self):
        rng = np.random.default_rng(0)
        nlev, ncol = 8, 50
        beta_mol = jnp.asarray(np.full((nlev,), 1e-6))
        # Scattering ratios either side of 5; echoes mostly under the radar's
        # -30 dBZe limit, so that in many subcolumns one level decides.
        beta_tot = jnp.asarray(1e-6 * rng.uniform(0.5, 9.0, (nlev, ncol)))
        dbze = jnp.asarray(rng.uniform(-44.0, -29.0, (nlev, ncol)))

        def products(beta_tot, dbze, smoothing=surrogate.DEFAULT_SMOOTHING):
            out = lidar_only_cloud(beta_tot, beta_mol, dbze,
                                   smoothing=smoothing)
            return jnp.concatenate([jnp.ravel(v) for v in out])

        np.testing.assert_array_equal(
            np.asarray(products(beta_tot, dbze)),
            np.asarray(products(beta_tot, dbze, surrogate.NO_SMOOTHING)))
        jac_b, jac_z = jax.jacobian(products, argnums=(0, 1))(beta_tot, dbze)
        jac_b, jac_z = np.asarray(jac_b), np.asarray(jac_z)
        self.assertTrue(np.isfinite(jac_b).all() and np.isfinite(jac_z).all())
        # Lidar-only frequency at each level sees both instruments; the
        # three cover products (last rows) see the radar.
        self.assertGreater(np.abs(jac_b[:nlev]).max(), 0.0)
        for row in range(nlev + 3):
            self.assertGreater(np.abs(jac_z[row]).max(), 0.0, msg=row)

    def test_precipitation_classes(self):
        rng = np.random.default_rng(1)
        nglev, ncol, nbatch = config.N_VGRID, 60, 4
        ze_non = jnp.asarray(rng.uniform(-20.0, 12.0, (nglev, ncol, nbatch)))
        ze = ze_non - jnp.asarray(rng.uniform(0.0, 45.0, (nglev, ncol, nbatch)))
        fpi = jnp.asarray(rng.uniform(0.0, 1.0, (ncol, nbatch)))
        land = jnp.asarray([0.0, 1.0, 1.0, 1.0])
        surfelev = jnp.zeros(nbatch)
        t2m = jnp.asarray([280.0, 272.5, 274.0, 275.5])

        def cover(ze, ze_non, t2m, fpi, smoothing=surrogate.DEFAULT_SMOOTHING):
            return cloudsat_precip_occurrence(
                ze, ze_non, land, surfelev, t2m, fpi, smoothing=smoothing)

        value, pia = cover(ze, ze_non, t2m, fpi)
        ref, pia_ref = cover(ze, ze_non, t2m, fpi, surrogate.NO_SMOOTHING)
        np.testing.assert_array_equal(np.asarray(value), np.asarray(ref))
        np.testing.assert_array_equal(np.asarray(pia), np.asarray(pia_ref))
        np.testing.assert_allclose(np.asarray(value).sum(axis=0), 1.0,
                                   rtol=1e-6)
        pick = jnp.asarray(rng.normal(size=value.shape))
        grads = jax.grad(
            lambda *a: jnp.sum(pick * cover(*a)[0]), argnums=(0, 1, 2, 3))(
                ze, ze_non, t2m, fpi)
        for name, g in zip(("ze", "ze_non", "t2m", "fpi"), grads):
            g = np.asarray(g)
            self.assertTrue(np.isfinite(g).all(), msg=name)
            self.assertGreater(np.abs(g).max(), 0.0, msg=name)
        # The 2-m temperature matters over land only, the frozen fraction
        # over ocean only.
        self.assertEqual(float(grads[2][0]), 0.0)
        self.assertTrue((np.abs(np.asarray(grads[2][1:])) > 0.0).all())
        np.testing.assert_array_equal(np.asarray(grads[3][:, 1:]), 0.0)


if __name__ == "__main__":
    unittest.main()
