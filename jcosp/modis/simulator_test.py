"""Tests for the MODIS simulator against Fortran loop transliterations.

The references transliterate ``modis_simulator.F90`` loop-for-loop:
two_stream (beam = 2), adding_doubling, compute_toa_reflectace,
cloud_top_pressure (including the call-site truncation that drops the bottom
layer), weight_by_extinction, interpolate_to_min, modis_subcolumn and
modis_column — plus physical sanity checks and gradient tests.
"""

import unittest

import jax
import jax.numpy as jnp
import numpy as np

from jcosp import config
from jcosp.constants import R_UNDEF
from jcosp.modis import optics
from jcosp.modis import simulator as msim
from jcosp.modis.optics_test import get_g_nir_reference, get_ssa_nir_reference
from jcosp.stats_test import hist2d_reference

# ---------------------------------------------------------------------------
# Fortran transliterations (single subcolumn / single point)
# ---------------------------------------------------------------------------
_XMU = 0.866
_MIN_CONS_W0 = 0.9999999


def two_stream_reference(tauint, gint, w0int):
    """Transliteration of two_stream / two_stream_reflectance (beam = 2)."""
    f = gint**2
    tau = (1 - w0int * f) * tauint
    w0 = (1 - f) * w0int / (1 - w0int * f)
    g = (gint - f) / (1 - f)
    gamma1 = (7 - w0 * (4 + 3 * g)) / 4
    gamma2 = -(1 - w0 * (4 - 3 * g)) / 4
    if w0int > _MIN_CONS_W0:
        ref = gamma1 * tau / (1 + gamma1 * tau)
        return ref, 1 - ref
    rk = np.sqrt(gamma1**2 - gamma2**2)
    r4 = (1 - (rk * _XMU) ** 2) * (rk + gamma1)
    r5 = (1 - (rk * _XMU) ** 2) * (rk - gamma1)
    beta = -r5 / r4
    e1 = min(rk * tau, 500.0)
    ef1 = np.exp(-e1)
    ef2 = np.exp(-2 * e1)
    ref = (gamma2 * (1 - ef2)) / ((rk + gamma1) * (1 - beta * ef2))
    tra = (2 * rk * ef1) / ((rk + gamma1) * (1 - beta * ef2))
    return ref, tra


def adding_doubling_reference(refl, tran):
    """Transliteration of adding_doubling."""
    r_cum, t_cum = refl[0], tran[0]
    for r, t in zip(refl[1:], tran[1:]):
        den = 1 - r_cum * r
        r_cum, t_cum = r_cum + r * t_cum**2 / den, t_cum * t / den
    return r_cum, t_cum


def compute_toa_reflectance_reference(tau, g, w0):
    """Transliteration of compute_toa_reflectace (packs cloudy layers)."""
    idx = np.flatnonzero(tau > 0)
    if len(idx) == 0:
        return 0.0  # adding over zero layers; the port's identity init
    pairs = [two_stream_reference(tau[i], g[i], w0[i]) for i in idx]
    return adding_doubling_reference([p[0] for p in pairs],
                                     [p[1] for p in pairs])[0]


def cloud_top_pressure_reference(tau, phalf, tau_limit, legacy=False):
    """Transliteration of cloud_top_pressure.

    With ``legacy`` the Fortran call-site truncation is reproduced: the
    (nlev+1)-long [0, tau] array is cut to nlev entries by the dummy
    argument, dropping the bottom layer from the integration (the FIXED
    UPSTREAM BUG; the default integrates every layer).
    """
    nlev = len(tau)
    n_used = nlev - 1 if legacy else nlev
    tot, prod = 0.0, 0.0
    for layer in range(n_used):  # layer between phalf[layer], phalf[layer+1]
        inc = tau[layer]
        p_lo, p_hi = phalf[layer], phalf[layer + 1]
        if tot + inc > tau_limit:
            dx = tau_limit - tot
            tot += dx
            prod += p_lo * dx + (p_hi - p_lo) * dx**2 / (2 * inc)
        else:
            tot += inc
            prod += inc * (p_hi + p_lo) / 2
        if tot >= tau_limit:
            break
    return prod / tot if tot > 0 else 0.0


def weight_by_extinction_reference(tau, f, tau_limit):
    """Transliteration of weight_by_extinction."""
    tot, prod = 0.0, 0.0
    for i in range(len(tau)):
        if tot + tau[i] > tau_limit:
            dx = tau_limit - tot
            tot += dx
            prod += dx * f[i]
        else:
            tot += tau[i]
            prod += tau[i] * f[i]
        if tot >= tau_limit:
            break
    return prod / tot if tot > 0 else 0.0


def interpolate_to_min_reference(x, y, yobs):
    """Transliteration of interpolate_to_min."""
    n = len(y)
    diff = y - yobs
    minloc = int(np.argmin(np.abs(diff)))  # first minimum, as minloc
    if minloc == 0:
        lower, upper = minloc, minloc + 1
    elif minloc == n - 1:
        lower, upper = minloc - 1, minloc
    else:
        if diff[minloc - 1] * diff[minloc] < 0:
            lower, upper = minloc - 1, minloc
        else:
            lower, upper = minloc, minloc + 1
    if diff[lower] * diff[upper] < 0:
        return x[upper] - diff[upper] * (x[upper] - x[lower]) / (
            diff[upper] - diff[lower])
    return config.MODIS_RE_FILL


def trial_tables_reference():
    """Transliteration of the COSP_MODIS_INIT trial-radius tables."""
    i = np.arange(config.MODIS_NUM_TRIAL_RES, dtype=float)
    n1 = config.MODIS_NUM_TRIAL_RES - 1
    trial_w = 4.0 + (30.0 - 4.0) / n1 * i
    trial_i = 5.0 + (90.0 - 5.0) / n1 * i
    g_w = np.array([get_g_nir_reference(1, r) for r in trial_w])
    w0_w = np.array([get_ssa_nir_reference(1, r) for r in trial_w])
    g_i = np.array([get_g_nir_reference(2, r) for r in trial_i])
    w0_i = np.array([get_ssa_nir_reference(2, r) for r in trial_i])
    return trial_w, trial_i, g_w, w0_w, g_i, w0_i


def modis_subcolumn_reference(phalf, tau, liqfrac, g, w0, isccp_ctp=None):
    """Transliteration of modis_subcolumn.

    Arrays are (ncol, nlev), TOA-first exactly as the Fortran (k = 1 top).
    """
    trial_w, trial_i, g_w, w0_w, g_i, w0_i = trial_tables_reference()
    ncol = tau.shape[0]
    phase = np.zeros(ncol, int)
    ctp = np.zeros(ncol)
    size = np.zeros(ncol)
    rtau = tau.sum(axis=1)
    cloud = rtau >= config.MODIS_MIN_OPTICAL_THICKNESS
    for i in range(ncol):
        if cloud[i]:
            ctp[i] = cloud_top_pressure_reference(tau[i], phalf, 1.0)
            ilf = weight_by_extinction_reference(tau[i], liqfrac[i], 1.0)
            if ilf >= 0.7:
                phase[i] = 1
            elif ilf <= 0.3:
                phase[i] = 2
            else:
                phase[i] = 3
            obs = compute_toa_reflectance_reference(tau[i], g[i], w0[i])
            if phase[i] in (1, 3):
                pred = np.array([two_stream_reference(rtau[i], g_w[k],
                                                      w0_w[k])[0]
                                 for k in range(len(trial_w))])
                size[i] = 1e-6 * interpolate_to_min_reference(trial_w, pred,
                                                              obs)
            else:
                pred = np.array([two_stream_reference(rtau[i], g_i[k],
                                                      w0_i[k])[0]
                                 for k in range(len(trial_i))])
                size[i] = 1e-6 * interpolate_to_min_reference(trial_i, pred,
                                                              obs)
        else:
            ctp[i] = R_UNDEF
            phase[i] = 0
            size[i] = R_UNDEF
            rtau[i] = R_UNDEF
    sel = (size < 0) & (size != R_UNDEF)
    size[sel] = 1e-6 * config.MODIS_RE_FILL
    if isccp_ctp is not None:
        sel = cloud & (ctp > config.MODIS_CO2SLICING_PRESSURE_LIMIT)
        ctp[sel] = isccp_ctp[sel]
    return phase, ctp, rtau, size


def modis_column_reference(phase, ctp, tau, size):
    """Transliteration of modis_column (single point; arrays (ncol,))."""
    ncol = len(phase)
    valid = size > 0
    cmask = (phase != 0) & valid
    wmask = (phase == 1) & valid
    imask = (phase == 2) & valid
    n_tot = float(cmask.sum())
    n_wat = float(wmask.sum())
    n_ice = float(imask.sum())
    n_hi = float((cmask & (ctp <= 44000.0)).sum())
    n_lo = float((cmask & (ctp > 68000.0)).sum())
    n_mid = n_tot - n_hi - n_lo
    lwp_c = config.MODIS_LWP_CONVERSION

    def mean(vals, mask, n):
        return vals[mask].sum() / n if n > 0 else R_UNDEF

    out = {
        "tau_total": mean(tau, cmask, n_tot),
        "tau_total_log": mean(np.log10(np.abs(tau)), cmask, n_tot),
        "tau_liquid": mean(tau, wmask, n_wat),
        "tau_liquid_log": mean(np.log10(np.abs(tau)), wmask, n_wat),
        "size_liquid": mean(size, wmask, n_wat),
        "lwp": mean(lwp_c * size * tau, wmask, n_wat),
        "tau_ice": mean(tau, imask, n_ice),
        "tau_ice_log": mean(np.log10(np.abs(tau)), imask, n_ice),
        "size_ice": mean(size, imask, n_ice),
        "iwp": mean(lwp_c * 0.93 * size * tau, imask, n_ice),
        # FIXED UPSTREAM BUG: mean CTP is undefined (not 0) when cloud-free.
        "ctp_mean": mean(ctp, cmask, n_tot),
        "cf_total": 100.0 * n_tot / ncol, "cf_liquid": 100.0 * n_wat / ncol,
        "cf_ice": 100.0 * n_ice / ncol, "cf_high": 100.0 * n_hi / ncol,
        "cf_mid": 100.0 * n_mid / ncol, "cf_low": 100.0 * n_lo / ncol,
    }
    tau_w = np.where(cmask, tau, -999.0)
    ctp_w = np.where(cmask, ctp, -999.0)
    tau_l = np.where(cmask, np.where(wmask, tau, R_UNDEF), -999.0)
    tau_i = np.where(cmask, np.where(imask, tau, R_UNDEF), -999.0)
    lwp_w = np.where(cmask, np.where(wmask, lwp_c * size * tau, R_UNDEF),
                     -999.0)
    iwp_w = np.where(cmask, np.where(imask, lwp_c * 0.93 * size * tau,
                                     R_UNDEF), -999.0)
    re_l = np.where(wmask, size, R_UNDEF)
    re_i = np.where(imask, size, R_UNDEF)
    s = 100.0 / ncol
    out["tau_vs_ctp"] = s * hist2d_reference(
        tau_w, ctp_w, config.MODIS_HIST_TAU, config.MODIS_HIST_PRES)
    out["tau_vs_ctp_liq"] = s * hist2d_reference(
        tau_l, ctp_w, config.MODIS_HIST_TAU, config.MODIS_HIST_PRES)
    out["tau_vs_ctp_ice"] = s * hist2d_reference(
        tau_i, ctp_w, config.MODIS_HIST_TAU, config.MODIS_HIST_PRES)
    out["lwp_vs_reff_liq"] = s * hist2d_reference(
        lwp_w, re_l, config.MODIS_HIST_LWP, config.MODIS_HIST_REFF_LIQ)
    out["iwp_vs_reff_ice"] = s * hist2d_reference(
        iwp_w, re_i, config.MODIS_HIST_IWP, config.MODIS_HIST_REFF_ICE)
    out["tau_vs_reff_ice"] = s * hist2d_reference(
        tau_w, re_i, config.MODIS_HIST_TAU, config.MODIS_HIST_REFF_ICE)
    out["tau_vs_reff_liq"] = s * hist2d_reference(
        tau_w, re_l, config.MODIS_HIST_TAU, config.MODIS_HIST_REFF_LIQ)
    return out


def _random_scene(rng, nlev=7, ncol=12):
    """Build a mixed-phase subcolumn scene with physical g/w0 from the fits."""
    cloudy = rng.uniform(size=(nlev, ncol)) < 0.5
    tau = np.where(cloudy, rng.uniform(0.02, 6.0, (nlev, ncol)), 0.0)
    liq = np.where(cloudy, rng.uniform(size=(nlev, ncol)), 0.0)
    size_l = rng.uniform(4.0, 30.0, (nlev, ncol))   # microns
    size_i = rng.uniform(5.0, 90.0, (nlev, ncol))
    _, g, w0 = optics.modis_optics(tau * liq, size_l, tau * (1 - liq), size_i)
    g, w0 = np.array(g), np.array(w0)
    # Exercise the conservative-scattering branch in a few cells.
    cons = cloudy & (rng.uniform(size=(nlev, ncol)) < 0.1)
    w0[cons] = 1.0
    phalf = np.linspace(3000.0, 101300.0, nlev + 1)
    return tau, liq, g, w0, phalf


class TwoStreamTest(unittest.TestCase):
    def test_matches_transliteration(self):
        rng = np.random.default_rng(10)
        for _ in range(200):
            tau = rng.uniform(0.01, 80.0)
            g = rng.uniform(0.4, 0.95)
            w0 = rng.choice([rng.uniform(0.3, 0.99999), 1.0])
            ref, tra = msim.two_stream(tau, g, w0)
            ref_r, tra_r = two_stream_reference(tau, g, w0)
            np.testing.assert_allclose(float(ref), ref_r, rtol=1e-12)
            np.testing.assert_allclose(float(tra), tra_r, rtol=1e-12)

    def test_energy_conservation_bounds(self):
        # Note: the delta-Eddington two-stream genuinely returns *negative*
        # reflectances for strongly absorbing inputs (gamma2 < 0) — the
        # Fortran does too (verified above), and the size retrieval compares
        # observed and predicted values in the same frame — so only the
        # upper bounds and energy conservation are asserted here.
        rng = np.random.default_rng(11)
        tau = rng.uniform(0.01, 50.0, 100)
        g = rng.uniform(0.4, 0.95, 100)
        w0 = rng.uniform(0.5, 1.0, 100)
        ref, tra = msim.two_stream(jnp.asarray(tau), jnp.asarray(g),
                                   jnp.asarray(w0))
        ref, tra = np.asarray(ref), np.asarray(tra)
        self.assertTrue(np.isfinite(ref).all() and np.isfinite(tra).all())
        self.assertTrue((ref < 1).all())
        self.assertTrue(((tra >= 0) & (tra <= 1)).all())
        self.assertTrue((ref + tra <= 1 + 1e-12).all())


class ToaReflectanceTest(unittest.TestCase):
    def test_matches_transliteration(self):
        rng = np.random.default_rng(12)
        tau, liq, g, w0, _ = _random_scene(rng)
        out = np.asarray(msim.compute_toa_reflectance(tau, g, w0))
        for c in range(tau.shape[1]):
            np.testing.assert_allclose(
                out[c], compute_toa_reflectance_reference(
                    tau[:, c], g[:, c], w0[:, c]), rtol=1e-11,
                err_msg=f"column {c}")


class ProfileScanTest(unittest.TestCase):
    def test_cloud_top_pressure_matches(self):
        rng = np.random.default_rng(13)
        nlev, ncol = 9, 30
        phalf = np.linspace(4000.0, 101300.0, nlev + 1)
        tau = np.where(rng.uniform(size=(nlev, ncol)) < 0.6,
                       rng.uniform(0.0, 2.0, (nlev, ncol)), 0.0)
        out = np.asarray(msim.cloud_top_pressure(tau, phalf, 1.0))
        for c in range(ncol):
            np.testing.assert_allclose(
                out[c], cloud_top_pressure_reference(tau[:, c], phalf, 1.0),
                rtol=1e-12, err_msg=f"column {c}")

    def test_bottom_layer_contributes_after_fix(self):
        # FIXED UPSTREAM BUG: the Fortran call-site truncation drops the
        # lowest layer, so a cloud confined there yields CTP = 0; the fixed
        # integral places the cloud top inside the bottom layer.
        nlev = 6
        phalf = np.linspace(10000.0, 100000.0, nlev + 1)
        tau = np.zeros((nlev, 1))
        tau[-1, 0] = 5.0
        out = float(np.asarray(msim.cloud_top_pressure(tau, phalf, 1.0))[0])
        self.assertGreater(out, phalf[-2])
        self.assertLess(out, phalf[-1])
        # The documented difference from the Fortran (legacy) form:
        self.assertEqual(
            cloud_top_pressure_reference(tau[:, 0], phalf, 1.0, legacy=True),
            0.0)

    def test_weight_by_extinction_matches(self):
        rng = np.random.default_rng(14)
        nlev, ncol = 9, 30
        tau = np.where(rng.uniform(size=(nlev, ncol)) < 0.6,
                       rng.uniform(0.0, 2.0, (nlev, ncol)), 0.0)
        f = rng.uniform(size=(nlev, ncol))
        out = np.asarray(msim.weight_by_extinction(tau, f, 1.0))
        for c in range(ncol):
            np.testing.assert_allclose(
                out[c], weight_by_extinction_reference(tau[:, c], f[:, c],
                                                       1.0),
                rtol=1e-12, err_msg=f"column {c}")


class InterpolateToMinTest(unittest.TestCase):
    def test_matches_transliteration(self):
        rng = np.random.default_rng(15)
        x = np.linspace(4.0, 30.0, 15)
        for _ in range(100):
            y = np.sort(rng.uniform(0.1, 0.8, 15))[::-1].copy()
            yobs = rng.uniform(0.0, 1.0)  # sometimes out of range -> fill
            out = float(msim.interpolate_to_min(x, jnp.asarray(y),
                                                jnp.asarray(yobs)))
            np.testing.assert_allclose(
                out, interpolate_to_min_reference(x, y, yobs), rtol=1e-12)

    def test_recovers_linear_root(self):
        # For y linear in x the interpolation is exact between nodes.
        x = np.linspace(4.0, 30.0, 15)
        y = 0.9 - 0.02 * x
        out = float(msim.interpolate_to_min(x, jnp.asarray(y),
                                            jnp.asarray(0.9 - 0.02 * 17.3)))
        np.testing.assert_allclose(out, 17.3, rtol=1e-12)

    def test_exact_node_hit_returns_node(self):
        # FIXED UPSTREAM BUG: the Fortran's strictly-negative bracket test
        # turns an exactly-matched trial reflectance into the fill value;
        # the fixed boundary case returns the matching trial radius.
        x = np.linspace(4.0, 30.0, 15)
        y = 1.0 / x
        out = float(msim.interpolate_to_min(x, jnp.asarray(y),
                                            jnp.asarray(1.0 / x[7])))
        np.testing.assert_allclose(out, x[7], rtol=1e-12)
        # The documented difference from the Fortran reference:
        self.assertEqual(interpolate_to_min_reference(x, y, 1.0 / x[7]),
                         config.MODIS_RE_FILL)


class ModisSubcolumnTest(unittest.TestCase):
    def test_matches_transliteration(self):
        rng = np.random.default_rng(16)
        for use_isccp in (False, True):
            tau, liq, g, w0, phalf = _random_scene(rng)
            isccp = rng.uniform(50000.0, 95000.0, tau.shape[1])
            out = msim.modis_subcolumn(
                phalf, tau, liq, g, w0,
                isccp_ctp=jnp.asarray(isccp) if use_isccp else None)
            ref = modis_subcolumn_reference(
                phalf, tau.T, liq.T, g.T, w0.T,
                isccp_ctp=isccp if use_isccp else None)
            np.testing.assert_array_equal(np.asarray(out.phase), ref[0],
                                          err_msg=f"isccp={use_isccp}")
            np.testing.assert_allclose(np.asarray(out.cloud_top_pressure),
                                       ref[1], rtol=1e-11)
            np.testing.assert_allclose(np.asarray(out.tau), ref[2],
                                       rtol=1e-12)
            np.testing.assert_allclose(np.asarray(out.size), ref[3],
                                       rtol=1e-9)

    def test_pure_liquid_size_recovered(self):
        # A single-phase liquid cloud built from the fits at re0 must
        # retrieve a size close to re0 (pseudo-retrieval interpolation).
        nlev, ncol = 4, 5
        re0 = np.array([6.0, 10.0, 14.0, 20.0, 26.0])  # microns
        tau = np.full((nlev, ncol), 2.0)
        g = np.asarray(optics.get_g_nir(1, jnp.asarray(
            np.broadcast_to(re0, (nlev, ncol)))))
        w0 = np.asarray(optics.get_ssa_nir(1, jnp.asarray(
            np.broadcast_to(re0, (nlev, ncol)))))
        liq = np.ones((nlev, ncol))
        phalf = np.linspace(30000.0, 90000.0, nlev + 1)
        out = msim.modis_subcolumn(phalf, tau, liq, g, w0)
        self.assertTrue((np.asarray(out.phase) == 1).all())
        np.testing.assert_allclose(np.asarray(out.size), re0 * 1e-6,
                                   rtol=0.02)

    def test_ctp_within_column_range(self):
        rng = np.random.default_rng(17)
        tau, liq, g, w0, phalf = _random_scene(rng)
        out = msim.modis_subcolumn(phalf, tau, liq, g, w0)
        ctp = np.asarray(out.cloud_top_pressure)
        cloudy = np.asarray(out.phase) != 0
        # ctp = 0 can only occur for cloud confined to the bottom layer.
        ok = (ctp[cloudy] >= 0.0) & (ctp[cloudy] <= phalf.max())
        self.assertTrue(ok.all())

    def test_gradient_through_retrieval(self):
        rng = np.random.default_rng(18)
        tau, liq, g, w0, phalf = _random_scene(rng)

        def mean_size(tau_in):
            out = msim.modis_subcolumn(phalf, tau_in, liq, g, w0)
            return jnp.sum(jnp.where(out.size > 0, out.size, 0.0))

        grad = np.asarray(jax.grad(mean_size)(jnp.asarray(tau)))
        self.assertTrue(np.isfinite(grad).all())
        self.assertGreater(np.abs(grad).max(), 0.0)


class ModisColumnTest(unittest.TestCase):
    def test_matches_transliteration(self):
        rng = np.random.default_rng(19)
        tau, liq, g, w0, phalf = _random_scene(rng, nlev=7, ncol=20)
        sub = msim.modis_subcolumn(phalf, tau, liq, g, w0)
        col = msim.modis_column(sub.phase, sub.cloud_top_pressure, sub.tau,
                                sub.size)
        ref = modis_column_reference(np.asarray(sub.phase),
                                     np.asarray(sub.cloud_top_pressure),
                                     np.asarray(sub.tau),
                                     np.asarray(sub.size))
        for name, want in ref.items():
            got = np.asarray(getattr(col, name))
            np.testing.assert_allclose(got, want, rtol=1e-10,
                                       err_msg=name)

    def test_histogram_sums_to_cloud_fraction(self):
        rng = np.random.default_rng(20)
        tau, liq, g, w0, phalf = _random_scene(rng, ncol=30)
        sub = msim.modis_subcolumn(phalf, tau, liq, g, w0)
        col = msim.modis_column(sub.phase, sub.cloud_top_pressure, sub.tau,
                                sub.size)
        np.testing.assert_allclose(float(np.asarray(col.tau_vs_ctp).sum()),
                                   float(np.asarray(col.cf_total)),
                                   rtol=1e-10)

    def test_batched_agrees_with_single(self):
        rng = np.random.default_rng(21)
        scenes = [_random_scene(rng) for _ in range(3)]
        tau = np.stack([s[0] for s in scenes], axis=-1)
        liq = np.stack([s[1] for s in scenes], axis=-1)
        g = np.stack([s[2] for s in scenes], axis=-1)
        w0 = np.stack([s[3] for s in scenes], axis=-1)
        phalf = np.stack([s[4] for s in scenes], axis=-1)
        sub = msim.modis_subcolumn(phalf, tau, liq, g, w0)
        col = msim.modis_column(sub.phase, sub.cloud_top_pressure, sub.tau,
                                sub.size)
        for b, (tau1, liq1, g1, w01, phalf1) in enumerate(scenes):
            sub1 = msim.modis_subcolumn(phalf1, tau1, liq1, g1, w01)
            col1 = msim.modis_column(sub1.phase, sub1.cloud_top_pressure,
                                     sub1.tau, sub1.size)
            for name in msim.ModisColumn._fields:
                np.testing.assert_allclose(
                    np.asarray(getattr(col, name))[..., b],
                    np.asarray(getattr(col1, name)), rtol=1e-12,
                    err_msg=f"{name} batch {b}")


if __name__ == "__main__":
    unittest.main()
