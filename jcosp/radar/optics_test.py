"""Tests for the runtime optics: Re quantization, temperature index, calc_Re."""

import unittest

import numpy as np

from jcosp import config
from jcosp.radar import optics


def quantize_re_reference(re):
    """Transliteration of the Fortran Re->iRe_type mapping (1-based out)."""
    i_re_type = 1
    if re > 0:
        n = int(re / config.RE_BIN_LENGTH)
        if n >= config.RE_MAX_BIN:
            n = config.RE_MAX_BIN - 1
        step = config.RE_STEP_LIST[n]
        base = config.RE_BASE_LIST[n]
        i_re = int(re / step)
        if i_re < 1:
            i_re = 1
        re_q = step * (i_re + 0.5)
        i_re_type = i_re + int(base) - int(n * config.RE_BIN_LENGTH / step)
        if i_re_type >= config.N_RE_TYPES:
            i_re_type = config.N_RE_TYPES
        return i_re_type, re_q
    return i_re_type, re


def infind_reference(grid, val):
    """Transliteration of array_lib.F90:infind (0-based out)."""
    if val >= grid[-1]:
        return len(grid) - 1
    if val <= grid[0]:
        return 0
    up = int(np.argmax(grid >= val))
    if abs(grid[up - 1] - val) < abs(grid[up] - val):
        return up - 1
    return up


class QuantizeReTest(unittest.TestCase):
    def test_matches_fortran_mapping(self):
        res = np.concatenate([np.arange(0.05, 60, 0.37),
                              np.arange(60, 2600, 7.3), [5000.0, 1e4]])
        idx, re_q = optics.quantize_re(res)
        idx = np.asarray(idx)
        re_q = np.asarray(re_q)
        for i, re in enumerate(res):
            ref_idx, ref_req = quantize_re_reference(float(re))
            self.assertEqual(int(idx[i]) + 1, ref_idx, msg=f"Re={re}")
            if ref_idx < config.N_RE_TYPES:  # capped slot keeps its own re_q
                self.assertAlmostEqual(float(re_q[i]), ref_req, places=9,
                                       msg=f"Re={re}")

    def test_nearly_monotone_within_bounds(self):
        # The Fortran base/step arithmetic overlaps neighbouring slots by one
        # at a handful of coarse-interval boundaries (e.g. Re=90, 120, 150
        # um), so the mapping is only *nearly* monotone; the port reproduces
        # those single-slot dips faithfully.
        res = np.arange(1.0, 2600.0, 0.11)
        idx = np.asarray(optics.quantize_re(res)[0])
        d = np.diff(idx)
        self.assertGreaterEqual(d.min(), -1)
        self.assertLessEqual((d < 0).sum(), 10)
        self.assertGreaterEqual(idx.min(), 0)
        self.assertLess(idx.max(), config.N_RE_TYPES)


class TemperatureIndexTest(unittest.TestCase):
    def test_matches_infind(self):
        for phase, grid in ((0, config.MT_TTL), (1, config.MT_TTI)):
            temps = np.arange(150.0, 330.0, 0.313)
            idx = np.asarray(optics.temperature_index(temps, phase))
            for t, i in zip(temps, idx):
                self.assertEqual(int(i), infind_reference(grid, t),
                                 msg=f"phase={phase} T={t}")


class CalcReCrossCheckTest(unittest.TestCase):
    def test_matches_numpy_port_for_1m_classes(self):
        # The runtime JAX calc_re_class and the offline NumPy calc_re (the
        # LUT generator's exact Fortran port) must agree for every 1M class.
        from jcosp.radar.lut.dsd import calc_re as calc_re_np

        sd = config.HYDRO_CLASSES_1M
        rng = np.random.default_rng(23)
        for tp in range(config.N_HYDRO):
            for _ in range(4):
                q = float(rng.uniform(0.001, 2.0))     # g/kg
                rho_a = float(rng.uniform(0.3, 1.3))   # kg/m3
                ref, _, _ = calc_re_np(
                    q, 0.0, rho_a, int(sd.dtype[tp]), float(sd.apm[tp]),
                    float(sd.bpm[tp]), float(sd.rho[tp]), float(sd.p1[tp]),
                    float(sd.p2[tp]), float(sd.p3[tp]))
                got = float(optics.calc_re_class(
                    np.array(q), None, np.array(rho_a),
                    dtype=int(sd.dtype[tp]), apm=float(sd.apm[tp]),
                    bpm=float(sd.bpm[tp]), rho=float(sd.rho[tp]),
                    p1=float(sd.p1[tp]), p2=float(sd.p2[tp]),
                    p3=float(sd.p3[tp])))
                self.assertAlmostEqual(
                    got, ref, delta=abs(ref) * 1e-9 + 1e-12,
                    msg=f"class {config.HYDRO_NAMES[tp]} q={q} rho_a={rho_a}")


class CalcReTest(unittest.TestCase):
    def test_1m_defaults(self):
        # LSL: lognormal defaults rg=6 um, sigma_g: Re = rg*exp(2.5 p3^2).
        sd = config.HYDRO_CLASSES_1M
        i = config.I_LSCLIQ
        re = optics.calc_re_class(
            np.array(0.1), None, np.array(1.0), dtype=int(sd.dtype[i]),
            apm=float(sd.apm[i]), bpm=float(sd.bpm[i]), rho=float(sd.rho[i]),
            p1=float(sd.p1[i]), p2=float(sd.p2[i]), p3=float(sd.p3[i]))
        self.assertAlmostEqual(float(re), 6.0 * np.exp(2.5 * 0.3**2), places=4)

    def test_1m_rain_scales_with_water_content(self):
        # Exponential PSD with fixed N0: Re ~ (rho_a q)^(1/4).
        sd = config.HYDRO_CLASSES_1M
        i = config.I_LSRAIN
        kw = dict(dtype=int(sd.dtype[i]), apm=float(sd.apm[i]),
                  bpm=float(sd.bpm[i]), rho=float(sd.rho[i]),
                  p1=float(sd.p1[i]), p2=float(sd.p2[i]), p3=float(sd.p3[i]))
        re1 = float(optics.calc_re_class(np.array(0.1), None, np.array(1.0), **kw))
        re2 = float(optics.calc_re_class(np.array(1.6), None, np.array(1.0), **kw))
        self.assertAlmostEqual(re2 / re1, 16.0 ** 0.25, places=5)
        # Plausible rain-drop effective radius for 0.1 g/kg: tens to ~500 um.
        self.assertGreater(re1, 50.0)
        self.assertLess(re1, 1000.0)

    def test_snow_constant_density_substitution(self):
        # LSS has rho=100, apm<0: the (pi/6)rho substitution must engage and
        # produce a finite radius.
        sd = config.HYDRO_CLASSES_1M
        i = config.I_LSSNOW
        re = float(optics.calc_re_class(
            np.array(0.05), None, np.array(0.7), dtype=int(sd.dtype[i]),
            apm=float(sd.apm[i]), bpm=float(sd.bpm[i]), rho=float(sd.rho[i]),
            p1=float(sd.p1[i]), p2=float(sd.p2[i]), p3=float(sd.p3[i])))
        self.assertGreater(re, 10.0)
        self.assertTrue(np.isfinite(re))


if __name__ == "__main__":
    unittest.main()
