"""Tests for the offline Mie/DSD radar-optics core and LUT generator.

Validates the NumPy ports of COSP v2.0 quickbeam_optics/optics_lib/math_lib
against analytic limits (Rayleigh), conservation (DSD mass closure), and
internal consistency (calc_re/dsd roundtrip, LUT linearity, Re-bin mapping).
"""

import math
import unittest

import numpy as np
import pytest

from jcosp.config import (
    D_GRID,
    HYDRO_CLASSES_1M,
    I_LSCICE,
    I_LSCLIQ,
    I_LSRAIN,
    MT_TTL,
    N_RE_TYPES,
)
from jcosp.radar.lut.dsd import avint, calc_re, dsd, fgamma, infind, zeff
from jcosp.radar.lut.generate import (
    enumerate_re_bins,
    generate_lut,
    hydro_optics_entry,
    re_quantize,
)
from jcosp.radar.lut.mie import m_ice, m_wat, mie_int

_SD = HYDRO_CLASSES_1M


def _class_args(tp):
    """(dmin, dmax, apm, bpm, rho, p1, p2, p3) scalars for class tp."""
    return (float(_SD.dmin[tp]), float(_SD.dmax[tp]), float(_SD.apm[tp]),
            float(_SD.bpm[tp]), float(_SD.rho[tp]), float(_SD.p1[tp]),
            float(_SD.p2[tp]), float(_SD.p3[tp]))


class TestGamma(unittest.TestCase):
    def test_known_values(self):
        self.assertEqual(fgamma(4.0), 6.0)
        self.assertAlmostEqual(fgamma(0.5), math.sqrt(math.pi), places=14)
        # Recurrence consistency at a non-integer DSD-typical argument.
        self.assertAlmostEqual(fgamma(4.91), 3.91 * fgamma(3.91), places=10)


class TestAvint(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        # Uneven, strictly increasing grid.
        self.x = np.sort(rng.uniform(0.0, 2.0, 40))

    def test_quadratic_exact(self):
        # Overlapping parabolas integrate quadratics exactly.
        f = 3.0 * self.x ** 2 - 2.0 * self.x + 0.7
        exact = (self.x[-1] ** 3 - self.x[0] ** 3) \
            - (self.x[-1] ** 2 - self.x[0] ** 2) + 0.7 * (self.x[-1] - self.x[0])
        got = avint(f, self.x, self.x[0], self.x[-1])
        # Exact up to floating-point roundoff (the ca/cb/cc combination
        # subtracts large intermediates, so allow ~1e-10 relative).
        self.assertAlmostEqual(got / exact, 1.0, delta=1e-10)

    def test_cubic_close(self):
        f = self.x ** 3
        exact = (self.x[-1] ** 4 - self.x[0] ** 4) / 4.0
        got = avint(f, self.x, self.x[0], self.x[-1])
        trap = np.trapezoid(f, self.x)
        # Better than trapezoid, close to exact for a smooth function.
        self.assertLess(abs(got - exact), abs(trap - exact) + 1e-12)
        self.assertAlmostEqual(got, exact, delta=1e-3 * abs(exact))

    def test_reversed_limits_flip_sign(self):
        f = np.ones_like(self.x)
        fwd = avint(f, self.x, self.x[0], self.x[-1])
        rev = avint(f, self.x, self.x[-1], self.x[0])
        self.assertAlmostEqual(fwd, -rev, places=14)

    def test_errors(self):
        with self.assertRaises(ValueError):
            avint([1.0, 2.0], [0.0, 1.0], 0.0, 1.0)  # ntab < 3
        with self.assertRaises(ValueError):
            avint([1.0, 2.0, 3.0], [0.0, 1.0, 1.0], 0.0, 1.0)  # not ascending
        self.assertEqual(avint(self.x, self.x, 0.5, 0.5), 0.0)  # a == b


class TestInfind(unittest.TestCase):
    def test_semantics(self):
        arr = np.array([0.0, 1.0, 2.0, 4.0])
        self.assertEqual(infind(arr, -5.0), 0)   # clamp low
        self.assertEqual(infind(arr, 10.0), 3)   # clamp high
        self.assertEqual(infind(arr, 1.1), 1)    # nearer below
        self.assertEqual(infind(arr, 3.9), 3)    # nearer above
        self.assertEqual(infind(arr, 3.0), 3)    # tie -> first >= val
        self.assertEqual(infind(arr, 2.0), 2)    # exact hit


class TestRefractiveIndices(unittest.TestCase):
    def test_m_wat_94ghz(self):
        n_r, n_i = m_wat(94.0, 280.0)
        print(f"m_wat(94 GHz, 280 K) = {n_r:.4f} + {n_i:.4f}i")
        self.assertTrue(np.isfinite(n_r) and np.isfinite(n_i))
        self.assertGreater(n_r, 2.0)
        self.assertLess(n_r, 8.0)
        self.assertGreater(n_i, 0.0)

    def test_m_ice_94ghz(self):
        n_r, n_i = m_ice(94.0, 260.0)
        print(f"m_ice(94 GHz, 260 K) = {n_r:.4f} + {n_i:.6f}i")
        # Microwave branch: n_r ~ 1.787 (Warren 1984), tiny absorption.
        self.assertAlmostEqual(n_r, 1.78, delta=0.03)
        self.assertGreater(n_i, 0.0)
        self.assertLess(n_i, 0.05)

    def test_m_ice_temperature_clamped(self):
        # Below the coldest reference temperature the result must equal the
        # clamped value (Fortran clamps tk into [213.16, 272.16]).
        self.assertEqual(m_ice(94.0, 200.0), m_ice(94.0, 213.16))
        self.assertEqual(m_ice(94.0, 300.0), m_ice(94.0, 272.16))


class TestMieInt(unittest.TestCase):
    def test_rayleigh_limit(self):
        x = 0.01
        m = 1.78 - 0.003j
        qext, qsca, qbsc = mie_int(x, m)
        lorenz = (m * m - 1) / (m * m + 2)
        qbsc_ray = 4.0 * x ** 4 * abs(lorenz) ** 2
        qext_ray = 4.0 * x * (-lorenz).imag
        self.assertAlmostEqual(qbsc / qbsc_ray, 1.0, delta=0.01)
        self.assertAlmostEqual(qext / qext_ray, 1.0, delta=0.05)
        self.assertGreater(qsca, 0.0)
        self.assertLess(qsca, qext)

    def test_large_sphere_extinction_limit(self):
        # Extinction paradox: qext -> 2 for large absorbing spheres.
        qext, _, _ = mie_int(500.0, 1.78 - 0.5j)
        self.assertAlmostEqual(qext, 2.0, delta=0.15)


class TestDsd(unittest.TestCase):
    def test_mass_closure_lsrain(self):
        # Exponential rain (N0 = 8e6): integrating apm*D^bpm * n(D) over the
        # size grid must recover rho_a*Q (kg m-3) to grid truncation.
        q, rho_a = 1.0, 1.0
        tp = I_LSRAIN
        dmin, dmax, apm, bpm, rho, p1, p2, p3 = _class_args(tp)
        n, apm_o, bpm_o, _ = dsd(q, 0.0, 0.0, D_GRID, int(_SD.dtype[tp]),
                                 rho_a, 273.15, dmin, dmax, apm, bpm, rho,
                                 p1, p2, p3)
        # n is cm^-3 um^-1 -> m^-3 m^-1 via 1e6 * 1e6.
        n_si = n * 1e12
        mass = np.trapezoid(apm_o * (D_GRID * 1e-6) ** bpm_o * n_si,
                            D_GRID * 1e-6)
        self.assertAlmostEqual(mass, rho_a * q * 1e-3,
                               delta=0.02 * rho_a * q * 1e-3)

    def test_calc_re_dsd_roundtrip_lognormal(self):
        # LSCLIQ lognormal defaults: Re from calc_re, then dsd at that Re;
        # the discrete distribution's (M3/M2)/2 must reproduce Re.
        tp = I_LSCLIQ
        dmin, dmax, apm, bpm, rho, p1, p2, p3 = _class_args(tp)
        re, apm_o, bpm_o = calc_re(1.0, 0.0, 1.0, int(_SD.dtype[tp]),
                                   apm, bpm, rho, p1, p2, p3)
        # rg = p2 = 6 um, sigma = 0.3: Re = 6*exp(2.5*0.09).
        self.assertAlmostEqual(re, 6.0 * math.exp(2.5 * 0.09), places=10)
        n, _, _, _ = dsd(1.0, re, 0.0, D_GRID, int(_SD.dtype[tp]), 1.0,
                         273.15, dmin, dmax, apm_o, bpm_o, rho, p1, p2, p3)
        m3 = np.trapezoid(n * D_GRID ** 3, D_GRID)
        m2 = np.trapezoid(n * D_GRID ** 2, D_GRID)
        self.assertAlmostEqual(0.5 * m3 / m2, re, delta=0.03 * re)

    def test_rho_substitution_returned(self):
        # LSSNOW: apm/bpm unset, rho = 100 -> apm = pi/6*100, bpm = 3.
        tp = 3
        dmin, dmax, apm, bpm, rho, p1, p2, p3 = _class_args(tp)
        _, apm_o, bpm_o, _ = dsd(1.0, 50.0, 0.0, D_GRID, int(_SD.dtype[tp]),
                                 1.0, 260.0, dmin, dmax, apm, bpm, rho,
                                 p1, p2, p3)
        self.assertAlmostEqual(apm_o, math.pi / 6 * 100.0, places=10)
        self.assertEqual(bpm_o, 3.0)


class TestZeff(unittest.TestCase):
    def test_rayleigh_consistency_small_droplets(self):
        # Small cloud droplets at 94 GHz scatter in the Rayleigh regime, so
        # z_eff (with the standard k2 = 0.75) should be close to z_ray.
        tp = I_LSCLIQ
        dmin, dmax, apm, bpm, rho, p1, p2, p3 = _class_args(tp)
        n, _, _, _ = dsd(1.0, 10.0, 0.0, D_GRID, int(_SD.dtype[tp]), 1.0,
                         280.0, dmin, dmax, apm, bpm, rho, p1, p2, p3)
        z_eff, z_ray, kr = zeff(94.0, D_GRID, n, -1.0, 280.0, 0, 1,
                                np.full(D_GRID.shape, -1.0))
        self.assertGreater(z_eff, 0.0)
        self.assertGreater(kr, 0.0)
        self.assertAlmostEqual(z_eff / z_ray, 1.0, delta=0.10)


class TestReQuantize(unittest.TestCase):
    def test_monotone_and_bounded(self):
        # Faithful-to-Fortran quirk: at exact coarse-interval boundaries
        # (Re = 90, 120, ... um) the slot index steps DOWN by exactly one
        # (interval n overruns into interval n+1's first slot), so assert
        # monotonicity up to that single-slot backstep.
        res = np.arange(0.01, 3000.0, 0.13)
        prev_idx = 0
        max_idx = 0
        for re in res:
            idx, re_q = re_quantize(float(re))
            self.assertGreaterEqual(idx, 1)
            self.assertLessEqual(idx, N_RE_TYPES)
            self.assertGreaterEqual(idx, prev_idx - 1)
            self.assertGreaterEqual(idx, max_idx - 1)
            prev_idx = idx
            max_idx = max(max_idx, idx)
            self.assertGreater(re_q, 0.0)

    def test_zero_re_is_slot_one(self):
        idx, re_q = re_quantize(0.0)
        self.assertEqual(idx, 1)
        self.assertEqual(re_q, 0.0)

    def test_enumerate_consistency(self):
        indices, re_quants = enumerate_re_bins()
        # Unique, sorted, within the usable range (overflow slot excluded).
        self.assertTrue(np.all(np.diff(indices) > 0))
        self.assertGreaterEqual(indices[0], 1)
        self.assertLess(indices[-1], N_RE_TYPES)
        # Quantized Re grows with slot index.
        self.assertTrue(np.all(np.diff(re_quants) > 0))
        # Round trip: quantizing a slot's Re must land back on that slot.
        for idx, re_q in list(zip(indices, re_quants))[::25]:
            idx2, _ = re_quantize(float(re_q))
            self.assertEqual(idx2, idx)


class TestLutGeneration(unittest.TestCase):
    def test_linearity_in_q(self):
        # The LUT stores Ze/(rho_a*Q); doubling Q must exactly double zeff.
        tp = I_LSRAIN
        t_kelvin, re_um = 273.15, 100.25
        ze1, zr1, kr1 = hydro_optics_entry(_SD, tp, t_kelvin, re_um, 94.0,
                                           -1.0, q_gkg=1.0)
        ze2, zr2, kr2 = hydro_optics_entry(_SD, tp, t_kelvin, re_um, 94.0,
                                           -1.0, q_gkg=2.0)
        # hydro_optics_entry divides by Q, so scaled values are identical...
        self.assertEqual(ze1, ze2)
        self.assertEqual(kr1, kr2)
        # ...and the raw path at Q=2 is exactly 2x the Q=1 zeff.
        dmin, dmax, apm, bpm, rho, p1, p2, p3 = _class_args(tp)
        n1, *_ = dsd(1.0, re_um, 0.0, D_GRID, 2, 1.0, t_kelvin, dmin, dmax,
                     apm, bpm, rho, p1, p2, p3)
        n2, *_ = dsd(2.0, re_um, 0.0, D_GRID, 2, 1.0, t_kelvin, dmin, dmax,
                     apm, bpm, rho, p1, p2, p3)
        rho_e = np.full(D_GRID.shape, -1.0)
        za, _, ka = zeff(94.0, D_GRID, n1, -1.0, t_kelvin, 0, 1, rho_e)
        zb, _, kb = zeff(94.0, D_GRID, n2, -1.0, t_kelvin, 0, 1, rho_e)
        self.assertEqual(zb, 2.0 * za)
        self.assertEqual(kb, 2.0 * ka)

    def test_entry_matches_dedupe_path(self):
        # The Mie-dedupe (qe/qs) path in generate_lut must agree exactly
        # with hydro_optics_entry's direct zeff computation.
        from jcosp.radar.lut.generate import (
            _class_deq_rhoe,
            _mie_for_class_temperature,
        )
        for tp in (I_LSCLIQ, I_LSCICE, I_LSRAIN):
            dtype = int(_SD.dtype[tp])
            phase = int(_SD.phase[tp])
            t_kelvin, re_um = 265.65, 45.25
            deq, rho_e = _class_deq_rhoe(_SD, tp)
            qext, qbsca = _mie_for_class_temperature(deq, rho_e, phase,
                                                     t_kelvin, 94.0)
            dmin, dmax, apm, bpm, rho, p1, p2, p3 = _class_args(tp)
            n, *_ = dsd(1.0, re_um, 0.0, D_GRID, dtype, 1.0, t_kelvin,
                        dmin, dmax, apm, bpm, rho, p1, p2, p3)
            ze_a, zr_a, kr_a = zeff(94.0, deq, n, -1.0, t_kelvin, phase, 1,
                                    rho_e, qe=qext, qs=qbsca)
            ze_b, zr_b, kr_b = hydro_optics_entry(_SD, tp, t_kelvin, re_um,
                                                  94.0, -1.0)
            self.assertEqual(ze_a, ze_b)
            self.assertEqual(zr_a, zr_b)
            self.assertEqual(kr_a, kr_b)

    def test_few_entries_plausible(self):
        # Rain, 1 g/kg at 0 C with Re = 500.25 um should be tens of dBZ;
        # tiny cloud droplets far less.
        ze_rain, zr_rain, kr_rain = hydro_optics_entry(
            _SD, I_LSRAIN, 273.15, 500.0, 94.0, -1.0)
        ze_cld, _, _ = hydro_optics_entry(
            _SD, I_LSCLIQ, 273.15, 10.0, 94.0, -1.0)
        self.assertGreater(ze_rain, ze_cld)
        self.assertGreater(10 * np.log10(ze_rain), 0.0)
        self.assertGreater(kr_rain, 0.0)
        # Rayleigh reflectivity should exceed attenuated Mie at 94 GHz for
        # large drops.
        self.assertGreater(zr_rain, ze_rain)

    def test_2m_not_implemented(self):
        with self.assertRaises(NotImplementedError):
            generate_lut(scheme="2M")

    @pytest.mark.slow
    def test_full_table_generation(self):
        lut = generate_lut(scheme="1M", freq=94.0, k2=-1.0)
        ze = lut["ze_scaled"]
        self.assertEqual(ze.shape[0], 9)
        occ = np.isfinite(ze)
        # Liquid classes fill 19 T slots, ice 20, over the reachable Re bins.
        indices, _ = enumerate_re_bins()
        n_bins = indices.size
        expected = (4 * 19 + 5 * 20) * n_bins
        self.assertEqual(occ.sum(), expected)
        # All occupied Ze/kr are positive and finite.
        self.assertTrue(np.all(ze[occ] > 0))
        self.assertTrue(np.all(lut["kr_scaled"][np.isfinite(lut["kr_scaled"])]
                               > 0))
        # LSR at 0C spans a wide dynamic range across Re bins.
        itt = int(np.argmin(np.abs(MT_TTL - 273.15)))
        lsr = ze[I_LSRAIN, itt, :]
        lsr = lsr[np.isfinite(lsr)]
        dbz = 10 * np.log10(lsr)
        print(f"LSR @0C dBZ range: {dbz.min():.1f} .. {dbz.max():.1f}")
        self.assertLess(dbz.min(), -20.0)
        self.assertGreater(dbz.max(), 10.0)


if __name__ == "__main__":
    unittest.main()
