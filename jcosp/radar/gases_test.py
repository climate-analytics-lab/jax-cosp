"""Sanity and regression tests for the Liebe-85 gaseous attenuation port."""

import unittest

import numpy as np

from jcosp.radar.gases import gases


class GasesTest(unittest.TestCase):
    def test_surface_magnitude_94ghz(self):
        # Two-way 94-GHz attenuation in a moist near-surface layer is of
        # order 1 dB/km (Liebe 1985 / CloudSat literature).
        g = float(gases(101325.0, 288.0, 0.01, 94.0))
        self.assertGreater(g, 0.3)
        self.assertLess(g, 5.0)

    def test_humidity_increases_attenuation(self):
        dry = float(gases(101325.0, 288.0, 1e-4, 94.0))
        moist = float(gases(101325.0, 288.0, 0.015, 94.0))
        self.assertGreater(moist, dry)

    def test_pressure_scaling(self):
        # Attenuation drops sharply with altitude/pressure.
        low = float(gases(101325.0, 288.0, 0.005, 94.0))
        high = float(gases(20000.0, 220.0, 1e-5, 94.0))
        self.assertLess(high, 0.3 * low)

    def test_shapes_broadcast(self):
        p = np.linspace(2e4, 1e5, 7)
        t = np.linspace(220, 290, 7)
        q = np.linspace(1e-5, 1e-2, 7)
        out = np.asarray(gases(p, t, q, 94.0))
        self.assertEqual(out.shape, (7,))
        self.assertTrue((out > 0).all())

    def test_against_compiled_fortran(self):
        # References computed 2026-07-09 by compiling quickbeam_optics.F90's
        # gases() with gfortran (double precision). The port agrees to ~2e-8
        # relative; the residual is the Fortran's single-precision coefficient
        # literals (e.g. 0.0000001 without _wp) rounding at compile time,
        # while our tables hold the decimal values in float64.
        cases = [
            ((101325.0, 288.0, 0.01), 1.2412674775933821),
            ((85000.0, 275.0, 0.005), 0.67301157581038629),
            ((20000.0, 220.0, 1e-5), 0.059211101367551602),
        ]
        for (p, t, q), ref in cases:
            g = float(gases(p, t, q, 94.0))
            self.assertAlmostEqual(g, ref, delta=abs(ref) * 1e-5,
                                   msg=f"gases({p},{t},{q})={g!r}")


if __name__ == "__main__":
    unittest.main()
