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

    def test_dry_air_matches_literature(self):
        # The corrected O2 interference term is anchored to the well-known
        # W-band dry-air value: ~0.045-0.065 dB/km ONE-way at sea level
        # (ITU-R P.676 / MPM). The uncorrected Fortran gives ~0.23.
        g_one_way = float(gases(101325.0, 288.0, 1e-6, 94.0)) / 2.0
        self.assertGreater(g_one_way, 0.04)
        self.assertLess(g_one_way, 0.07)

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

    # References computed 2026-07-09 by compiling quickbeam_optics.F90's
    # gases() with gfortran (double precision).
    _FORTRAN_REFS = [
        ((101325.0, 288.0, 0.01), 1.2412674775933821),
        ((85000.0, 275.0, 0.005), 0.67301157581038629),
        ((20000.0, 220.0, 1e-5), 0.059211101367551602),
    ]

    def test_legacy_mode_matches_compiled_fortran(self):
        # With legacy_o2_interference the port agrees with the compiled
        # Fortran to ~2e-8 relative; the residual is the Fortran's
        # single-precision coefficient literals rounding at compile time,
        # while our tables hold the decimal values in float64.
        for (p, t, q), ref in self._FORTRAN_REFS:
            g = float(gases(p, t, q, 94.0, legacy_o2_interference=True))
            self.assertAlmostEqual(g, ref, delta=abs(ref) * 1e-5,
                                   msg=f"gases({p},{t},{q})={g!r}")

    def test_fixed_mode_regression_and_documented_difference(self):
        # Post-fix frozen references (2026-07-10) and the expected departure
        # from the Fortran: the corrected O2 interference REDUCES attenuation
        # everywhere, by up to ~5x in dry air (docs/translation_notes.md).
        fixed_refs = [
            ((101325.0, 288.0, 0.01), 0.8964498863556891),
            ((85000.0, 275.0, 0.005), 0.3598822615123132),
            ((20000.0, 220.0, 1e-5), 0.011273356758285964),
        ]
        for ((p, t, q), ref), (_, fort) in zip(fixed_refs, self._FORTRAN_REFS):
            g = float(gases(p, t, q, 94.0))
            self.assertAlmostEqual(g, ref, delta=abs(ref) * 1e-4,
                                   msg=f"gases({p},{t},{q})={g!r}")
            self.assertLess(g, fort)


if __name__ == "__main__":
    unittest.main()
