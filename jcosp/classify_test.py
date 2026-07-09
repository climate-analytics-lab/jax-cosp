"""Tests for the Mülmenstädt et al. (2020) warm-rain classification."""

import unittest

import numpy as np

from jcosp.classify import (WarmRainFlags, warm_rain_classification,
                            warm_rain_fraction)
from jcosp.constants import R_UNDEF


def _column(dbze_profile, liq_levels, ice_levels, nlev=10):
    """Build (nlev, ncol=1, batch=1) single-subcolumn inputs from level lists."""
    dbze = np.full((nlev, 1, 1), R_UNDEF)
    liq = np.zeros((nlev, 1, 1))
    ice = np.zeros((nlev, 1, 1))
    for k, v in dbze_profile.items():
        dbze[k, 0, 0] = v
    liq[list(liq_levels), 0, 0] = 1e-4
    if ice_levels:
        ice[list(ice_levels), 0, 0] = 1e-4
    return dbze, liq, ice


class WarmRainClassificationTest(unittest.TestCase):
    def _classify(self, dbze, liq, ice, raining=True):
        return warm_rain_classification(
            dbze, liq, ice, np.array([[raining]]))

    def test_warm_rain(self):
        # Single liquid cloud layer with 5 dBZe: warm rain.
        dbze, liq, ice = _column({6: 5.0, 7: 2.0}, [5, 6, 7], [])
        out = self._classify(dbze, liq, ice)
        self.assertEqual(float(out.warm_rain[0]), 1.0)
        self.assertEqual(float(out.cold_rain[0]), 0.0)

    def test_cold_rain_ice_in_topmost_hit_layer(self):
        # The layer where the threshold is reached contains ice at another
        # level of the same contiguous layer -> cold.
        dbze, liq, ice = _column({6: 5.0}, [4, 5, 6], [4])
        out = self._classify(dbze, liq, ice)
        self.assertEqual(float(out.cold_rain[0]), 1.0)
        self.assertEqual(float(out.warm_rain[0]), 0.0)

    def test_warm_despite_separate_ice_layer_above(self):
        # Ice exists in a *separate* higher cloud layer where the threshold
        # is NOT reached: the highest layer reaching the threshold is liquid,
        # so the column is warm (this is the key difference from a simple
        # "ice anywhere" test).
        dbze, liq, ice = _column({7: 5.0, 2: -20.0}, [2, 6, 7], [2])
        out = self._classify(dbze, liq, ice)
        self.assertEqual(float(out.warm_rain[0]), 1.0)

    def test_cold_when_higher_icy_layer_also_hits(self):
        # Two separate layers hit the threshold; the higher one has ice.
        dbze, liq, ice = _column({7: 5.0, 2: 3.0}, [2, 6, 7], [2])
        out = self._classify(dbze, liq, ice)
        self.assertEqual(float(out.cold_rain[0]), 1.0)

    def test_drizzle_regime(self):
        # Max reflectivity between -15 and 0: drizzle, not rain.
        dbze, liq, ice = _column({6: -5.0}, [6], [])
        out = self._classify(dbze, liq, ice)
        self.assertEqual(float(out.warm_drizzle[0]), 1.0)
        self.assertEqual(float(out.warm_rain[0]), 0.0)

    def test_no_surface_liquid_precip_no_flags(self):
        dbze, liq, ice = _column({6: 5.0}, [6], [])
        out = self._classify(dbze, liq, ice, raining=False)
        self.assertEqual(sum(float(x[0]) for x in out), 0.0)

    def test_reflectivity_outside_cloud_ignored(self):
        # Threshold reached only at a cloud-free level (e.g. precip below
        # cloud base): the paper's layer search is over cloud layers.
        dbze, liq, ice = _column({9: 5.0}, [6], [])
        out = self._classify(dbze, liq, ice)
        self.assertEqual(sum(float(x[0]) for x in out), 0.0)

    def test_fraction(self):
        flags = WarmRainFlags(
            warm_rain=np.array([0.3]), cold_rain=np.array([0.1]),
            warm_drizzle=np.array([0.0]), cold_drizzle=np.array([0.2]))
        np.testing.assert_allclose(np.asarray(warm_rain_fraction(flags)), 0.75)
        np.testing.assert_allclose(
            np.asarray(warm_rain_fraction(flags, include_drizzle=True)), 0.5)
        empty = WarmRainFlags(*(np.zeros(1) for _ in range(4)))
        self.assertTrue(np.isnan(np.asarray(warm_rain_fraction(empty))).all())


if __name__ == "__main__":
    unittest.main()
