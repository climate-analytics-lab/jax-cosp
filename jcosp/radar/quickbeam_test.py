"""Tests for the quickbeam attenuation/dBZe port against the Fortran gate loop."""

import unittest

import numpy as np

from jcosp.constants import R_UNDEF
from jcosp.radar.quickbeam import quickbeam_subcolumn


def quickbeam_subcolumn_reference(z_vol, kr_vol, g_vol, hgt, radar_at_layer_one):
    """Loop transliteration of quickbeam.F90:quickbeam_subcolumn (one profile)."""
    ngate = len(z_vol)
    a_to_vol = np.zeros(ngate)
    g_to_vol = np.zeros(ngate)
    gates = range(ngate) if radar_at_layer_one else range(ngate - 1, -1, -1)
    d_gate = 1 if radar_at_layer_one else -1
    for k in gates:
        if d_gate == 1:
            if k > 1:  # Fortran k > 2 (1-based)
                a_to_vol[k] = a_to_vol[k - 1] + (
                    kr_vol[k - 1] + kr_vol[k]) * (hgt[k - 1] - hgt[k])
            else:
                a_to_vol[k] = kr_vol[k] * (hgt[k] - hgt[k + 1])
            if k > 0:
                g_to_vol[k] = g_to_vol[k - 1] + 0.5 * (
                    g_vol[k - 1] + g_vol[k]) * (hgt[k - 1] - hgt[k])
            else:
                g_to_vol[k] = 0.5 * g_vol[k] * (hgt[k] - hgt[k + 1])
        else:
            if k < ngate - 1:
                a_to_vol[k] = a_to_vol[k + 1] + (
                    kr_vol[k + 1] + kr_vol[k]) * (hgt[k + 1] - hgt[k])
                g_to_vol[k] = g_to_vol[k + 1] + 0.5 * (
                    g_vol[k + 1] + g_vol[k]) * (hgt[k + 1] - hgt[k])
            else:
                a_to_vol[k] = kr_vol[k] * (hgt[k] - hgt[k - 1])
                g_to_vol[k] = 0.5 * g_vol[k] * (hgt[k] - hgt[k - 1])
    dbze = np.where(z_vol > 0,
                    10 * np.log10(np.maximum(z_vol, 1e-300)) - a_to_vol - g_to_vol,
                    R_UNDEF)
    ze_non = np.where(z_vol > 0,
                      10 * np.log10(np.maximum(z_vol, 1e-300)), R_UNDEF)
    return dbze, ze_non


class QuickbeamSubcolumnTest(unittest.TestCase):
    def test_legacy_matches_fortran_transliteration(self):
        rng = np.random.default_rng(11)
        nlev = 15
        hgt = np.sort(rng.uniform(0.2, 16.0, nlev))[::-1].copy()  # km, TOA-first
        for radar_top in (True, False):
            z_vol = np.where(rng.uniform(size=nlev) > 0.4,
                             10 ** rng.uniform(-6, 4, nlev), 0.0)
            kr = np.where(z_vol > 0, rng.uniform(0, 2, nlev), 0.0)
            g = rng.uniform(0.1, 1.5, nlev)
            ref_dbze, ref_non = quickbeam_subcolumn_reference(
                z_vol, kr, g, hgt, radar_top)
            dbze, ze_non = quickbeam_subcolumn(
                z_vol, kr, g, hgt, radar_at_layer_one=radar_top,
                legacy_gate2=True)
            np.testing.assert_allclose(np.asarray(dbze), ref_dbze, rtol=1e-6,
                                       err_msg=f"radar_top={radar_top}")
            np.testing.assert_allclose(np.asarray(ze_non), ref_non, rtol=1e-6)

    def test_fixed_gate2_accumulates_and_differs_only_there(self):
        # FIXED UPSTREAM BUG: gate 2 (0-based index 1) of the spaceborne
        # hydrometeor sum must accumulate gate 1 with the trapezoid rule.
        # The correction propagates from index 1 downward but the *increment*
        # relative to the legacy path is set entirely at index 1.
        rng = np.random.default_rng(13)
        nlev = 12
        hgt = np.linspace(16.0, 0.4, nlev)
        z_vol = 10 ** rng.uniform(-4, 3, nlev)
        kr = rng.uniform(0.1, 2.0, nlev)
        g = np.zeros(nlev)
        dbze_fix, _ = quickbeam_subcolumn(z_vol, kr, g, hgt)
        dbze_leg, _ = quickbeam_subcolumn(z_vol, kr, g, hgt, legacy_gate2=True)
        diff = np.asarray(dbze_leg) - np.asarray(dbze_fix)
        # Gate 1 (index 0) identical; correction appears at index 1.
        self.assertAlmostEqual(diff[0], 0.0, places=10)
        expected_index1 = (kr[0] * (hgt[0] - hgt[1])          # a0
                           + (kr[0] + kr[1]) * (hgt[0] - hgt[1])
                           - kr[1] * (hgt[1] - hgt[2]))
        self.assertAlmostEqual(diff[1], expected_index1, places=8)
        # Downstream gates inherit exactly the same offset.
        np.testing.assert_allclose(diff[2:], diff[1], rtol=1e-8)

    def test_attenuation_only_reduces(self):
        nlev = 10
        hgt = np.linspace(15, 0.5, nlev)
        z_vol = np.full(nlev, 100.0)
        kr = np.full(nlev, 0.5)
        g = np.full(nlev, 0.2)
        dbze, ze_non = quickbeam_subcolumn(z_vol, kr, g, hgt)
        self.assertTrue((np.asarray(dbze) <= np.asarray(ze_non)).all())
        # Attenuation accumulates downward for a spaceborne radar.
        self.assertTrue(np.all(np.diff(np.asarray(dbze)) < 0))


if __name__ == "__main__":
    unittest.main()
