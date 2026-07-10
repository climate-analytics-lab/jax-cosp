"""Tests for the CFODD / warm-rain-occurrence port (COSP_DIAG_WARMRAIN).

The reference below transliterates the Fortran loop-for-loop on the same
TOA-first arrays the port consumes: the Fortran's ``k = Nlevels, 1, -1``
(surface upward over TOA-first data) becomes ``for k in range(nlev-1, -1,
-1)``, which *proves* the kcbtm/kctop index mapping (cloud top at the
smaller index). Constructed single-layer warm-cloud scenes pin down the
regime and Reff-class routing, the screening thresholds and the exclusion
of ice-topped/cold clouds.
"""

import unittest

import jax
import numpy as np

from jcosp import config
from jcosp.constants import R_GROUND, R_UNDEF, TMELT
from jcosp.modis import cfodd
from jcosp.stats_test import hist2d_reference


def diag_warmrain_reference(temp, zlev, lwp, liqcot, liqreff,
                            iwp, icecot, icereff, fracout, dbze):
    """Loop transliteration of COSP_DIAG_WARMRAIN (one gridpoint).

    ``temp``/``zlev`` are (nlev,); ``fracout``/``dbze`` are (nlev, ncol),
    TOA-first (index 0 = model top), matching the Fortran's k = 1 = top.
    """
    nlev, ncol = dbze.shape
    fr = np.zeros((nlev, ncol), int)
    fr[(fracout >= 0.5) & (fracout < 1.5)] = 1
    fr[(fracout >= 1.5) & (fracout < 2.5)] = 2

    if lwp == R_UNDEF:
        return (np.full((config.CFODD_NDBZE, config.CFODD_NICOD,
                         config.CFODD_NCLASS), R_UNDEF),
                np.full(config.WR_NREGIME, R_UNDEF))
    cfodd_n = np.zeros((config.CFODD_NDBZE, config.CFODD_NICOD,
                        config.CFODD_NCLASS))
    wr = np.zeros(config.WR_NREGIME)
    icod = np.zeros((nlev, ncol))

    br = config.CFODD_BNDRE
    if ((lwp <= config.CWP_THRESHOLD and lwp != R_UNDEF)
            or liqcot <= config.COT_THRESHOLD
            or liqreff < br[0] or liqreff > br[3]
            or iwp > config.CWP_THRESHOLD
            or icecot > config.COT_THRESHOLD
            or icereff > br[0]):
        return cfodd_n, wr
    if br[0] <= liqreff < br[1]:
        icls = 1
    elif br[1] <= liqreff < br[2]:
        icls = 2
    else:  # br[2] <= liqreff <= br[3], guaranteed by the screening
        icls = 3

    for j in range(ncol):
        octop, ocbtm = True, True
        kcbtm = kctop = 0
        for k in range(nlev - 1, -1, -1):  # Fortran k = Nlevels, 1, -1
            if dbze[k, j] == R_GROUND or dbze[k, j] == R_UNDEF:
                continue
            if ocbtm and fr[k, j] != 0 and dbze[k, j] >= config.CFODD_DBZE_MIN:
                ocbtm = False
                kcbtm = k
                kctop = k
            if (octop and not ocbtm and fr[k, j] != 0
                    and dbze[k, j] >= config.CFODD_DBZE_MIN):
                kctop = k
        if ocbtm:
            continue
        if temp[kctop] < TMELT:
            continue
        oslwc = True
        cmxdbz = config.CFODD_DBZE_MIN
        for k in range(kcbtm, kctop - 1, -1):
            cmxdbz = max(cmxdbz, dbze[k, j])
            if fr[k, j] in (config.SGCLD_CLR, config.SGCLD_CUM) \
                    or dbze[k, j] < config.CFODD_DBZE_MIN:
                oslwc = False
        if not oslwc:
            continue
        if cmxdbz < config.CFODD_BNDZE[0]:
            iregime = 1
        elif cmxdbz < config.CFODD_BNDZE[1]:
            iregime = 2
        else:
            iregime = 3
        wr[iregime - 1] += 1.0
        diagcgt = zlev[kctop] - zlev[kcbtm]
        cbtmh = zlev[kcbtm]
        for k in range(kcbtm, kctop - 1, -1):
            if k == kcbtm:
                diagicod = liqcot
            else:
                diagicod = liqcot * (1.0 - ((zlev[k] - cbtmh) / diagcgt)
                                     ** (5.0 / 3.0))
            icod[k, j] = min(diagicod, config.CFODD_ICOD_MAX)

    cfodd_n[:, :, icls - 1] = hist2d_reference(
        dbze.ravel(), icod.ravel(),
        config.CFODD_HIST_DBZE, config.CFODD_HIST_ICOD)
    return cfodd_n, wr


def _stat_grid(nlev=40):
    """Statistical-grid heights (vgrid_zu, TOA-first) and a warm profile."""
    zlev = config.VGRID_ZU.astype(float)
    temp = 288.0 - 0.0065 * np.clip(config.VGRID_Z, 0, 12000.0)
    return zlev, temp


def _single_layer_scene(nlev=40, ncol=4, ktop=35, kbtm=38, dbze_val=-20.0,
                        frac_val=1.0):
    """All-stratiform warm cloud between levels ktop..kbtm in every column."""
    dbze = np.full((nlev, ncol), -100.0)
    frac = np.zeros((nlev, ncol))
    dbze[ktop:kbtm + 1] = dbze_val
    frac[ktop:kbtm + 1] = frac_val
    return frac, dbze


class TransliterationTest(unittest.TestCase):
    def test_matches_fortran_on_random_scenes(self):
        rng = np.random.default_rng(30)
        nlev, ncol, nb = 12, 10, 8
        zlev = np.linspace(11000.0, 200.0, nlev)
        temp = rng.uniform(255.0, 295.0, (nlev, nb))
        frac = rng.integers(0, 3, (nlev, ncol, nb)).astype(float)
        dbze = rng.uniform(-45.0, 15.0, (nlev, ncol, nb))
        dbze[rng.uniform(size=dbze.shape) < 0.05] = R_GROUND
        dbze[rng.uniform(size=dbze.shape) < 0.05] = R_UNDEF
        # MODIS retrievals spanning pass/fail screening combinations.
        lwp = rng.uniform(0.0, 0.2, nb)
        lwp[0] = R_UNDEF
        liqcot = rng.uniform(0.0, 30.0, nb)
        liqreff = rng.uniform(2e-6, 40e-6, nb)
        iwp = rng.choice([R_UNDEF, 0.0, 0.05], nb)
        icecot = rng.choice([R_UNDEF, 0.0, 1.0], nb)
        icereff = rng.choice([R_UNDEF, 0.0, 2.0e-5], nb)

        out = cfodd.diag_warmrain(temp, zlev, lwp, liqcot, liqreff,
                                  iwp, icecot, icereff, frac, dbze)
        for b in range(nb):
            ref_cf, ref_wr = diag_warmrain_reference(
                temp[:, b], zlev, lwp[b], liqcot[b], liqreff[b],
                iwp[b], icecot[b], icereff[b], frac[..., b], dbze[..., b])
            np.testing.assert_allclose(
                np.asarray(out.cfodd_ntotal)[..., b], ref_cf,
                err_msg=f"cfodd batch {b}")
            np.testing.assert_allclose(
                np.asarray(out.wr_occfreq_ntotal)[..., b], ref_wr,
                err_msg=f"wr batch {b}")


class ConstructedSceneTest(unittest.TestCase):
    def _run(self, frac, dbze, liqreff=15e-6, liqcot=10.0, lwp=0.08,
             iwp=R_UNDEF, icecot=R_UNDEF, icereff=R_UNDEF, temp=None):
        zlev, temp_def = _stat_grid(dbze.shape[0])
        t = temp_def if temp is None else temp
        return cfodd.diag_warmrain(
            np.asarray(t, float), zlev,
            np.asarray(lwp), np.asarray(liqcot), np.asarray(liqreff),
            np.asarray(iwp), np.asarray(icecot), np.asarray(icereff),
            frac, dbze)

    def test_nonprecip_drizzle_rain_regimes(self):
        for dbz, regime in ((-25.0, 0), (-10.0, 1), (5.0, 2)):
            frac, dbze = _single_layer_scene(dbze_val=dbz)
            out = self._run(frac, dbze)
            wr = np.asarray(out.wr_occfreq_ntotal)
            ncol = dbze.shape[1]
            self.assertEqual(wr[regime], ncol, f"dbz={dbz}")
            self.assertEqual(wr.sum(), ncol, f"dbz={dbz}")

    def test_reff_class_routing(self):
        for reff, icls in ((8e-6, 0), (15e-6, 1), (25e-6, 2)):
            frac, dbze = _single_layer_scene()
            out = self._run(frac, dbze, liqreff=reff)
            cf = np.asarray(out.cfodd_ntotal)
            self.assertGreater(cf[..., icls].sum(), 0.0, f"reff={reff}")
            for other in set(range(3)) - {icls}:
                self.assertEqual(cf[..., other].sum(), 0.0, f"reff={reff}")

    def test_icod_profile_bounded_by_cot(self):
        frac, dbze = _single_layer_scene(dbze_val=-10.0)
        out = self._run(frac, dbze, liqcot=12.0)
        cf = np.asarray(out.cfodd_ntotal)[..., 1]  # class 2 for 15 um
        # Samples appear only in ICOD bins up to the cloud optical depth.
        icod_max_bin = int(12.0 // config.CFODD_ICOD_WIDTH)
        self.assertGreater(cf[:, :icod_max_bin + 1].sum(), 0.0)
        self.assertEqual(cf[:, icod_max_bin + 1:].sum(), 0.0)

    def test_cold_topped_cloud_excluded(self):
        # Same geometry, but the cloud-top temperature is below freezing.
        frac, dbze = _single_layer_scene(dbze_val=-10.0)
        zlev, temp = _stat_grid()
        temp = np.full_like(temp, 250.0)
        out = self._run(frac, dbze, temp=temp)
        self.assertEqual(np.asarray(out.wr_occfreq_ntotal).sum(), 0.0)

    def test_convective_cells_disqualify_slwc(self):
        frac, dbze = _single_layer_scene(dbze_val=-10.0, frac_val=2.0)
        out = self._run(frac, dbze)
        self.assertEqual(np.asarray(out.wr_occfreq_ntotal).sum(), 0.0)

    def test_modis_screening_thresholds(self):
        frac, dbze = _single_layer_scene(dbze_val=-10.0)
        for kwargs in (dict(liqcot=0.2),        # COT below threshold
                       dict(lwp=0.0),           # no liquid water path
                       dict(liqreff=3e-6),      # Reff below class range
                       dict(liqreff=40e-6),     # Reff above class range
                       dict(iwp=0.01),          # ice contamination
                       dict(icecot=1.0),
                       dict(icereff=1e-5)):
            out = self._run(frac, dbze, **kwargs)
            self.assertEqual(np.asarray(out.cfodd_ntotal).sum(), 0.0,
                             f"{kwargs}")
            self.assertEqual(np.asarray(out.wr_occfreq_ntotal).sum(), 0.0,
                             f"{kwargs}")

    def test_dark_point_undefined(self):
        frac, dbze = _single_layer_scene()
        out = self._run(frac, dbze, lwp=R_UNDEF)
        self.assertTrue((np.asarray(out.cfodd_ntotal) == R_UNDEF).all())
        self.assertTrue((np.asarray(out.wr_occfreq_ntotal) == R_UNDEF).all())

    def test_jit_matches_eager(self):
        frac, dbze = _single_layer_scene(dbze_val=-10.0)
        zlev, temp = _stat_grid()
        args = (temp, zlev, np.asarray(0.08), np.asarray(10.0),
                np.asarray(15e-6), np.asarray(R_UNDEF), np.asarray(R_UNDEF),
                np.asarray(R_UNDEF), frac, dbze)
        eager = cfodd.diag_warmrain(*args)
        jitted = jax.jit(cfodd.diag_warmrain)(*args)
        np.testing.assert_allclose(np.asarray(eager.cfodd_ntotal),
                                   np.asarray(jitted.cfodd_ntotal))
        np.testing.assert_allclose(np.asarray(eager.wr_occfreq_ntotal),
                                   np.asarray(jitted.wr_occfreq_ntotal))


if __name__ == "__main__":
    unittest.main()
