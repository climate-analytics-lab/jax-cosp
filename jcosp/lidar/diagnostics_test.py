"""Tests for the CALIPSO column-stage diagnostics.

``cldfrac`` is checked against a loop-for-loop NumPy transliteration of
``COSP_CLDFRAC`` (lidar_simulator.F90), including the phase classification,
the full-attenuation (undefined phase) scan and the temperature-binned phase
fractions; ``cfad_sr`` against a direct histogram; ``lidar_only_cloud``
against a transliteration of ``COSP_LIDAR_ONLY_CLOUD``; plus an end-to-end
``simulate_calipso`` scene test.

The reference implements the Fortran verbatim except where noted: the
``point0_pressure_bug`` flag reproduces the Fortran's use of gridpoint 1's
pressure in the high-altitude section (4.1); the port always uses the
per-point pressure (documented deviation), so the multi-point comparison runs
with the flag off and a single-point comparison demonstrates exact
equivalence with the unmodified Fortran logic.
"""

import unittest

import jax.numpy as jnp
import numpy as np

from jcosp import config
from jcosp.constants import R_GROUND, R_UNDEF
from jcosp.lidar import CalipsoInputs, simulate_calipso
from jcosp.lidar.diagnostics import cfad_sr, cldfrac, lidar_only_cloud
from jcosp.lidar.optics_test import make_atmosphere

S_ATT, S_CLD, S_CLD_ATT = 0.01, 5.0, 30.0


def ref_cldfrac(tmp, x, atb, atbperp, pplay, point0_pressure_bug=False):
    """Transliteration of COSP_CLDFRAC; Fortran layout (npoints, ncol, nlev)."""
    undef = R_UNDEF
    tempmod = np.asarray(config.LIDAR_PHASE_TEMP_EDGES)
    a50, b50, g50, d50, e50, z50 = config.CALIPSO_PHASE_DISCR_COEFFS
    npoints, ncol, nlev = x.shape

    cldy = np.where((x > S_CLD) & (x != undef), 1.0, 0.0)
    srok = np.where((x > S_ATT) & (x != undef), 1.0, 0.0)

    lidarcld = np.zeros((npoints, nlev))
    nsub = np.zeros((npoints, nlev))
    cldlay = np.zeros((npoints, ncol, 4))
    nsublay = np.zeros((npoints, ncol, 4))
    lidarcldtemp = np.zeros((npoints, 40, 5))
    lidarcldtempind = np.zeros((npoints, 40))
    lidarcldphase = np.zeros((npoints, nlev, 6))
    cldlayphase = np.zeros((npoints, ncol, 4, 6))
    tmpi = np.zeros((npoints, ncol, nlev))
    tmpl = np.zeros_like(tmpi)
    tmpu = np.zeros_like(tmpi)

    # Section 3: 3D cloud fraction, layered covers, temperature indicator.
    for k in range(nlev):
        for ic in range(ncol):
            for ip in range(npoints):
                if srok[ip, ic, k] > 0:
                    for it in range(40):
                        if tempmod[it] <= tmp[ip, k] < tempmod[it + 1]:
                            lidarcldtempind[ip, it] += 1
                if cldy[ip, ic, k] == 1:
                    for it in range(40):
                        if tempmod[it] <= tmp[ip, k] < tempmod[it + 1]:
                            lidarcldtemp[ip, it, 0] += 1
                iz = 0
                p1 = pplay[ip, k]
                if 0.0 < p1 < 440.0e2:
                    iz = 2
                elif 440.0e2 <= p1 < 680.0e2:
                    iz = 1
                cldlay[ip, ic, iz] = max(cldlay[ip, ic, iz], cldy[ip, ic, k])
                cldlay[ip, ic, 3] = max(cldlay[ip, ic, 3], cldy[ip, ic, k])
                lidarcld[ip, k] += cldy[ip, ic, k]
                nsublay[ip, ic, iz] = max(nsublay[ip, ic, iz], srok[ip, ic, k])
                nsublay[ip, ic, 3] = max(nsublay[ip, ic, 3], srok[ip, ic, k])
                nsub[ip, k] += srok[ip, ic, k]

    lidarcld = np.where(nsub > 0, lidarcld / np.where(nsub > 0, nsub, 1), undef)
    cldlayer = cldlay.sum(axis=1)
    nsublayer = nsublay.sum(axis=1)
    cldlayer = np.where(nsublayer > 0,
                        cldlayer / np.where(nsublayer > 0, nsublayer, 1), undef)

    def cat(p1):
        if 0.0 < p1 < 440.0e2:
            return 2
        if 440.0e2 <= p1 < 680.0e2:
            return 1
        return 0

    def classify(ip, ic, k, p1):
        v = atb[ip, ic, k]
        line = (v**5 * a50 + v**4 * b50 + v**3 * g50 + v**2 * d50
                + v * e50 + z50)
        iz = cat(p1)
        if atbperp[ip, ic, k] - line >= 0.0:  # ice side of the line
            if tmp[ip, k] > 273.15:           # false ice -> liquid
                lidarcldphase[ip, k, 1] += 1
                tmpl[ip, ic, k] = tmp[ip, k]
                lidarcldphase[ip, k, 4] += 1
                cldlayphase[ip, ic, 3, 1] = 1
                cldlayphase[ip, ic, iz, 1] = 1
                cldlayphase[ip, ic, 3, 4] = 1
                cldlayphase[ip, ic, iz, 4] = 1
            else:
                lidarcldphase[ip, k, 0] += 1
                tmpi[ip, ic, k] = tmp[ip, k]
                cldlayphase[ip, ic, 3, 0] = 1
                cldlayphase[ip, ic, iz, 0] = 1
        else:                                 # liquid side
            if tmp[ip, k] > 231.15:
                lidarcldphase[ip, k, 1] += 1
                tmpl[ip, ic, k] = tmp[ip, k]
                cldlayphase[ip, ic, 3, 1] = 1
                cldlayphase[ip, ic, iz, 1] = 1
            else:                             # false liquid -> ice
                tmpi[ip, ic, k] = tmp[ip, k]
                lidarcldphase[ip, k, 0] += 1
                lidarcldphase[ip, k, 3] += 1
                cldlayphase[ip, ic, 3, 3] = 1
                cldlayphase[ip, ic, iz, 3] = 1
                cldlayphase[ip, ic, 3, 0] = 1
                cldlayphase[ip, ic, iz, 0] = 1

    # Sections 4.1/4.2: phase classification, full-attenuation scan.
    for ic in range(ncol):
        for ip in range(npoints):
            for k in range(min(23, nlev)):
                if cldy[ip, ic, k] == 1 and atbperp[ip, ic, k] > 0:
                    p1 = pplay[0, k] if point0_pressure_bug else pplay[ip, k]
                    classify(ip, ic, k, p1)
            toplvlsat = 0
            for k in range(23, nlev):
                if cldy[ip, ic, k] == 1 and atbperp[ip, ic, k] > 0:
                    classify(ip, ic, k, pplay[ip, k])
                    if x[ip, ic, k] > S_CLD_ATT:
                        toplvlsat = k + 1
                        break
            if toplvlsat != 0:
                for k in range(toplvlsat, nlev):
                    if cldy[ip, ic, k] == 1:
                        lidarcldphase[ip, k, 2] += 1
                        tmpu[ip, ic, k] = tmp[ip, k]
                        cldlayphase[ip, ic, 3, 2] = 1
                        cldlayphase[ip, ic, cat(pplay[ip, k]), 2] = 1

    # Final phase normalizations.
    tsum = lidarcldphase[:, :, 0] + lidarcldphase[:, :, 1]
    lidarcldphase[:, :, 5] = np.where(
        tsum > 0, lidarcldphase[:, :, 0] / np.where(tsum > 0, tsum, 1), undef)
    for i in range(5):
        lidarcldphase[:, :, i] = np.where(
            nsub > 0, lidarcldphase[:, :, i] / np.where(nsub > 0, nsub, 1),
            undef)

    cldlayerphase = cldlayphase.sum(axis=1)          # (npoints, 4, 6)
    phasesum = cldlayerphase[:, :, 0:3].sum(axis=2)
    t2 = cldlayerphase[:, :, 0] + cldlayerphase[:, :, 1]
    cldlayerphase[:, :, 5] = np.where(
        t2 > 0, cldlayerphase[:, :, 0] / np.where(t2 > 0, t2, 1), undef)
    for i in range(5):
        m = phasesum > 0
        cldlayerphase[:, :, i] = np.where(
            m, cldlayerphase[:, :, i] / np.where(m, phasesum, 1) * cldlayer,
            cldlayerphase[:, :, i])
    for i in range(5):
        cldlayerphase[:, :, i] = np.where(nsublayer == 0, undef,
                                          cldlayerphase[:, :, i])

    # Phase fractions vs temperature (tmpi/tmpl/tmpu priority chain).
    for k in range(nlev):
        for ic in range(ncol):
            for ip in range(npoints):
                for it in range(40):
                    if tmpi[ip, ic, k] > 0:
                        if tempmod[it] <= tmpi[ip, ic, k] < tempmod[it + 1]:
                            lidarcldtemp[ip, it, 1] += 1
                    elif tmpl[ip, ic, k] > 0:
                        if tempmod[it] <= tmpl[ip, ic, k] < tempmod[it + 1]:
                            lidarcldtemp[ip, it, 2] += 1
                    elif tmpu[ip, ic, k] > 0:
                        if tempmod[it] <= tmpu[ip, ic, k] < tempmod[it + 1]:
                            lidarcldtemp[ip, it, 3] += 1
    sumt = lidarcldtemp[:, :, 1] + lidarcldtemp[:, :, 2]
    lidarcldtemp[:, :, 4] = np.where(
        sumt > 0, lidarcldtemp[:, :, 1] / np.where(sumt > 0, sumt, 1), undef)
    for i in range(4):
        lidarcldtemp[:, :, i] = np.where(
            lidarcldtempind > 0,
            lidarcldtemp[:, :, i] / np.where(lidarcldtempind > 0,
                                             lidarcldtempind, 1), undef)

    return dict(lidarcld=lidarcld.T, cldlayer=cldlayer.T,
                lidarcldphase=np.transpose(lidarcldphase, (1, 2, 0)),
                cldlayerphase=np.transpose(cldlayerphase, (1, 2, 0)),
                lidarcldtemp=np.transpose(lidarcldtemp, (1, 2, 0)))


def ref_lidar_only(beta_tot, beta_mol, ze_tot):
    """Transliteration of COSP_LIDAR_ONLY_CLOUD; layout (npoints, ncol, nlev)."""
    npoints, ncol, nlev = beta_tot.shape
    freq = np.zeros((npoints, nlev))
    tcc = np.zeros(npoints)
    rtcc = np.zeros(npoints)
    rtcc2 = np.zeros(npoints)
    for pr in range(npoints):
        for i in range(ncol):
            flag_sat = -1
            flag_cld = 0
            flag_radarcld = 0
            flag_no1km = 0
            j = 0
            while ze_tot[pr, i, j] == R_GROUND:
                j += 1
            j_first = j
            for j in range(nlev):
                sc = beta_tot[pr, i, j] / beta_mol[pr, j]
                if sc <= S_ATT and flag_sat < 0:
                    flag_sat = j
                if ze_tot[pr, i, j] < -30.0:
                    if sc > S_CLD or flag_sat == j:
                        freq[pr, j] += 1
                        flag_cld = 1
                else:
                    flag_cld = 1
                    flag_radarcld = 1
                    if j > j_first + 1:
                        flag_no1km = 1
            tcc[pr] += flag_cld
            rtcc[pr] += flag_radarcld
            rtcc2[pr] += flag_no1km
    return (freq.T / ncol * 100.0, tcc / ncol * 100.0,
            rtcc / ncol * 100.0, rtcc2 / ncol * 100.0)


def random_column_scene(rng, npoints=2, ncol=6, nlev=40):
    """Random SR/ATB fields exercising every threshold branch."""
    tmp = (np.linspace(200.0, 288.0, nlev)[None]
           + rng.uniform(-25.0, 10.0, (npoints, nlev)))
    pplay = (np.linspace(60.0e2, 1010.0e2, nlev)[None]
             * rng.uniform(0.9, 1.1, (npoints, 1)))
    atb = rng.uniform(1e-7, 6e-3, (npoints, ncol, nlev))
    c = config.CALIPSO_PHASE_DISCR_COEFFS
    line = (atb**5 * c[0] + atb**4 * c[1] + atb**3 * c[2] + atb**2 * c[3]
            + atb * c[4] + c[5])
    atbperp = line * rng.uniform(0.3, 2.0, atb.shape)
    atbperp[rng.uniform(size=atb.shape) < 0.15] = 0.0
    # SR spanning attenuated / clear / cloudy / saturated, plus R_UNDEF.
    x = rng.uniform(0.0, 60.0, atb.shape)
    x[rng.uniform(size=x.shape) < 0.1] = R_UNDEF
    x[0, :, -2:] = R_UNDEF  # below-ground levels of point 0
    return tmp, x, atb, atbperp, pplay


class CldfracTest(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(2026)

    def _compare(self, tmp, x, atb, atbperp, pplay, **ref_kwargs):
        ref = ref_cldfrac(tmp, x, atb, atbperp, pplay, **ref_kwargs)
        out = cldfrac(jnp.asarray(tmp.T),
                      jnp.asarray(np.transpose(x, (2, 1, 0))),
                      jnp.asarray(np.transpose(atb, (2, 1, 0))),
                      jnp.asarray(np.transpose(atbperp, (2, 1, 0))),
                      jnp.asarray(pplay.T))
        for name, got in (("lidarcld", out.lidarcld),
                          ("cldlayer", out.cldlayer),
                          ("lidarcldphase", out.lidarcldphase),
                          ("cldlayerphase", out.cldlayerphase),
                          ("lidarcldtemp", out.lidarcldtemp)):
            np.testing.assert_allclose(np.asarray(got), ref[name],
                                       rtol=1e-12, atol=0, err_msg=name)

    def test_matches_fortran_transliteration(self):
        for _ in range(3):
            self._compare(*random_column_scene(self.rng))

    def test_single_point_matches_unmodified_fortran(self):
        # With one gridpoint the Fortran's pplay(1,:) slip is invisible, so
        # the port matches the verbatim Fortran logic exactly.
        tmp, x, atb, atbperp, pplay = random_column_scene(self.rng, npoints=1)
        self._compare(tmp, x, atb, atbperp, pplay, point0_pressure_bug=True)

    def test_undefined_phase_below_saturation(self):
        # One column: saturating cloud (SR > 30) at level 25, another cloud
        # at level 30 -> the lower cloud's phase is undefined; the saturating
        # level itself is still classified.
        nlev, ncol, npoints = 40, 1, 1
        tmp = np.full((npoints, nlev), 260.0)
        pplay = np.linspace(60.0e2, 1000.0e2, nlev)[None]
        x = np.ones((npoints, ncol, nlev))
        atb = np.full((npoints, ncol, nlev), 1e-4)
        atbperp = np.full((npoints, ncol, nlev), 1e-4)  # well above the line
        x[0, 0, 25] = 40.0   # saturating cloud
        x[0, 0, 30] = 10.0   # cloud below it
        x[0, 0, 20] = 10.0   # independent high cloud, above 8.16 km
        out = cldfrac(jnp.asarray(tmp.T),
                      jnp.asarray(np.transpose(x, (2, 1, 0))),
                      jnp.asarray(np.transpose(atb, (2, 1, 0))),
                      jnp.asarray(np.transpose(atbperp, (2, 1, 0))),
                      jnp.asarray(pplay.T))
        phase = np.asarray(out.lidarcldphase)[:, :, 0]
        self.assertEqual(phase[25, 0], 1.0)  # ice (cold, above the line)
        self.assertEqual(phase[20, 0], 1.0)
        self.assertEqual(phase[30, 2], 1.0)  # undefined below saturation
        self.assertEqual(phase[30, 0], 0.0)
        # And the transliteration agrees on this constructed scene.
        self._compare(tmp, x, atb, atbperp, pplay)


class CfadTest(unittest.TestCase):
    def test_cfad_matches_histogram(self):
        rng = np.random.default_rng(5)
        _, x, _, _, _ = random_column_scene(rng)
        sr = jnp.asarray(np.transpose(x, (2, 1, 0)))  # (nlev, ncol, npts)
        out = np.asarray(cfad_sr(sr))
        bins = config.CALIPSO_HIST_BSCT
        nlev, ncol, npts = sr.shape
        for ip in range(npts):
            for k in range(nlev):
                want, _ = np.histogram(x[ip, :, k], bins=bins)
                # np.histogram closes the last bin; COSP's is half-open, but
                # no sample sits at exactly 999 here.
                np.testing.assert_allclose(out[:, k, ip], want / ncol,
                                           err_msg=f"ip={ip} k={k}")

    def test_cfad_sums_to_defined_fraction(self):
        rng = np.random.default_rng(6)
        _, x, _, _, _ = random_column_scene(rng)
        sr = jnp.asarray(np.transpose(x, (2, 1, 0)))
        out = np.asarray(cfad_sr(sr))
        ncol = sr.shape[1]
        counted = np.sum(np.transpose(x, (2, 1, 0)) != R_UNDEF, axis=1) / ncol
        np.testing.assert_allclose(out.sum(axis=0), counted)


class LidarOnlyCloudTest(unittest.TestCase):
    def test_matches_fortran_transliteration(self):
        rng = np.random.default_rng(77)
        npoints, ncol, nlev = 3, 5, 40
        beta_mol = rng.uniform(1e-7, 1e-5, (npoints, nlev))
        # Attenuated total signal: sometimes way below molecular (attenuated),
        # sometimes way above (cloud).
        factor = 10.0 ** rng.uniform(-3, 2, (npoints, ncol, nlev))
        beta_tot = beta_mol[:, None, :] * factor
        dbze = rng.uniform(-60.0, 10.0, (npoints, ncol, nlev))
        dbze[rng.uniform(size=dbze.shape) < 0.2] = R_UNDEF
        dbze[:, :, -3:] = R_GROUND  # below-ground gates at the profile bottom
        ref = ref_lidar_only(beta_tot, beta_mol, dbze)
        out = lidar_only_cloud(jnp.asarray(np.transpose(beta_tot, (2, 1, 0))),
                               jnp.asarray(beta_mol.T),
                               jnp.asarray(np.transpose(dbze, (2, 1, 0))))
        np.testing.assert_allclose(np.asarray(out.lidar_only_freq_cloud),
                                   ref[0], rtol=1e-12)
        np.testing.assert_allclose(np.asarray(out.radar_lidar_tcc), ref[1])
        np.testing.assert_allclose(np.asarray(out.cloudsat_tcc), ref[2])
        np.testing.assert_allclose(np.asarray(out.cloudsat_tcc2), ref[3])


def _model_heights(pres, presf, temp):
    """Hydrostatic heights consistent with the optics (TOA-first, m)."""
    rhoair = pres / (config.LIDAR_RD * temp)
    nlev = pres.shape[1]
    zh = np.zeros((pres.shape[0], nlev + 1))
    for k in range(nlev - 1, -1, -1):
        zh[:, k] = zh[:, k + 1] + (presf[:, k + 1] - presf[:, k]) / (
            rhoair[:, k] * config.LIDAR_GRAV)
    zhalf = zh[:, 1:]
    zfull = 0.5 * (zh[:, :-1] + zh[:, 1:])
    return zfull, zhalf


class SimulateCalipsoTest(unittest.TestCase):
    def _state(self, q_low_cloud=0.0, nlev=30, ncol=4, npoints=2):
        rng = np.random.default_rng(31)
        pres, presf, temp = make_atmosphere(rng, nlev, npoints)
        zfull, zhalf = _model_heights(pres, presf, temp)
        q = np.zeros((nlev, ncol, npoints))
        if q_low_cloud:
            # A warm liquid cloud in the lowest ~2-3 km (present in every
            # subcolumn), thick enough to saturate the signal beneath it.
            cloud_lev = np.argmin(np.abs(zfull[0] - 2500.0))
            q[cloud_lev] = q_low_cloud
        zeros = np.zeros_like(q)
        rad = np.full((nlev, npoints), 10e-6)
        return CalipsoInputs(
            pressure=jnp.asarray(pres.T), pressure_half=jnp.asarray(presf.T),
            temperature=jnp.asarray(temp.T), zfull=jnp.asarray(zfull.T),
            zhalf=jnp.asarray(zhalf.T), q_lsliq=jnp.asarray(q),
            q_lsice=jnp.asarray(zeros), q_cvliq=jnp.asarray(zeros),
            q_cvice=jnp.asarray(zeros), reff_lsliq=jnp.asarray(rad),
            reff_lsice=jnp.asarray(rad), reff_cvliq=jnp.asarray(rad),
            reff_cvice=jnp.asarray(rad))

    def test_clear_sky(self):
        out = simulate_calipso(self._state())
        lidarcld = np.asarray(out.lidarcld)
        defined = lidarcld != R_UNDEF
        self.assertTrue(defined.any())
        np.testing.assert_array_equal(lidarcld[defined], 0.0)
        cldlayer = np.asarray(out.cldlayer)
        np.testing.assert_array_equal(cldlayer[cldlayer != R_UNDEF], 0.0)
        # All defined SR mass sits in the SR ~ 1 bin ([0.01, 1.2)).
        cfad = np.asarray(out.cfad_sr)
        sr = np.asarray(out.sr)
        for k in range(sr.shape[0]):
            if np.all(sr[k] != R_UNDEF):
                np.testing.assert_allclose(cfad[1, k], 1.0)
                np.testing.assert_allclose(sr[k], 1.0, rtol=1e-9)

    def test_warm_liquid_cloud_scene(self):
        out = simulate_calipso(self._state(q_low_cloud=2e-3))
        cldlayer = np.asarray(out.cldlayer)
        np.testing.assert_allclose(cldlayer[0], 100.0)  # low cloud cover
        np.testing.assert_allclose(cldlayer[3], 100.0)  # total
        np.testing.assert_allclose(cldlayer[2], 0.0)    # no high cloud
        # The cloud is warm (> 273 K): classified liquid, never ice.
        phase = np.asarray(out.lidarcldphase)
        liq = phase[:, 1]
        ice = phase[:, 0]
        self.assertGreater(liq[liq != R_UNDEF].max(), 0.0)
        np.testing.assert_array_equal(ice[ice != R_UNDEF], 0.0)
        # Phase-vs-temperature: liquid occurrences only, at warm bins.
        tphase = np.asarray(out.lidarcldtmp)
        liq_t = tphase[:, 2]
        self.assertGreater(liq_t[liq_t != R_UNDEF].max(), 0.0)
        # ATB outputs finite everywhere.
        self.assertTrue(np.all(np.isfinite(np.asarray(out.beta_tot))))
        self.assertTrue(np.all(np.isfinite(np.asarray(out.betaperp_tot))))

    def test_lidar_only_product(self):
        # A radar that sees nothing: every lidar cloud is lidar-only.
        state = self._state(q_low_cloud=2e-3)
        nlev, ncol = state.q_lsliq.shape[:2]
        npoints = state.q_lsliq.shape[2]
        dbze = jnp.full((nlev, ncol, npoints), -100.0)
        out = simulate_calipso(state, dbze=dbze)
        self.assertIsNotNone(out.lidar_only)
        freq = np.asarray(out.lidar_only.lidar_only_freq_cloud)
        self.assertGreater(freq.max(), 0.0)
        np.testing.assert_allclose(
            np.asarray(out.lidar_only.radar_lidar_tcc), 100.0)
        np.testing.assert_allclose(np.asarray(out.lidar_only.cloudsat_tcc), 0.0)

    def test_no_dbze_no_lidar_only(self):
        out = simulate_calipso(self._state())
        self.assertIsNone(out.lidar_only)


if __name__ == "__main__":
    unittest.main()
