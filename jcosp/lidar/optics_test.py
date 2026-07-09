"""Tests for the CALIPSO lidar optics port.

The vectorized JAX implementation is checked against a loop-for-loop NumPy
transliteration of ``cosp_optics.F90:lidar_optics`` (spaceborne branch),
plus physical sanity checks on the molecular profile and a gradient check.
"""

import unittest

import jax
import jax.numpy as jnp
import numpy as np

from jcosp import config
from jcosp.lidar.optics import lidar_optics


def ref_lidar_optics(q, rad, pres, presf, temp, ice_type):
    """Transliteration of lidar_optics (532 nm, spaceborne).

    Fortran array layout: q[npoints, ncol, nlev, 4], rad[npoints, nlev, 4],
    pres/temp[npoints, nlev], presf[npoints, nlev+1].
    """
    npoints, ncol, nlev, npart = q.shape
    cmol, rdiffm, qscat = 6.2446e-32, 0.7, 2.0
    grav, rd, km = 9.80665, 287.04, 1.38e-23
    rhopart = [1000.0, 0.5e3, 1000.0, 0.5e3]
    polliq = [2.6980e-8, -3.7701e-6, 1.6594e-4, -0.0024, 0.0626]
    police = ([-1.0176e-8, 1.7615e-6, -1.0480e-4, 0.0019, 0.0460] if ice_type == 0
              else [1.3615e-8, -2.04206e-6, 7.51799e-5, 0.00078213, 0.0182131])
    polpart = [polliq, police, polliq, police]

    rad = np.clip(rad, 0.0, 70.0e-6)
    rhoair = pres / (rd * temp)

    zheight = np.zeros((npoints, nlev + 1))
    for k in range(nlev - 1, -1, -1):
        zheight[:, k] = zheight[:, k + 1] - (presf[:, k] - presf[:, k + 1]) / (
            rhoair[:, k] * grav)

    beta_mol = pres / km / temp * cmol
    alpha_mol = 8.0 * np.pi / 3.0 * beta_mol
    tau_mol = alpha_mol * (zheight[:, :nlev] - zheight[:, 1:nlev + 1])
    for k in range(1, nlev):
        tau_mol[:, k] = tau_mol[:, k] + tau_mol[:, k - 1]

    kp = np.zeros((npoints, nlev, npart))
    for i in range(npart):
        for j in range(nlev):
            for p in range(npoints):
                if rad[p, j, i] > 0.0:
                    x = rad[p, j, i] * 1.0e6
                    pp = polpart[i]
                    kp[p, j, i] = (pp[0] * x**4 + pp[1] * x**3 + pp[2] * x**2
                                   + pp[3] * x + pp[4])

    betatot = np.tile(beta_mol[:, None, :], (1, ncol, 1))
    tautot = np.tile(tau_mol[:, None, :], (1, ncol, 1))
    betatot_ice = betatot.copy()
    betatot_liq = betatot.copy()
    tautot_ice = tautot.copy()
    tautot_liq = tautot.copy()
    tau_sfc_liq = np.zeros((npoints, ncol))
    tau_sfc_ice = np.zeros((npoints, ncol))

    for icol in range(ncol):
        alpha_part = np.zeros((npoints, nlev, npart))
        for i in range(npart):
            mask = rad[:, :, i] > 0.0
            alpha_part[:, :, i] = np.where(
                mask,
                0.75 * qscat * rhoair * q[:, icol, :, i]
                / (rhopart[i] * np.where(mask, rad[:, :, i], 1.0)), 0.0)
        tau_part = rdiffm * alpha_part
        for i in range(npart):
            tau_part[:, :, i] *= zheight[:, :nlev] - zheight[:, 1:nlev + 1]
            for k in range(1, nlev):
                tau_part[:, k, i] += tau_part[:, k - 1, i]
        for i in range(npart):
            betatot[:, icol, :] += kp[:, :, i] * alpha_part[:, :, i]
            tautot[:, icol, :] += tau_part[:, :, i]
        betatot_ice[:, icol, :] += (kp[:, :, 1] * alpha_part[:, :, 1]
                                    + kp[:, :, 3] * alpha_part[:, :, 3])
        tautot_ice[:, icol, :] += tau_part[:, :, 1] + tau_part[:, :, 3]
        betatot_liq[:, icol, :] += (kp[:, :, 0] * alpha_part[:, :, 0]
                                    + kp[:, :, 2] * alpha_part[:, :, 2])
        tautot_liq[:, icol, :] += tau_part[:, :, 0] + tau_part[:, :, 2]
        tau_sfc_liq[:, icol] = tau_part[:, -1, 0] + tau_part[:, -1, 2]
        tau_sfc_ice[:, icol] = tau_part[:, -1, 1] + tau_part[:, -1, 3]

    return dict(beta_mol=beta_mol, tau_mol=tau_mol, betatot=betatot,
                tautot=tautot, betatot_ice=betatot_ice, tautot_ice=tautot_ice,
                betatot_liq=betatot_liq, tautot_liq=tautot_liq,
                tau_sfc_liq=tau_sfc_liq, tau_sfc_ice=tau_sfc_ice)


def make_atmosphere(rng, nlev=18, npoints=3):
    """Random plausible TOA-first pressure/temperature profiles."""
    presf = np.linspace(30.0e2, 1000.0e2, nlev + 1)[None] * rng.uniform(
        0.97, 1.03, (npoints, 1))
    presf.sort(axis=1)
    pres = 0.5 * (presf[:, :-1] + presf[:, 1:])
    temp = np.linspace(215.0, 288.0, nlev)[None] + rng.uniform(
        -5.0, 5.0, (npoints, nlev))
    return pres, presf, temp


class LidarOpticsTest(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(4242)

    def _random_inputs(self, nlev=18, ncol=5, npoints=3):
        rng = self.rng
        pres, presf, temp = make_atmosphere(rng, nlev, npoints)
        # Sparse cloud field with occasional zero radii (disabled volumes).
        q = np.where(rng.uniform(size=(npoints, ncol, nlev, 4)) < 0.3,
                     rng.uniform(1e-6, 2e-3, (npoints, ncol, nlev, 4)), 0.0)
        rad = np.where(rng.uniform(size=(npoints, nlev, 4)) < 0.8,
                       rng.uniform(2e-6, 90e-6, (npoints, nlev, 4)), 0.0)
        return q, rad, pres, presf, temp

    def test_matches_fortran_transliteration(self):
        for ice_type in (0, 1):
            q, rad, pres, presf, temp = self._random_inputs()
            ref = ref_lidar_optics(q, rad, pres, presf, temp, ice_type)
            # jcosp layout: level axis 0, subcolumn axis 1, point trailing.
            out = lidar_optics(
                *(jnp.asarray(np.transpose(q[..., i], (2, 1, 0)))
                  for i in range(4)),
                *(jnp.asarray(rad[..., i].T) for i in range(4)),
                jnp.asarray(pres.T), jnp.asarray(presf.T), jnp.asarray(temp.T),
                ice_type=ice_type)
            for name in ref:
                got = np.asarray(getattr(out, name))
                want = ref[name]
                if got.ndim == 3:      # (nlev, ncol, npoints)
                    want = np.transpose(want, (2, 1, 0))
                else:                  # (nlev, npoints) or (ncol, npoints)
                    want = want.T
                np.testing.assert_allclose(
                    got, want, rtol=1e-9, atol=1e-300,
                    err_msg=f"{name} ice_type={ice_type}")

    def test_molecular_profile_positive_and_decreasing_with_height(self):
        q, rad, pres, presf, temp = self._random_inputs(npoints=1)
        out = lidar_optics(
            *(jnp.asarray(np.transpose(q[..., i], (2, 1, 0))) for i in range(4)),
            *(jnp.asarray(rad[..., i].T) for i in range(4)),
            jnp.asarray(pres.T), jnp.asarray(presf.T), jnp.asarray(temp.T))
        beta = np.asarray(out.beta_mol)[:, 0]
        self.assertTrue(np.all(beta > 0.0))
        # TOA-first: index 0 is the top, so beta_mol increases along axis 0.
        self.assertTrue(np.all(np.diff(beta) > 0.0))
        # Molecular optical depth accumulates monotonically downward and its
        # 532-nm column value is the Rayleigh optical depth (~0.13-0.15).
        tau = np.asarray(out.tau_mol)[:, 0]
        self.assertTrue(np.all(np.diff(tau) > 0.0))
        self.assertGreater(tau[-1], 0.05)
        self.assertLess(tau[-1], 0.3)

    def test_clear_sky_reduces_to_molecular(self):
        q, rad, pres, presf, temp = self._random_inputs()
        zeros = jnp.zeros(np.transpose(q[..., 0], (2, 1, 0)).shape)
        out = lidar_optics(
            zeros, zeros, zeros, zeros,
            *(jnp.asarray(rad[..., i].T) for i in range(4)),
            jnp.asarray(pres.T), jnp.asarray(presf.T), jnp.asarray(temp.T))
        betatot = np.asarray(out.betatot)
        tautot = np.asarray(out.tautot)
        np.testing.assert_array_equal(
            betatot, np.broadcast_to(np.asarray(out.beta_mol)[:, None],
                                     betatot.shape))
        np.testing.assert_array_equal(
            tautot, np.broadcast_to(np.asarray(out.tau_mol)[:, None],
                                    tautot.shape))
        np.testing.assert_array_equal(np.asarray(out.tau_sfc_liq), 0.0)

    def test_effective_radius_cap(self):
        # Radii above 70 um are clipped, not rejected: same result as 70 um.
        q, rad, pres, presf, temp = self._random_inputs(npoints=1)
        args_tail = (jnp.asarray(pres.T), jnp.asarray(presf.T), jnp.asarray(temp.T))
        qs = [jnp.asarray(np.transpose(q[..., i], (2, 1, 0))) for i in range(4)]
        big = [jnp.full(rad[..., i].T.shape, 200e-6) for i in range(4)]
        cap = [jnp.full(rad[..., i].T.shape, config.LIDAR_MAX_RADIUS)
               for i in range(4)]
        out_big = lidar_optics(*qs, *big, *args_tail)
        out_cap = lidar_optics(*qs, *cap, *args_tail)
        np.testing.assert_allclose(np.asarray(out_big.betatot),
                                   np.asarray(out_cap.betatot), rtol=1e-12)

    def test_gradient_finite_and_nonzero(self):
        q, rad, pres, presf, temp = self._random_inputs(nlev=10, ncol=3,
                                                        npoints=2)
        # Keep some zero-radius (masked) volumes to exercise the hardening.
        qs = [jnp.asarray(np.transpose(q[..., i], (2, 1, 0))) for i in range(4)]
        rads = [jnp.asarray(rad[..., i].T) for i in range(4)]

        def total_beta(qliq):
            out = lidar_optics(qliq, qs[1], qs[2], qs[3], *rads,
                               jnp.asarray(pres.T), jnp.asarray(presf.T),
                               jnp.asarray(temp.T))
            return jnp.sum(out.betatot) + jnp.sum(out.tautot)

        g = jax.grad(total_beta)(qs[0])
        self.assertTrue(np.all(np.isfinite(np.asarray(g))))
        self.assertGreater(np.abs(np.asarray(g)).max(), 0.0)


if __name__ == "__main__":
    unittest.main()
